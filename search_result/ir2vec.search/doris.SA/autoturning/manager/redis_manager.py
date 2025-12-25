import os
import queue
import re
import shutil
import subprocess
import psutil
import threading
import time
import redis
import statistics
from pathlib import Path

FLOAT_MAX = float("inf")
MIN_EXEC_TIME = 1.0  # redis 测试的最小执行时间，低于该值视为异常

"""get,set: 1846027.1200 requests per second"""
"""flot 2364911.5000 requests per second"""
"""flot O3 2149604.7500"""


def parse_exec_time(stdout: list[str]) -> float:
    # print("stderrs:", stdout)
    rps_get_set: float = 0.0
    for out in stdout:
        if "requests per second" in out:
            nums = re.findall(r"\d*\.?\d+", out)
            assert len(nums) == 1, "Expect get,set: xxxx requests per second format"
            rps_get_set = float(nums[0])
    return rps_get_set


class RedisManager:
    def __init__(
        self,
        redis_home: str,
        enable_gperftools: bool = False,
        num_repeat: int = 1,
        redis_start_core: str = "None",  # 新增参数
        redis_bench_core: str = "None",  # 新增参数
    ):
        """
        Initialize the RedisManager class.
        """
        self.redis_home = Path(redis_home)
        self.enable_gperftools = enable_gperftools
        self.redis_build_dir = self.redis_home / "redis"
        self.num_repeat = num_repeat
        self.redis_start_core = redis_start_core
        self.redis_bench_core = redis_bench_core
        self.socket_path = self.redis_build_dir / "redis.sock"  # 新增：socket 文件路径，与 conf 同级


    def build(self, opt_config: str = "-g -O3"):
        """
        Compile Redis source code.

        :param opt_config: Redis compilation options, defaults to "-g -O3"
        """
        self.clean()

        redis_tar_path = self.redis_home / "redis-6.0.20.tar.gz"
        assert redis_tar_path.exists()

        # Untar Redis
        subprocess.run(
            ["tar", "-xzf", str(redis_tar_path)], cwd=str(self.redis_home), check=True
        )
        os.rename(str(self.redis_home / "redis-6.0.20"), str(self.redis_build_dir))

        # Compile Redis
        make_env = os.environ.copy()
        make_env["OPTIMIZATION"] = f"-g -O3 -flto=auto {opt_config}"

        result = subprocess.run(
            ["make", "-j32"],
            cwd=str(self.redis_build_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=make_env,
            check=False,
            text=True,
            timeout=10 * 60,
        )
        if not "It's a good idea to run 'make test'" in result.stdout:
            return -1
        subprocess.run(
            ["make", "PREFIX=" + str(self.redis_build_dir / "output"), "install"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.redis_build_dir),
            check=True,
            timeout=10 * 60,
        )

        # Move redis-server to a separate directory for easier testing with gperftools
        redis_server_src = self.redis_build_dir / "output/bin/redis-server"
        redis_server_dest_dir = self.redis_build_dir / "output/bin/servers"
        redis_server_dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(redis_server_src), str(redis_server_dest_dir))
        shutil.copy(
            str(self.redis_home / "project-func-info-databaase.json"),
            str(redis_server_dest_dir),
        )

        return 0

    def run_single_test(self) -> float:
        """
        Test Redis performance as per the task requirements: SET with requests=5000000, clients=80 and GET with requests=5000000, clients=80.
        The task requires the metric to be requests per second (RPS). Here, the average time per request (in microseconds) is returned. To get RPS, divide 1000000 by the return value.
        During testing, abnormal values for get_time and set_time were observed. The cause is unknown, so a threshold is set. If abnormal values are detected, the test is rerun.
        """
        self.clean_cache()
        result_queue = queue.Queue()
        redis_thread = threading.Thread(target=self.start_redis)
        benchmark_thread = threading.Thread(
            target=self.run_benchmark, args=(result_queue,)
        )

        redis_thread.start()
        benchmark_thread.start()

        redis_thread.join()
        benchmark_thread.join()

        perf = FLOAT_MAX / 32

        if result_queue.queue:
            perf = result_queue.get()

        if self.enable_gperftools:
            self.analysis()

        return perf

    def test(self, num_repeat: int = -1) -> float:
        """
        Run the Redis performance test multiple times and return the average time per request (in microseconds).

        :param num_repeat: Number of test repetitions, defaults to 1
        """
        if num_repeat < 0:
            num_repeat = self.num_repeat

        perfs = []
        attempts = 3
        for i in range(num_repeat):
            perf = -1

            for j in range(attempts):
                if perf < 0 or perf > 100000000:
                    perf = self.run_single_test()

            if perf < 0 or perf > 100000000:
                return FLOAT_MAX

            perfs.append(perf)

        return statistics.median(perfs) if perfs else FLOAT_MAX

    def clean(self):
        """
        Stop the Redis server and clean the Redis build directory.
        """
        self.kill_redis()

        if self.redis_build_dir.exists():
            shutil.rmtree(self.redis_build_dir, ignore_errors=True)

        report_path = self.redis_home / "report.txt"
        if report_path.exists():
            os.remove(report_path)

    def clean_cache(self):
        """
        Clear Redis cache by deleting the dump.rdb file and executing the FLUSHALL command. (不过看起来作用不是太大...)
        """
        redis_dump_file = self.redis_build_dir / "output/bin/servers/dump.rdb"
        if os.path.exists(redis_dump_file):
            os.remove(redis_dump_file)

        # FLUSHALL
        if os.path.exists(self.redis_build_dir / "output/bin/redis-cli"):
            subprocess.run(
                ["./redis-cli", "-s", str(self.socket_path), "FLUSHALL"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.redis_build_dir / "output/bin"),
            )

    def start_redis(self):
        """
        Start the Redis server.
        """
        server_path = self.redis_build_dir / "output/bin/servers"

        if self.enable_gperftools:
            start_commands = f"""
            rm -f main.prof*;
            rm -f dump.rdb
            LD_PRELOAD=/usr/local/lib/libprofiler.so.0 CPUPROFILE=./main.prof \
            CPUPROFILE_FREQUENCY=1000 CPUPROFILESIGNAL=12 ./redis-server {self.redis_build_dir}/redis.conf;  # 修改：移除 --port，使用 conf 中的 unixsocket
            """
        else:
            start_commands = f"""
                rm -f dump.rdb
                numactl -N 3 -m 3 taskset -c {self.redis_start_core}  ./redis-server {self.redis_build_dir}/redis.conf;  # 修改：移除 --port，使用 conf 中的 unixsocket
            """
            if self.redis_start_core != "None":
                start_commands = f"""
                    rm -f dump.rdb
                    numactl -N 3 -m 3 taskset -c {self.redis_start_core}  ./redis-server {self.redis_build_dir}/redis.conf;  # 修改：移除 --port，使用 conf 中的 unixsocket
                """
            else:
                raise ValueError("redis_start_core should not be None")

        timeout_sec = 30 * 60  # Single test duration should be less than 30 minutes
        try:
            p = subprocess.run(
                args=start_commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                cwd=server_path,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as e:
            print("TimeoutExpired Exception:", e)
            self.kill_redis()
            return

    def run_benchmark(self, result_queue: queue.Queue):
        """
        Test Redis server performance using redis-benchmark and return the average time per request (in microseconds).
        Test case: SET with requests=5000000, clients=80 and GET with requests=5000000, clients=80.
        """
        # Wait for the server to start...
        if not self.wait_redis_start():
            print("Redis server did not start in time.")
            self.kill_redis()
            result_queue.put(FLOAT_MAX)
            return

        benchmark_path = self.redis_build_dir / "output/bin"
        if self.enable_gperftools is None:
            raise ValueError("enable_gperftools should not be None")
        
        if self.enable_gperftools:
            test_commands = f"""
                (killall -12 servers/redis-server; \
                numactl -N 3 -m 3 taskset -c {self.redis_bench_core}  ./redis-benchmark -s {self.socket_path} -d 3 -n 100000000 -P 100 -c 80 -q get,set; \
                killall -12 servers/redis-server);
                ./redis-cli -s {self.socket_path} shutdown nosave;
            """
        else:
            test_commands = f"""
                numactl -N 3 -m 3 taskset -c {self.redis_bench_core}  ./redis-benchmark -s {self.socket_path} -d 3 -n 100000000 -P 100 -c 80 -q get,set;  # 修改：使用 -s socket
                ./redis-cli -s {self.socket_path} shutdown nosave;
                """

        timeout_sec = 30 * 60  # Set timeout to 30 minutes
        warmup_command = f"""./redis-cli -s {self.socket_path} CONFIG SET appendonly no
                            ./redis-cli -s {self.socket_path} CONFIG SET save ""
                            numactl -N 3 -m 3 taskset -c {self.redis_bench_core}  ./redis-benchmark -s {self.socket_path} -d 3 -n 50000000 -P 100 -c 80 -q get,set;  # 修改：使用 -s socket
                            ./redis-cli -s {self.socket_path} FLUSHALL"""
        subprocess.run(
            warmup_command,
            shell=True,
            text=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=benchmark_path,
            timeout=timeout_sec,
        )

        try:
            p = subprocess.run(
                test_commands,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True,
                cwd=benchmark_path,
                timeout=timeout_sec,
            )

            stdouts = p.stdout.split("\n")
            perf = parse_exec_time(stdouts)
            print(f"perf: {perf:.4f}", flush=True)
            if perf > MIN_EXEC_TIME:
                result_queue.put(perf)
        except Exception as e:
            print("An error occurred:", e)
            self.kill_redis()

    def analysis(self):
        """
        Generate the test report.
        """
        analysis_path = self.redis_build_dir / "output/bin/servers"

        analysis_commands = f"""
            if [ -s main.prof.0 ]; then
                pprof --lines redis-server main.prof.0 > tmp.txt ;
                rm main.prof.0;
            fi;
        """

        subprocess.run(
            analysis_commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=analysis_path,
        )

        # gperftools cannot specify append mode, so tmp.txt is used to prevent overwriting
        tmp_report_path = self.redis_build_dir / "output/bin/servers/tmp.txt"
        report_path = self.redis_build_dir / "output/bin/servers/report.txt"

        # with open(tmp_report_path, "r") as f1, open(report_path, "a") as f2:
        #     shutil.copyfileobj(f1, f2)

        with open(tmp_report_path, "r") as f1, open(report_path, "a") as f2:
            first_line = f1.readline()  # 读取 f1 的第一行
            f2.write(first_line)  # 将第一行追加到 f2

    def wait_redis_start(self, host: str = "localhost", timeout: int = 60):
        """
        Wait for the Redis server to start. Returns True if the server starts within the timeout period, otherwise False.
        """
        time.sleep(2)
        client = redis.StrictRedis(unix_socket_path=str(self.socket_path))  # 修改：使用 unix socket
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client.ping()
                return True
            except redis.ConnectionError:
                time.sleep(1)

        print("Redis server did not start within the timeout period.")
        return False

    def kill_redis(self):
        """
        Terminate the Redis process if it is running.
        """
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] == "redis-server":
                try:
                    # 检查 socket 文件是否存在（可选，替代端口检查）
                    if os.path.exists(str(self.socket_path)):
                        proc.kill()
                        return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print("Failed to kill Redis process.")


if __name__ == "__main__":
    redis_manager = RedisManager(
        redis_home="/home/whq/dataset/redis/redis",
        enable_gperftools=False,
        num_repeat=1,
        redis_start_core="311",
        redis_bench_core="319",
    )

    redis_manager.build(opt_config="-O3 -flto=12")
    print("Build completed.", flush=True)
    perfs = []
    for _ in range(100):
        result: float = 0
        result = redis_manager.test()
        print(f"Test [{_+1}/100] result: {result:.4f} us", flush=True)
        perfs.append(result)
    median_perf = statistics.median(perfs)
    print(f"Median Test result over 100 runs: {median_perf:.4f} us", flush=True)

