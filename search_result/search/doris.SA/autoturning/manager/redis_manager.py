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


# def parse_exec_time(stdouts: list[str]) -> dict[str, float]:
#     """
#     Parse the number of GET and SET operations per second from the test output.

#     :param stdouts: Standard output from the test execution
#     """
#     requests_per_second = {"get": FLOAT_MAX / 16, "set": FLOAT_MAX / 16}
#     key = ""
#     for stdout in stdouts:
#         key = "get" if "GET" in stdout else ("set" if "SET" in stdout else key)
#         if "requests per second" in stdout:
#             requests_per_second[key] = 1000000 / float(stdout.split()[0])
#     return requests_per_second

def parse_exec_time(stderrs: list[str]) -> float:
    tot: float = 0.0
    for out in stderrs:
        if out.startswith("real"):
            out = out.replace("real\t", "")
            nums = re.findall(r"\d*\.?\d+", out)
            assert len(nums) == 2, "Expect %dm %ds format"
            secs = float(nums[0])*60+float(nums[1])
            tot += secs
    return tot


class RedisManager:
    def __init__(
        self,
        redis_home: str,
        enable_gperftools: bool = False,
        num_repeat: int = 1,
        port: int = 6379,
        redis_start_core: str = 'None',      # 新增参数
        redis_bench_core: str = 'None',      # 新增参数
    ):
        """
        Initialize the RedisManager class.
        """
        self.redis_home = Path(redis_home)
        self.port = port
        self.enable_gperftools = enable_gperftools
        self.redis_build_dir = self.redis_home / "redis"
        self.num_repeat = num_repeat
        self.redis_start_core = redis_start_core
        self.redis_bench_core = redis_bench_core

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
        make_env["OPTIMIZATION"] = f"-g {opt_config}"

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
                if perf < 0 or perf > 1000000:
                    perf = self.run_single_test()

            if perf < 0 or perf > 1000000:
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
                ["./redis-cli", "FLUSHALL"],
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
            CPUPROFILE_FREQUENCY=1000 CPUPROFILESIGNAL=12 ./redis-server --port {self.port};
            """
        else:
            start_commands = f"""
                rm -f dump.rdb
                taskset -c {self.redis_start_core} numactl -N 3 -m 3 ./redis-server {self.redis_build_dir}/redis.conf --port {self.port};
            """
            if self.redis_start_core != 'None':
                start_commands = f"""
                    rm -f dump.rdb
                    taskset -c {self.redis_start_core} numactl -N 3 -m 3 ./redis-server {self.redis_build_dir}/redis.conf --port {self.port};
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
        if self.enable_gperftools:
            test_commands = f"""
                (killall -12 servers/redis-server; \
                time taskset -c {self.redis_bench_core} numactl -N 3 -m 3 ./redis-benchmark -h 127.0.0.1 -p {self.port} -d 3 -n 1000000 -P 100 -c 20 -q;
                killall -12 servers/redis-server);
                ./redis-cli -h 127.0.0.1 -p {self.port} shutdown nosave;
            """
        else:
            if self.redis_bench_core != 'None':
                test_commands = f"""
                    time taskset -c {self.redis_bench_core} numactl -N 3 -m 3 ./redis-benchmark -h 127.0.0.1 -p {self.port} -d 3 -n 1000000 -P 100 -c 20;
                    ./redis-cli -h 127.0.0.1 -p {self.port} shutdown nosave;
                    """
            else:
                raise ValueError("redis_bench_core should not be None")

        timeout_sec = 30 * 60  # Set timeout to 30 minutes
        warmup_command = f"""./redis-cli CONFIG SET appendonly no
                            ./redis-cli CONFIG SET save ""
                            time taskset -c {self.redis_bench_core} numactl -N 3 -m 3 ./redis-benchmark -h 127.0.0.1 -p {self.port} -n 10000 -r 100000000 -c 80 -t set,get;
                            ./redis-cli FLUSHALL"""
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

            stderrs = p.stderr.split("\n")
            perf = parse_exec_time(stderrs)
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
        client = redis.StrictRedis(host=host, port=self.port)
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
                    for conn in proc.net_connections():
                        if (
                            conn.status == psutil.CONN_LISTEN
                            and conn.laddr.port == self.port
                        ):
                            proc.kill()
                            return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print("Failed to kill Redis process.")


if __name__ == "__main__":
    redis_manager = RedisManager(
        redis_home="/home/whq/dataset/redis/redis",
        enable_gperftools=False,
        num_repeat=1,
        port=6379,
        redis_start_core='311',
        redis_bench_core='319',
    )
    redis_manager.clean()
    current_time = time.time()
    redis_manager.build(opt_config="-g -O3")
    print(f"Build time: {time.time() - current_time:.4f} seconds", flush=True)

    result: float = 0
    tot: list[float] = []
    for _ in range(100):
        current_time = time.time()
        result = redis_manager.test()
        tot.append(result)
        print(f"[{_}] perf: {result:.4f} us", flush=True)
    from statistics import median

    print(f"[+] median perf: {median(tot):.4f} us", flush=True)
    print("=================== Done ===================", flush=True)
