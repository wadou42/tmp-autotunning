import warnings
warnings.warn("This code is the old version of redis manager. Please use managers in autotuning.manager", DeprecationWarning, stacklevel=2)

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil
import redis

from autoturning.evaluator_old.redis.line_level_analysis import LineAnalysis




def parse_real_time(stderrs: list[str]) -> float:
    """
    Parse the time output from the Linux `time` command and return the time in seconds.

    :param stderrs: Output from the `time` command
    """
    secs = 0
    for stderr in stderrs:
        if stderr.startswith("real"):
            out = stderr.replace("real\t", "")
            nums = re.findall(r"\d*\.?\d+", out)
            assert len(nums) == 2, "Expected format: %dm %ds"
            
            secs = float(nums[0]) * 60 + float(nums[1])
    return secs


# Modified to return the average time per request in microseconds
def parse_exec_time(stdouts: list[str]) -> dict[str:float]:
    """
    Parse the number of GET and SET operations per second from the test output.

    :param stdouts: Standard output from the test execution
    """
    requests_per_second = {"get": FLOAT_MAX / 16, "set": FLOAT_MAX / 16}
    key = ""
    for stdout in stdouts:
        key = "get" if "GET" in stdout else ("set" if "SET" in stdout else key) 
        if "requests per second" in stdout:
            requests_per_second[key] = 1000000 / float(stdout.split()[0])
    
    return requests_per_second


FLOAT_MAX = float("inf")


class RedisManager:
    def __init__(self, redis_home:str=None, enable_gperftools:bool=False, port:int=6379):
        """
        Initialize the RedisManager class.

        :param redis_home: Redis installation directory, defaults to "/home/whq/dataset/redis"
        :param enable_gperftools: Whether to enable gperftools, defaults to False
        """
        
        self.redis_home =  Path(redis_home or "/home/whq/CodeRep/software/redis")
        self.port = port    
        self.enable_gperftools = enable_gperftools
        # Redis build directory
        self.redis_build_dir = self.redis_home / "redis"
        self.is_built = False

    def build(self, optimization:str="-g -O3"):
        """
        Compile Redis source code.

        :param optimization: Redis compilation options, defaults to "-g -O3"
        """
        self.clean() 

        redis_tar_path = self.redis_home / "redis-6.0.20.tar.gz"
        assert redis_tar_path.exists()
            
        # Untar Redis
        subprocess.run(["tar", "-xzf", str(redis_tar_path)], cwd=str(self.redis_home), check=True)
        os.rename(str(self.redis_home / "redis-6.0.20"), str(self.redis_build_dir))

        # Compile Redis
        make_env = os.environ.copy()
        make_env["OPTIMIZATION"] = f"-g {optimization}"
        make_env["CC"] = f"gcc"
        make_env["CXX"] = f"g++"

        print(self.redis_build_dir)

        subprocess.run(["make", "-j96", ">> log.log"], cwd=str(self.redis_build_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=make_env, check=True)
        subprocess.run(["make", "PREFIX=" + str(self.redis_build_dir / "output"), "install"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(self.redis_build_dir), check=True)

        # Move redis-server to a separate directory for easier testing with gperftools
        redis_server_src = self.redis_build_dir / "output/bin/redis-server"
        redis_server_dest_dir = self.redis_build_dir / "output/bin/servers"
        redis_server_dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(redis_server_src), str(redis_server_dest_dir))
        shutil.copy(str(self.redis_home / "project-func-info-databaase.json"), str(redis_server_dest_dir))
        
        self.is_built = True

    def test(self):
        """
        Test Redis performance as per the task requirements: SET with requests=5000000, clients=80 and GET with requests=5000000, clients=80.
        The task requires the metric to be requests per second (RPS). Here, the average time per request (in microseconds) is returned.
        To get RPS, divide 1000000 by the return value.
        During testing, abnormal values for get_time and set_time were observed.
        The cause is unknown, so a threshold is set. If abnormal values are detected, the test is rerun.
        """
        self.clean_cache()
        result_queue = queue.Queue()
        redis_thread = threading.Thread(target=self.start_redis)
        benchmark_thread = threading.Thread(target=self.run_benchmark,args=(result_queue,))
        
        redis_thread.start()
        benchmark_thread.start()

        redis_thread.join()
        benchmark_thread.join()
        
        get_time, set_time = FLOAT_MAX / 32, FLOAT_MAX / 32
        
        if result_queue.queue:
            item = result_queue.get()
            get_time = item[0]
            set_time = item[1]
        
        if get_time > 10 or set_time > 10:
            return self.test()
        
        if self.enable_gperftools:
            self.analysis()
            
        return get_time + set_time
    
    def clean(self):
        """
        Stop the Redis server and clean the Redis build directory.
        """
        self.kill_redis()  
        
        if self.redis_build_dir.exists():
            # assert str(self.redis_build_dir).startswith("/home/whq/CodeRep/software/")
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
            subprocess.run(["./redis-cli", "FLUSHALL"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        cwd=str(self.redis_build_dir / "output/bin"))

    def start_redis(self):
        """
        Start the Redis server.
        """
        server_path = self.redis_build_dir / "output/bin/servers" 
        
        if self.enable_gperftools :
            start_commands = f"""
            rm -f main.prof*;
            rm -f dump.rdb
            LD_PRELOAD=/usr/local/lib/libprofiler.so.0 CPUPROFILE=./main.prof \
            CPUPROFILE_FREQUENCY=1000 CPUPROFILESIGNAL=12 ./redis-server --port {self.port};
            """
        else:
            start_commands = f"""
                rm -f dump.rdb
                ./redis-server --port {self.port};
            """
        
        timeout_sec = 30 * 60 # Single test duration should be less than 30 minutes
        try:
            p = subprocess.run(
                args = start_commands,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True, 
                text=True,  
                cwd=server_path,
                timeout=timeout_sec  
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
            result_queue.put((FLOAT_MAX, FLOAT_MAX))
            return
        
        benchmark_path = self.redis_build_dir / "output/bin"
        if self.enable_gperftools:
            test_commands = f"""
                (killall -12 servers/redis-server &&\
                ./redis-benchmark -h 127.0.0.1 -p {self.port} -n 5000000 -r 100000000 -c 80 -t get &&\
                ./redis-benchmark -h 127.0.0.1 -p {self.port} -n 5000000 -r 100000000 -c 80 -t set && \
                killall -12 servers/redis-server);
                
                ./redis-cli -h 127.0.0.1 -p {self.port} shutdown;
            """
        else:
            test_commands = f"""
                ./redis-benchmark -h 127.0.0.1 -p {self.port} -n 5000000 -r 100000000 -c 80 -t get &&\
                ./redis-benchmark -h 127.0.0.1 -p {self.port} -n 5000000 -r 100000000 -c 80 -t set && \
                ./redis-cli -h 127.0.0.1 -p {self.port} shutdown;
                """
        
        timeout_sec = 30 * 60  # Set timeout to 30 minutes

        try:
            p = subprocess.run(
                test_commands, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                shell=True, 
                text=True,
                cwd=benchmark_path,
                timeout=timeout_sec
            )
            
            stdouts = p.stdout.split("\n")
            get_time, set_time = parse_exec_time(stdouts).values() 
            get_time, set_time = float(get_time), float(set_time)
            print(f"get_time: {get_time}, set_time: {set_time}")
            if 0 not in [get_time, set_time]:
                result_queue.put((get_time, set_time))
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
        
        subprocess.run(analysis_commands, 
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       shell=True, 
                       cwd=analysis_path)
        
        # gperftools cannot specify append mode, so tmp.txt is used to prevent overwriting
        tmp_report_path = self.redis_build_dir / "output/bin/servers/tmp.txt"
        report_path = self.redis_build_dir / "output/bin/servers/report.txt"

        with open(tmp_report_path, "r") as f1, open(report_path, "a") as f2:
            shutil.copyfileobj(f1, f2)

    def wait_redis_start(self, host:str='localhost', timeout:int=60):
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

    def get_ports_by_pid(self, pid):
        ports = []
        try:
            process = psutil.Process(pid)
            for conn in psutil.net_connections(kind='inet'):
                if conn.pid == pid:
                    if conn.laddr:
                        ports.append(conn.laddr.port)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return ports

    def kill_redis(self):
        """
        Terminate the Redis process if it is running.
        """
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == "redis-server":
                ports = self.get_ports_by_pid(proc.info['pid'])
                if self.port in ports:
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        print("Redis process no longer exists.")
                    except psutil.AccessDenied:
                        print("Permission denied: Unable to kill Redis process.")
                    return

    def analysis_report(management1: 'RedisManager', management2: 'RedisManager'):
        """
        Analyze the performance reports of two RedisManager instances, 
        where management1 is the baseline instance (O3) and management2 is the comparison instance.

        :param management1: Baseline instance
        :param management2: Comparison instance

        :return: Returns the performance reports of the two instances in the format [(key, value1, value2), ...], 
            where key is the unique ID determined by the file path and function line number, 
            value1 is the [self_time, total_time] of the baseline instance, and value2 is the [self_time, total_time] of the comparison instance.
        """
        # path1 = management1.redis_build_dir / "output/bin/servers"
        # path2 = management2.redis_build_dir / "output/bin/servers"

        path1 = "/home/whq/CodeRep/software/redisO3/redis/output/bin/servers"
        path2 = "/home/whq/CodeRep/software/redis/redis/output/bin/servers"

        analysis1 = LineAnalysis(base_path=path1, program_name="redis-server")
        analysis2 = LineAnalysis(base_path=path2, program_name="redis-server")
        result1 = analysis1.parse_report()
        result2 = analysis2.parse_report()
        for key in result1:
            result1[key].extend([0,0] if key not in result2 else result2[key])
        return result1

    def analysis_report_single(self, template_json_file):
        analysis1 = LineAnalysis(base_path=f'{self.redis_build_dir}/output/bin/servers', program_name="redis-server")
        result1 = analysis1.parse_report()
        return result1


# Test RedisManager
if __name__ == "__main__":
    redis_manager = RedisManager(redis_home="/home/scy/2025.02.13.attempt.fuction.tuning/src/redis", enable_gperftools=True, port=50009)
    redis_manager_base = RedisManager(redis_home="/home/scy/2025.02.13.attempt.fuction.tuning/src/redisO3", enable_gperftools=True, port=50008)

    redis_manager.clean()
    redis_manager_base.clean()

    redis_manager.build(optimization="-g -O3")
    redis_manager_base.build(optimization="-g -O0")
    perf = redis_manager.test()
    perf_base = redis_manager_base.test()

    acc = perf_base / perf

    print('acc=' + str(acc))
    
    # 如果启用gperftools，则使用RedisManager.analysis_report方法
    analysis_report = RedisManager.analysis_report(redis_manager_base, redis_manager)
    for a in analysis_report:
        print(a)
