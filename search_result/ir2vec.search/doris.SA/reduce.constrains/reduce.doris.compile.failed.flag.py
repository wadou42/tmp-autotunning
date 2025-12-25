import argparse
import os
import random
import requests
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Process
from pathlib import Path

import numpy as np
import pymysql


OPT_LIST = ['-faggressive-loop-optimizations', '-falign-functions', '-falign-jumps', '-falign-labels', '-falign-loops',
            '-fallocation-dce', '-fasynchronous-unwind-tables', '-fauto-inc-dec', '-fbit-tests', '-fbranch-count-reg',
            '-fcaller-saves', '-fcode-hoisting', '-fcombine-stack-adjustments', '-fcompare-elim', '-fcprop-registers',
            '-fcrossjumping', '-fcse-follow-jumps', '-fcse-skip-blocks', '-fdce', '-fdefer-pop', '-fdelayed-branch',
            '-fdevirtualize', '-fdevirtualize-speculatively', '-fdse', '-fearly-inlining', '-fexpensive-optimizations',
            '-fforward-propagate', '-ffp-int-builtin-inexact', '-ffunction-cse', '-fgcse', '-fgcse-after-reload',
            '-fgcse-lm', '-fguess-branch-probability', '-fhoist-adjacent-loads', '-fif-conversion', '-fif-conversion2',
            '-findirect-inlining', '-finline', '-finline-atomics', '-finline-functions',
            '-finline-functions-called-once', '-finline-small-functions', '-fipa-bit-cp', '-fipa-cp', '-fipa-cp-clone',
            '-fipa-icf', '-fipa-icf-functions', '-fipa-icf-variables', '-fipa-modref', '-fipa-profile',
            '-fipa-pure-const', '-fipa-ra', '-fipa-reference', '-fipa-reference-addressable', '-fipa-sra',
            '-fipa-stack-alignment', '-fipa-strict-aliasing', '-fipa-vrp', '-fira-hoist-pressure',
            '-fira-share-save-slots', '-fira-share-spill-slots', '-fisolate-erroneous-paths-dereference', '-fivopts',
            '-fjump-tables', '-flifetime-dse', '-floop-interchange', '-floop-unroll-and-jam', '-flra-remat',
            '-fmath-errno', '-fmerge-constants', '-fmove-loop-invariants', '-fmove-loop-stores', '-fomit-frame-pointer',
            '-foptimize-sibling-calls', '-foptimize-strlen', '-fpartial-inlining', '-fpeel-loops', '-fpeephole',
            '-fpeephole2', '-fplt', '-fpredictive-commoning', '-fprintf-return-value', '-free', '-freg-struct-return',
            '-freorder-blocks', '-freorder-blocks-and-partition', '-freorder-functions', '-frerun-cse-after-loop',
            '-fsched-critical-path-heuristic', '-fsched-dep-count-heuristic', '-fsched-group-heuristic',
            '-fsched-interblock', '-fsched-last-insn-heuristic', '-fsched-pressure', '-fsched-rank-heuristic',
            '-fsched-spec', '-fsched-spec-insn-heuristic', '-fsched-stalled-insns-dep', '-fschedule-fusion',
            '-fschedule-insns', '-fschedule-insns2', '-fsection-anchors', '-fsemantic-interposition', '-fshort-enums',
            '-fshrink-wrap', '-fshrink-wrap-separate', '-fsigned-zeros', '-fsplit-ivs-in-unroller', '-fsplit-loops',
            '-fsplit-paths', '-fsplit-wide-types', '-fssa-backprop', '-fssa-phiopt', '-fstdarg-opt', '-fstore-merging',
            '-fstrict-aliasing', '-fstrict-overflow', '-fstrict-volatile-bitfields', '-fthread-jumps',
            '-ftoplevel-reorder', '-ftrapping-math', '-ftree-bit-ccp', '-ftree-builtin-call-dce', '-ftree-ccp',
            '-ftree-ch', '-ftree-coalesce-vars', '-ftree-copy-prop', '-ftree-dce', '-ftree-dominator-opts',
            '-ftree-dse', '-ftree-forwprop', '-ftree-fre', '-ftree-loop-distribute-patterns',
            '-ftree-loop-distribution', '-ftree-loop-im', '-ftree-loop-ivcanon', '-ftree-loop-optimize',
            '-ftree-loop-vectorize', '-ftree-partial-pre', '-ftree-phiprop', '-ftree-pre', '-ftree-pta',
            '-ftree-reassoc', '-ftree-scev-cprop', '-ftree-sink', '-ftree-slp-vectorize', '-ftree-slsr', '-ftree-sra',
            '-ftree-switch-conversion', '-ftree-tail-merge', '-ftree-ter', '-ftree-vrp',
            '-funroll-completely-grow-size', '-funswitch-loops', '-funwind-tables', '-fversion-loops-for-strides']
IMPORTANT_OPT_IDX = [37, 124, 32, 136, 41, 39, 70, 112, 140, 137, 25, 68, 59, 69, 78, 143, 115, 61, 60, 116, 150, 139,
                     146, 62, 18, 10, 9, 135, 51, 36, 8, 58, 50]
REGISTERED_QUETIES = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15',
                      'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22']


def rm_dir(dir_path: Path):
    """删除文件夹下的所有文件，保留父目录"""
    if dir_path.exists():
        shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)


def get_folder_size(folder_path):
    total_size = 0
    for entry in os.scandir(folder_path):
        if entry.is_file():
            total_size += entry.stat().st_size  # file size, byte
        elif entry.is_dir():
            total_size += get_folder_size(entry.path)
    return total_size


class DorisManager:
    def __init__(self,
                 build_dir: str,
                 fe_ip: str = "173.52.1.2",
                 be_ip: str = "173.52.1.2",
                 fe_port: int = 9030,
                 be_port: int = 9050,
                 user: str = "root",
                 repeat: int = 20,
                 scale_factor: str = "10"
                 ):
        """
        这里能更改的只有build_dir.... 因为没有设置更改端口的方法
        程序的编译至多两个程序一起编译，而测试的时间时低于编译时间的，我认为没有必要再并行了
        """
        self.build_dir = Path(build_dir)
        self.fe_ip = fe_ip
        self.be_ip = be_ip
        self.fe_port = fe_port
        self.be_port = be_port
        self.user = user
        self.repeat = repeat
        self.scale_factor = scale_factor
        self.fe_meta_dir = self.build_dir / "data/fe/meta"
        self.be_data_dir = self.build_dir / "data/be/storage"
        self.tpch_home = self.build_dir / "tools/tpch-tools/bin"

    def replace_bad_config(self, opt_config: str) -> str:
        # 当-ftoplevel-reorder关闭时  -fsection-anchors也必须关闭
        if "-fno-toplevel-reorder" in opt_config:
            opt_config = opt_config.replace("-fsection-anchors", "-fno-section-anchors")

        # doris 需要金禁用 -finline-atomics 否则会在最后一步链接时出问题
        opt_config = opt_config.replace("-finline-atomics", "")
        return opt_config

    def build(self, opt_config: str = " -g -O3 ", jobs: int = 128) -> int:
        opt_config = self.replace_bad_config(opt_config=opt_config)

        self.stop_services()
        self.clean()

        env = os.environ.copy()
        env["DISABLE_JAVA_CHECK_STYLE"] = "ON"
        env["EXTRA_CXX_FLAGS"] = ' -g ' + opt_config
        env["DORIS_TOOLCHAIN"] = "gcc"
        env["LDFLAGS"] = "-Wl,--as-needed -Wl,--push-state -Wl,-Bstatic -latomic -Wl,--pop-state"
        # jobs = min(int(0.8 * os.cpu_count()), 80)
        # jobs = 128

        # 三个分别对应无输出编译be/fe和有输出编译be/fe以及有输出编译be
        process = subprocess.run(f"bash build.sh --be --fe -j{jobs}", cwd=self.build_dir, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, shell=True, env=env, text=True)
        # process = subprocess.run(f"bash build.sh --be --fe -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)
        # process = subprocess.run(f"bash build.sh --be -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)

        if process.returncode != 0:
            print("Build failed")
            return -1

        for d in [self.fe_meta_dir, self.be_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.modify_config()
        return 0

    def modify_config(self) -> int:
        """修改 Doris FE 和 BE 的配置文件"""

        fe_conf_path = self.build_dir / "output/fe/conf/fe.conf"
        be_conf_path = self.build_dir / "output/be/conf/be.conf"

        fe_config_lines = [
            "# Network and metadata settings",
            f"priority_networks = {self.fe_ip}/32",
            f"meta_dir = {self.fe_meta_dir}"
        ]

        be_config_lines = [
            "# Network and storage settings",
            f"priority_networks = {self.be_ip}/32",
            f"storage_root_path = {self.be_data_dir}"
        ]

        with open(fe_conf_path, 'a') as f:
            f.write("\n" + "\n".join(fe_config_lines) + "\n")

        with open(be_conf_path, 'a') as f:
            f.write("\n" + "\n".join(be_config_lines) + "\n")

        return 0

    def start_doris(self):
        """启动 Doris FE 和 BE"""
        self.stop_services()

        be_start = os.path.join(self.build_dir, "output/be/bin/start_be.sh")
        fe_start = os.path.join(self.build_dir, "output/fe/bin/start_fe.sh")

        start_fe = subprocess.run(["bash", fe_start, "--daemon"], cwd=self.build_dir, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=30)

        # 等待FE启动成功再启动BE
        if start_fe.returncode != 0 or self.check_fe_ready() == -1:
            return -1

        start_be = subprocess.run(["bash", be_start, "--daemon"], cwd=self.build_dir, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=30)

        if start_be.returncode != 0:
            return -1
        return 0

    def register_be_to_fe(self):
        """注册 Doris BE到FE
        mysql -h 173.52.1.2 -P 9030 -u root -p
        ALTER SYSTEM ADD BACKEND '173.52.1.2:9050';"""
        conn = None
        try:
            conn = pymysql.connect(
                host="127.0.0.1",
                port=self.fe_port,
                user="root",
                password="",
                connect_timeout=30
            )
            cursor = conn.cursor()

            check_sql = f"SHOW BACKENDS;"
            cursor.execute(check_sql)
            backends = cursor.fetchall()

            # 检查当前 BE 是否已在列表中
            be_address = self.be_ip
            already_register_be_to_feed = any(be_address in str(backend) for backend in backends)
            if already_register_be_to_feed:
                return 0

            register_be_to_fe_sql = f"ALTER SYSTEM ADD BACKEND '{self.be_ip}:{self.be_port}';"
            cursor.execute(register_be_to_fe_sql)
            conn.commit()

        except pymysql.Error as e:
            print(f"操作失败: {e}")
        finally:
            if conn:
                conn.close()

    def check_fe_ready(self, max_attempt: int = 30, interval: int = 2) -> int:
        """检查FE是否准备就绪"""
        attempt = 0
        while attempt < max_attempt:
            try:
                resp = requests.get(f"http://{self.fe_ip}:8030/api/bootstrap", timeout=5)
                if '"msg":"success"' in resp.text:
                    return 0
            except Exception as http_err:
                time.sleep(interval)
                attempt += 1
        return -1

    def check_cluster_health(self):
        """
        通过MySQL协议验证服务状态
        """
        conn = None
        try:
            conn = pymysql.connect(
                host="127.0.0.1",
                port=self.fe_port,
                user="root",
                password="",
                database="information_schema",
                connect_timeout=30
            )
            with conn.cursor() as cursor:
                cursor.execute("SHOW FRONTENDS")
                fe_status = cursor.fetchall()

                cursor.execute("SHOW BACKENDS")
                be_status = cursor.fetchall()

                print("fe_status is:" + str(fe_status))
                print("be_status is:" + str(be_status))

                # 本机测试只有一个fe节点和一个be节点，但是保证代码的健壮性，还是检查所有节点
                # fe 节点存活信息在第9列，be 节点存活信息在第10列
                fe_alive = all(row[8] == "true" for row in fe_status)
                be_alive = all(row[9] == "true" for row in be_status)

                if fe_alive and be_alive:
                    return 0
                elif not fe_alive:
                    return -1
                else:
                    return -2
        except pymysql.Error as e:
            return -3
        finally:
            if 'conn' in locals() and conn:
                conn.close()

    def wait_cluster_ready(self, max_attempts: int = 30, interval: int = 10) -> int:
        """等待 Doris FE 和 BE 准备就绪"""
        time.sleep(interval * 3)
        attempt = 0
        return_code = None
        while attempt < max_attempts:
            return_code = self.check_cluster_health()
            if return_code == 0:
                return 0
            else:
                time.sleep(interval)
                attempt += 1
        if return_code == -1:
            print("FE is not ready")
        elif return_code == -2:
            print("BE is not ready")
        elif return_code == -3:
            print("mysql connection error")
        return -1

    def clean(self):
        """
        清理编译的输出/加载的数据/编译的缓存
        """
        self.clean_data()
        subprocess.run(["bash", "build.sh", "--clean"], cwd=self.build_dir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        output_dir = self.build_dir / "output"
        rm_dir(output_dir)

    # 删除数据be/fe
    def clean_data(self):
        rm_dir(self.fe_meta_dir)
        rm_dir(self.be_data_dir)

    def stop_services(self):
        """Stop frontend and backend services"""
        # 先关闭 FE，再关闭 BE
        fe_stop = os.path.join(self.build_dir, "output/fe/bin/stop_fe.sh")
        be_stop = os.path.join(self.build_dir, "output/be/bin/stop_be.sh")

        if os.path.exists(be_stop):
            subprocess.run(["bash", be_stop], cwd=self.build_dir, check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60 * 60)
        if os.path.exists(fe_stop):
            subprocess.run(["bash", fe_stop], cwd=self.build_dir, check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60 * 60)

    def init_doris(self):
        """stop_services -> start_doris -> register_be_to_fe -> wait_cluster_ready"""
        self.stop_services()

        if self.start_doris() == -1:
            print("Doris start failed", flush=True)
            self.stop_services()
            return -1

        self.register_be_to_fe()

        if self.wait_cluster_ready() != 0:
            self.stop_services()
            print("Doris clusters are not alive", flush=True)
            return -1

        if get_folder_size(self.be_data_dir) < 0.3 * float(self.scale_factor) * 1024 * 1024:
            self.load_data()
        return 0

    def test(self, interval: int = 0, task: str = "hot") -> tuple[float, dict]:
        if task == "hot":
            return self.hot_test(interval)
        elif task == "cold":
            return self.cold_test(interval)
        else:
            raise ValueError("Invalid task. Use 'hot' or 'cold'.")

    def hot_test(self, interval: int) -> tuple[float, dict]:
        """gen_data(if not exists) -> load_data -> """
        """热测试：两次测试之间不停止服务，无间隔"""
        build_flag = False
        for _ in range(3):
            if self.init_doris() == 0:
                build_flag = True
                break

        if not build_flag:
            return -1, {_: -1 for _ in REGISTERED_QUETIES}

        successful_runs = []
        total_hot_query_perf = dict()

        self.run_benchmark(interval=0, repeat=6)  # warm up, 经验值
        for repeat_time in range(self.repeat):
            cold_time, hot_time, cold_time_per_query, hot_time_per_query = self.run_benchmark(interval=interval,
                                                                                              repeat=1)
            if hot_time != -1:
                successful_runs.append(hot_time)
                for query in hot_time_per_query:
                    total_hot_query_perf.setdefault(query, []).extend(hot_time_per_query[query])
            else:
                print("Error: 热测试失败")

        self.stop_services()

        if not successful_runs:
            print("Error: 热测试全部失败")
            return -1, {_: -1 for _ in REGISTERED_QUETIES}

        avg_time = sum(successful_runs) / len(successful_runs)
        avg_time_per_query = {_: sum(total_hot_query_perf[_]) / len(total_hot_query_perf[_]) for _ in
                              total_hot_query_perf}
        print(f"[hot_test] avg_time_per_query={avg_time_per_query}")

        return avg_time, avg_time_per_query

    def cold_test(self, interval: int) -> tuple[float, dict]:
        """冷测试：两次测试之间停止服务，有间隔"""
        successful_runs = []
        total_cold_time_per_query = dict()

        for _ in range(self.repeat):
            self.stop_services()
            if self.init_doris() == -1:
                print("Error: 冷测试初始化失败")
                continue

            cold_time, hot_time, cold_time_per_query, hot_time_per_query = self.run_benchmark(interval=interval,
                                                                                              repeat=1)
            self.stop_services()

            if cold_time != -1:
                successful_runs.append(cold_time)
                for query in cold_time_per_query:
                    total_cold_time_per_query.setdefault(query, []).extend(cold_time_per_query[query])
            else:
                print("Error: 冷测试失败")

        if not successful_runs:
            print("Error: 冷测试全部失败")
            return -1, {_: -1 for _ in REGISTERED_QUETIES}

        avg_cold_time = sum(successful_runs) / len(successful_runs)
        avg_cold_time_per_query = {_: sum(total_cold_time_per_query[_]) / len(total_cold_time_per_query[_]) for _ in
                                   total_cold_time_per_query}
        return avg_cold_time, avg_cold_time_per_query

    # build dbgen -> generate data
    def load_data(self):
        tpch_tools_path = self.tpch_home / "TPC-H_Tools_v3.0.0"
        tpch_data_path = self.tpch_home / "tpch-data"

        if not tpch_tools_path.exists():
            print("build tpch dbgen")
            subprocess.run(["bash", "build-tpch-dbgen.sh"], cwd=self.tpch_home, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=60 * 60)

        if not tpch_data_path.exists():
            print("generating tpch data...")
            subprocess.run(["bash", "gen-tpch-data.sh", "-s", self.scale_factor], cwd=self.tpch_home,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60 * 60)

        attempt, max_attempt = 0, 5
        while attempt < max_attempt:
            try:
                subprocess.run(["bash", "create-tpch-tables.sh", "-s", self.scale_factor], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, check=True, cwd=self.tpch_home, timeout=60 * 60)
                break
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                attempt += 1
                time.sleep(5)

        subprocess.run(["bash", "load-tpch-data.sh"], stdout=subprocess.DEVNULL, cwd=self.tpch_home, timeout=60 * 60)

    def run_benchmark(self, interval: float = 0, repeat: int = 3) -> tuple[float, float, dict, dict]:
        total_cold_time, total_hot_time = 0, 0
        total_cold_query_perf = dict()
        total_hot_query_perf = dict()
        for _ in range(repeat):
            try:
                process = subprocess.run(["bash", "run-tpch-queries.sh", "-s", self.scale_factor],
                                         stdout=subprocess.PIPE, cwd=self.tpch_home, text=True, timeout=60 * 60)
            except subprocess.CalledProcessError as e:
                print(f"Error: {e}")
                print("Error: subprocess failed")
                return -1, -1, dict(), dict()

            cold_time, hot_time, cold_query_perf, hot_query_perf = self.parse_result(process.stdout)
            if -1 in (cold_time, hot_time):
                return -1, -1, dict(), dict()
            print(f"Cold time: {cold_time}, Hot time: {hot_time}", flush=True)
            total_cold_time += cold_time
            total_hot_time += hot_time

            for query in cold_query_perf:
                total_cold_query_perf.setdefault(query, []).append(cold_query_perf[query])

            for query in hot_query_perf:
                total_hot_query_perf.setdefault(query, []).append(hot_query_perf[query])
            time.sleep(interval)
        print(f"[run_benchmark] total_cold_query_perf={total_cold_query_perf}")
        print(f"[run_benchmark] total_hot_query_perf={total_hot_query_perf}")
        return total_cold_time / repeat, total_hot_time / repeat, total_cold_query_perf, total_hot_query_perf

    def parse_result(self, result: str) -> tuple[float, float, dict, dict]:
        stdouts = result.splitlines()

        cold_query_perf = dict()
        hot_query_perf = dict()

        cold_time, hot_time = 0, 0
        for stdout in stdouts:
            stdout_info = stdout.split('\t')
            potential_query_prefix = stdout_info[0]
            if potential_query_prefix in REGISTERED_QUETIES:
                query_name = stdout_info[0]
                cold_perf = -1
                hot_perf = -1
                if stdout_info[1] != '':
                    cold_perf = int(stdout_info[1])
                if stdout_info[-1] != '':
                    hot_perf = int(stdout_info[-1])
                assert query_name not in cold_query_perf
                assert query_name not in hot_query_perf
                cold_query_perf.setdefault(query_name, cold_perf)
                hot_query_perf.setdefault(query_name, hot_perf)

            if "Total cold run time" in stdout:
                cold_time = float(stdout.split(":")[1].replace("ms", "").strip())
            elif "Total hot run time" in stdout:
                hot_time = float(stdout.split(":")[1].replace("ms", "").strip())
        if cold_time == 0 or hot_time == 0:
            return -1, -1, dict(), dict()
        print(f"[parse_result] cold_query_perf={cold_query_perf}")
        print(f"[parse_result] hot_query_perf={hot_query_perf}")
        return cold_time, hot_time, cold_query_perf, hot_query_perf


def reduceFlags(optList, doris_build_path, add_part: str):
    level = optList[0]
    del optList[0]
    start = 0
    step = len(optList) / 2
    step = int(step)
    end = len(optList) if start + step > len(optList) else start + step
    while step >= 1:
        while start < len(optList):
            print('[reduceFlags] [len=' + str(len(optList)) + ', s=' + str(start) + ', e=' + str(end) + ', step=' + str(step) + ']')
            print(optList)

            tmpOpt = optList[:start] + optList[end:]
            doris_manager = DorisManager(build_dir=doris_build_path)
            passed = doris_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part, jobs=128)

            test_time = 0
            avg_time = None
            while test_time < 5:
                avg_time, avg_time_per_query = doris_manager.test(interval=0, task='hot')
                if avg_time != -1:
                    break
                test_time += 1

            if avg_time == -1:
                print('[reduceFlags] failed')
                optList = tmpOpt[:]
                end = len(optList) if start + step > len(optList) else start + step
            else:
                print('[reduceFlags] pass')
                start = end
                end = len(optList) if start + step > len(optList) else start + step
        start = 0
        step = step / 2
        step = int(step)
        end = len(optList) if start + step > len(optList) else start + step
    optList.insert(0, level)
    return optList


def reduceMore(optList, doris_build_path, add_part):
    level = optList[0]
    del optList[0]
    inx = 0
    while inx < len(optList):
        print('[len=' + str(len(optList)) + ', inx=' + str(inx) + ']')
        print(optList)

        tmpOpt = optList[:inx] + optList[inx + 1:]

        doris_manager = DorisManager(build_dir=doris_build_path)
        passed = doris_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part, jobs=128)

        test_time = 0
        avg_time = None
        while test_time < 5:
            avg_time, avg_time_per_query = doris_manager.test(interval=0, task='hot')
            if avg_time != -1:
                break
            test_time += 1

        if avg_time == -1:
            print('[reduceMore] failed')
            optList = tmpOpt[:]
        else:
            print('[reduceMore] pass')
            inx += 1
    optList.insert(0, level)
    return optList


if __name__ == '__main__':
    doris_build_path: str = '/home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning'
    optlist = [
        "-O3",
        "-O3 -fno-aggressive-loop-optimizations -fno-align-functions -fallow-store-data-races -fassociative-math -fno-asynchronous-unwind-tables -fno-auto-inc-dec -fno-branch-count-reg -fno-code-hoisting -fno-crossjumping -fcx-limited-range -fno-dce -fno-dse -fno-early-inlining -fno-expensive-optimizations -ffinite-math-only -fno-fp-int-builtin-inexact -fno-function-cse -fno-gcse -fno-gcse-after-reload -fno-gcse-lm -fno-if-conversion2 -fno-inline-functions -fno-inline-small-functions -fno-ipa-bit-cp -fno-ipa-icf-functions -fno-ipa-profile -fno-ipa-pure-const -fno-ipa-reference -fno-ipa-stack-alignment -fno-ipa-strict-aliasing -fno-ipa-vrp -fno-ira-hoist-pressure -fno-ira-share-spill-slots -fno-isolate-erroneous-paths-dereference -fno-ivopts -fno-jump-tables -fno-lifetime-dse -fno-loop-interchange -fno-lra-remat -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-partial-inlining -fno-peephole -fno-predictive-commoning -freciprocal-math -fno-ree -fno-reorder-blocks -fno-reorder-functions -fno-rerun-cse-after-loop -fno-sched-dep-count-heuristic -fno-sched-interblock -fno-sched-last-insn-heuristic -fno-sched-pressure -fno-sched-rank-heuristic -fno-sched-spec -fno-sched-spec-insn-heuristic -fno-schedule-insns -fno-schedule-insns2 -fno-section-anchors -fno-short-enums -fno-shrink-wrap -fno-shrink-wrap-separate -fno-split-ivs-in-unroller -fno-split-loops -fno-split-paths -fno-split-wide-types -fno-store-merging -fno-strict-aliasing -fno-strict-volatile-bitfields -fno-tree-ch -fno-tree-dse -fno-tree-fre -fno-tree-loop-distribute-patterns -fno-tree-loop-distribution -fno-tree-loop-im -fno-tree-loop-vectorize -fno-tree-partial-pre -fno-tree-pre -fno-tree-pta -fno-tree-reassoc -fno-tree-sink -fno-tree-slp-vectorize -fno-tree-slsr -fno-tree-sra -funsafe-math-optimizations -fno-unswitch-loops -fno-unwind-tables -frtti -flive-patching -fhandle-exceptions -fexceptions -fdelete-null-pointer-checks",
        "-O3 -fno-align-functions -fno-align-labels -fno-align-loops -fno-allocation-dce -fallow-store-data-races -fno-asynchronous-unwind-tables -fno-combine-stack-adjustments -fno-crossjumping -fno-dce -fno-devirtualize -ffinite-math-only -fno-function-cse -fno-gcse-lm -fno-guess-branch-probability -fno-if-conversion2 -fno-indirect-inlining -fno-inline-functions -fno-inline-functions-called-once -fno-inline-small-functions -fno-ipa-cp-clone -fno-ipa-icf -fno-ipa-icf-functions -fno-ipa-pure-const -fno-ipa-ra -fno-ipa-reference -fno-ipa-stack-alignment -fno-ipa-strict-aliasing -fno-ipa-vrp -fno-ira-share-save-slots -fno-loop-interchange -fno-move-loop-stores -fno-optimize-sibling-calls -fno-optimize-strlen -fno-plt -fno-printf-return-value -fno-sched-dep-count-heuristic -fno-sched-group-heuristic -fno-sched-interblock -fno-sched-rank-heuristic -fno-sched-spec-insn-heuristic -fno-sched-stalled-insns-dep -fno-schedule-fusion -fno-schedule-insns -fno-schedule-insns2 -fno-section-anchors -fno-short-enums -fno-split-ivs-in-unroller -fno-split-loops -fno-split-wide-types -fno-ssa-phiopt -fno-stdarg-opt -fno-store-merging -fno-strict-volatile-bitfields -fno-toplevel-reorder -fno-tree-bit-ccp -fno-tree-builtin-call-dce -fno-tree-ch -fno-tree-coalesce-vars -fno-tree-copy-prop -fno-tree-dce -fno-tree-dse -fno-tree-loop-distribution -fno-tree-loop-optimize -fno-tree-loop-vectorize -fno-tree-partial-pre -fno-tree-pta -fno-tree-reassoc -fno-tree-sink -fno-tree-slsr -fno-tree-switch-conversion -fno-unroll-completely-grow-size -funsafe-math-optimizations -fnothrow-opt -frtti -fstrict-enums -fno-threadsafe-statics -ftree-loop-if-convert -fhandle-exceptions -fexceptions",
    ]

    add_parts = [""] * len(optlist)

    # optlist = ["-O3"] * 100

    f = open('result.txt', "a+")

    for idx in range(len(optlist)):
        print(f'[main] {idx}-th opt')

        doris_manager = DorisManager(build_dir=doris_build_path, fe_port=9030, be_port=9050)
        passed = doris_manager.build(opt_config=' -g ' + optlist[idx], jobs=128)

        test_time = 0
        avg_time = None
        while test_time < 5:
            avg_time, avg_time_per_query = doris_manager.test(interval=0, task='hot')
            if avg_time != -1:
                break
            test_time += 1

        if avg_time != -1:
            f.write('success\n')
            f.flush()
            continue

        o = optlist[idx].split(' ')

        o = reduceFlags(o, doris_build_path, add_parts[idx])
        o = reduceMore(o, doris_build_path, add_parts[idx])

        f.write(' '.join(o) + '\n')
        f.flush()
    f.close()
