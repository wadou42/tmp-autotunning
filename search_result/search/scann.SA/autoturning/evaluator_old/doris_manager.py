# Author: whq
# Description: 使用时需注意一下几点：
# 0. 在编译脚本时需要 source /home/whq/dataset/doris/doris.env
# 1. 必须使用 bash build.sh --be --fe 成功编译之后才可使用这个脚本，该脚本无异常处理能力
# 2. 在load_data时, 脚本默认使用的是10G的数据, 但是tpch默认的并不支持10G, 需要自己修改tpch脚本(如果是使用我的doris文件，那已经修改过了)
# 3. 端口号并没有做修改，默认 9030 和 9050 , 也没有加修改逻辑
# 4. 最好不要从头下载doris的压缩包进行编译，非常耗时，并且有bug。可以cp -r /home/whq/dataset/doris
import warnings
warnings.warn("This code is the old version of doris manager. Please use managers in autotuning.manager", DeprecationWarning, stacklevel=2)

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


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.close()


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


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    try:
        pipe = os.popen(cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"


def distance(m1, m2):
    res = 0
    for i in range(len(m1)):
        if m1[i] != m2[i]:
            res += 1
    return res


def explore_mask(d, c):
    res0 = [0 for _ in range(d)]
    res = [res0[:]]
    while len(res) < c:
        res0 = [random.randint(0, 1) for _ in range(d)]
        tmp_res0 = [res0 for _ in range(len(res))]
        tmp_res0 = np.array(tmp_res0)
        tmp_res = np.array(res)
        dist_map = tmp_res ^ tmp_res0
        dist = np.sum(dist_map, 1)
        this_min = np.min(dist)
        min_dist = this_min
        for i in range(len(res0)):
            res0[i] ^= 1
            dist_map[:, i] ^= 1
            dist = np.sum(dist_map, 1)
            this_min = np.min(dist)
            if this_min < min_dist:
                res0[i] ^= 1
                dist_map[:, i] ^= 1
            else:
                min_dist = this_min
        print(min_dist)
        res.append(res0[:])
    return res


def get_opt_negation(flag):
    if '-fno-' not in flag:
        if '-fweb-' not in flag:
            return flag[:2] + 'no-' + flag[2:]


def get_original_optimization_status(level):
    res = execmd(f'gcc -Q --help=optimizers {level}').split('\n')
    res = [_ for _ in res if '-O' not in _ and '-' in _]
    ori_opt_state = [-1 for _ in OPT_LIST]
    for opt_idx in range(len(OPT_LIST)):
        default_open = False
        for res_idx in range(len(res)):
            if OPT_LIST[opt_idx] in res[res_idx]:
                default_open = '[enabled]' in res[res_idx]
                if default_open:
                    break
        ori_opt_state[opt_idx] = default_open
    for _ in ori_opt_state:
        assert ori_opt_state[_] != -1
    print(f'[get_original_optimization_status] The collected {level} default status are:')
    for idx in range(len(ori_opt_state)):
        print(f'[get_original_optimization_status] {OPT_LIST[idx]}: {ori_opt_state[idx]}')
    return ori_opt_state


def generate_compile_options(opt_state, original_opt_state):
    idx = 0
    options_to_specify = []
    while idx < len(opt_state):
        if opt_state[idx] != original_opt_state[idx]:
            if opt_state[idx]:
                options_to_specify.append(OPT_LIST[idx])
            else:
                options_to_specify.append(get_opt_negation(OPT_LIST[idx]))
        idx += 1
    return ' '.join(options_to_specify)


class Param:
    def __init__(self, name, t, lb=None, ub=None):
        self.name = name
        self.type = t
        self.lb = lb
        self.ub = ub

    def random_instance_str(self):
        if self.type == 'int':
            return "{" f"{self.name}: {random.randint(self.lb, self.ub)}" "}"
        if self.type == 'bool':
            return "{" f"{self.name}: '{random.randint(0, 1)}'" "}"


class SearchSpace:
    def __init__(self):
        self.params = [
            Param(name="flag_tree_vrp", t="bool"),
            Param(name="flag_inline_small_functions", t="bool"),
            Param(name="flag_tree_tail_merge", t="bool"),
            Param(name="flag_tree_slp_vectorize", t="bool"),
            Param(name="flag_tree_pre", t="bool"),
            Param(name="flag_tree_builtin_call_dce", t="bool"),
            Param(name="flag_finite_loops", t="bool"),
            Param(name="flag_optimize_sibling_calls", t="bool"),
            Param(name="flag_prefetch_loop_arrays", t="bool"),
            Param(name="flag_schedule_insns", t="bool"),
            Param(name="flag_schedule_insns_after_reload", t="bool"),
            Param(name="flag_unroll_loops", t="bool"),
            Param(name="param_max_inline_insns_auto", t="int", lb=10, ub=190),
            Param(name="param_inline_unit_growth", t="int", lb=30, ub=300),
            Param(name="param_max_inline_recursive_depth_auto", t="int", lb=4, ub=8),
            Param(name="param_large_function_insns", t="int", lb=1100, ub=3100),
            Param(name="param_large_function_insns", t="int", lb=20, ub=100),
            Param(name="param_large_unit_insns", t="int", lb=6000, ub=16000),
            Param(name="param_max_unrolled_insns", t="int", lb=100, ub=2000),
            Param(name="param_max_average_unrolled_insns", t="int", lb=10, ub=800),
            Param(name="param_max_unroll_times", t="int", lb=1, ub=64),
            Param(name="param_prefetch_latency", t="int", lb=100, ub=2000),
            Param(name="param_simultaneous_prefetches", t="int", lb=1, ub=80),
            Param(name="param_min_insn_to_prefetch_ratio", t="int", lb=1, ub=30),
            Param(name="param_prefetch_min_insn_to_mem_ratio", t="int", lb=1, ub=30),
        ]

    def random_instance(self):
        return f'Args: [{", ".join([_.random_instance_str() for _ in self.params])}]'


def multi_thread_new_conf_generation(request: Queue, done: Queue, target_size: int):
    sp = SearchSpace()
    while request.qsize() != 0 or done.qsize() != target_size:
        old_line = request.get()
        start_idx = old_line.find('Args: [{')
        end_idx = old_line.find('}], CodeRegionHash:')
        end_idx += len('}]')
        new_line = old_line[:start_idx] + sp.random_instance() + old_line[end_idx:]
        done.put(new_line)


def generate_param_yaml_from_existing(new_yaml_file, template_yaml_file):
    f = open(template_yaml_file)
    old_lines = f.readlines()
    old_lines = [_ if not _.endswith('\n') else _[:-1] for _ in old_lines]
    f.close()

    request = Queue()
    done = Queue()
    for old_line in old_lines:
        request.put(old_line)

    ps = []
    for _ in range(os.cpu_count()):
        p = Process(target=multi_thread_new_conf_generation, args=(request, done, len(old_lines), ), daemon=True)
        p.start()
        ps.append(p)
    while done.qsize() != len(old_lines):
        time.sleep(10)
    new_lines = []
    while done.qsize() != 0:
        new_lines.append(done.get() + '\n')

    for p in ps:
        p.terminate()

    # sp = SearchSpace()
    #
    # old_lines = [_ if not _.endswith('\n') else _[:-1] for _ in old_lines]
    # new_lines = []
    # for old_line in old_lines:
    #     start_idx = old_line.find('Args: [{')
    #     end_idx = old_line.find('}], CodeRegionHash:')
    #     end_idx += len('}]')
    #     new_line = old_line[:start_idx] + sp.random_instance() + old_line[end_idx:]
    #     new_lines.append(new_line + '\n')

    f = open(new_yaml_file, 'a+')
    f.writelines(new_lines)
    f.close()


def tar_yaml_file(yaml_file):
    tar_file = f"{'.'.join(yaml_file.split('.')[:-1])}.tar.bz2"
    tar_cmd = f'tar -I pbzip2 -cvf {tar_file} {yaml_file} ; rm -f {yaml_file} &'
    os.system(tar_cmd)


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
        process = None
        try:
            process = subprocess.run(f"bash build.sh --be --fe -j{jobs}", cwd=self.build_dir, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, shell=True, env=env, text=True, timeout=60*60*2)
        except subprocess.TimeoutExpired:
            pass
        # process = subprocess.run(f"bash build.sh --be --fe -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)
        # process = subprocess.run(f"bash build.sh --be -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)

        if process is not None and process.returncode != 0:
            print("Build failed")
            return -1

        for d in [self.fe_meta_dir, self.be_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.modify_config()
        return 0

    def build_be(self, opt_config: str = " -g -O3 ", jobs: int = 128) -> int:
        """
        This function is only used for quickly debug.
        :param opt_config:
        :param jobs:
        :return:
        """
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

        process = None
        try:
            process = subprocess.run(f"bash build.sh --be -j{jobs}", cwd=self.build_dir, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, shell=True, env=env, text=True, timeout=60*60*2)
        except subprocess.TimeoutExpired:
            pass

        # process = subprocess.run(f"bash build.sh --be --fe -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)
        # process = subprocess.run(f"bash build.sh --be -j{jobs}", cwd=self.build_dir, shell=True, env=env, text=True)

        if process is not None and process.returncode != 0:
            print("Build failed")
            return -1

        for d in [self.fe_meta_dir, self.be_data_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.modify_config_be()
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

    def modify_config_be(self):
        be_conf_path = self.build_dir / "output/be/conf/be.conf"

        be_config_lines = [
            "# Network and storage settings",
            f"priority_networks = {self.be_ip}/32",
            f"storage_root_path = {self.be_data_dir}"
        ]

        with open(be_conf_path, 'a') as f:
            f.write("\n" + "\n".join(be_config_lines) + "\n")

        return 0
        pass

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

    def start_doris_be(self):
        """
        This function is only for debug.
        :return:
        """
        be_start = os.path.join(self.build_dir, "output/be/bin/start_be.sh")
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

    def test(self, interval: int = 0, task: str = "hot") -> tuple[list[float], dict[str, list[float]]]:
        if task == "hot":
            return self.hot_test(interval)
        elif task == "cold":
            return self.cold_test(interval)
        else:
            raise ValueError("Invalid task. Use 'hot' or 'cold'.")

    def hot_test(self, interval: int) -> tuple[list[float], dict[str, list[float]]]:
        """gen_data(if not exists) -> load_data -> """
        """热测试：两次测试之间不停止服务，无间隔"""
        build_flag = False
        for _ in range(3):
            if self.init_doris() == 0:
                build_flag = True
                break

        if not build_flag:
            return [], {_: [] for _ in REGISTERED_QUETIES}

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
            return [], {_: [] for _ in REGISTERED_QUETIES}

        # avg_time = sum(successful_runs) / len(successful_runs)
        # avg_time_per_query = {_: sum(total_hot_query_perf[_]) / len(total_hot_query_perf[_]) for _ in
        #                       total_hot_query_perf}
        # print(f"[hot_test] avg_time_per_query={avg_time_per_query}")

        return successful_runs, total_hot_query_perf

    def cold_test(self, interval: int) -> tuple[list[float], dict[str, list[float]]]:
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
            return [], {_: [] for _ in REGISTERED_QUETIES}

        # avg_cold_time = sum(successful_runs) / len(successful_runs)
        # avg_cold_time_per_query = {_: sum(total_cold_time_per_query[_]) / len(total_cold_time_per_query[_]) for _ in
        #                            total_cold_time_per_query}
        return successful_runs, total_cold_time_per_query

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


def parallel_build(manager, manager_base, opt, opt_base=" -O3 "):
    with ThreadPoolExecutor(max_workers=4) as executor:
        future1 = executor.submit(manager.build, opt_config=opt)
        future2 = executor.submit(manager_base.build, opt_config=opt_base)

        result1 = future1.result()
        result2 = future2.result()
        return result1, result2


def build(manager: DorisManager, opt, jobs):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future1 = executor.submit(manager.build, opt_config=opt, jobs=jobs)
        result1 = future1.result()
        return result1


def main():
    doris_manager = DorisManager(build_dir=args.doris_turning_path, repeat=20, fe_port=args.fe_port,
                                 be_port=args.be_port)
    doris_O3_manager = DorisManager(build_dir=args.doris_O3_path, repeat=20, fe_port=args.fe_port,
                                    be_port=args.be_port)

    iter_num = args.iter_start
    mask = explore_mask(len(IMPORTANT_OPT_IDX), args.max_iter)
    O3_opt_status = get_original_optimization_status('-O3')
    new_important_opt_status = None
    O3_built = False
    while iter_num < args.max_iter:
        options = None
        new_yaml_file = None
        if args.type == "O3":
            options = f'-O3'
        if args.type == "option":
            new_important_opt_status = mask[iter_num]
            new_opt_status = O3_opt_status[:]
            for idx in range(len(IMPORTANT_OPT_IDX)):
                new_opt_status[IMPORTANT_OPT_IDX[idx]] = new_important_opt_status[idx]
            options = generate_compile_options(new_opt_status, O3_opt_status)
            options = f'-O3 {options}'
        if args.type == "function":
            new_yaml_file = f'{os.getcwd()}/{args.AI4C_yaml_dir}/{iter_num}.yamls'
            generate_param_yaml_from_existing(new_yaml_file, args.template_file)
            options = f"-g -O3 -fplugin={args.AI4C_plagin_lib} " \
                      f"-fplugin-arg-coarse_option_tuning_plugin_gcc12-yaml={args.AI4C_option_file} " \
                      f"-fplugin-arg-coarse_option_tuning_plugin_gcc12-autotune={new_yaml_file}"
        assert options is not None
        if args.type == 'O3' and not O3_built:
            build(doris_manager, options, 128)
        if args.type != 'O3':
            build(doris_manager, options, 80)
        if args.mode == 'mix' and not O3_built:
            build(doris_O3_manager, '-O3', 128)

        opt_perf, opt_perf_per_query = doris_manager.test(interval=0, task='hot')
        O3_perf = -1
        O3_perf_per_query = {_: -1 for _ in REGISTERED_QUETIES}
        if args.mode == 'mix':
            O3_perf, O3_perf_per_query = doris_O3_manager.test(interval=0, task='hot')

        print(f"[main] perf_per_query={opt_perf_per_query}")
        acc_rate = (O3_perf / opt_perf - 1) * 100
        put_file_content(args.perf_recorder,
                         f'{iter_num}: opt_perf={opt_perf}, O3_perf={O3_perf}, acc_rate={acc_rate}%\n')
        for query in opt_perf_per_query:
            query_perf_file = f"{args.query_perf_dir}/{query}.txt"
            acc_rate = (O3_perf_per_query[query] / opt_perf_per_query[query] - 1) * 100
            put_file_content(query_perf_file,
                             f'{iter_num}: opt_perf={opt_perf_per_query[query]}, O3_perf={O3_perf_per_query[query]}, acc_rate={acc_rate}%\n')
        if args.type == "option":
            put_file_content(args.opt_recorder, f'{iter_num}: {options}\n')
            put_file_content(args.opt_encode_recorder, f'{iter_num}: {new_important_opt_status}\n')
            assert new_important_opt_status is not None

        if args.type == 'function':
            assert new_yaml_file is not None
            tar_yaml_file(new_yaml_file)
        iter_num += 1


"""
Usage:
python turn_doris.py \
--doris_turning_path /home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning \
--doris_O3_path /home/scy/2025.04.17.doris.exceeds.O3/src/doris.O3 \
--type option \
--iter_start 94 \
--mode mix | tee log.log

python turn_doris.py \
--doris_turning_path /home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning \
--doris_O3_path /home/scy/2025.04.17.doris.exceeds.O3/src/doris.O3 \
--type function \
--mode mix

python turn_doris.py \
--doris_turning_path /home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning \
--type O3 \
--max_iter 50 \
--mode single
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='doris build and test script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # arguments about doris build and test
    parser.add_argument('--doris_turning_path',
                        type=str,
                        default=None,
                        help='The path of doris compiled with specified options')
    parser.add_argument('--doris_O3_path',
                        type=str,
                        default=None,
                        help='The path of doris compiled with specified options')
    parser.add_argument('--fe_port',
                        type=int,
                        default=9030,
                        help='The port used by frontend.')
    parser.add_argument('--be_port',
                        type=int,
                        default=9050,
                        help='The port used by backend.')

    # arguments about output
    parser.add_argument('--opt_recorder',
                        type=str,
                        default="opt.txt",
                        help='The optimization recorder file, record important optimization only.')
    parser.add_argument('--opt_encode_recorder',
                        type=str,
                        default="opt.encode.txt",
                        help='The optimization recorder file, record important optimization only.')
    parser.add_argument('--perf_recorder',
                        type=str,
                        default="perf.txt",
                        help='The perf recorder file.')
    parser.add_argument('--query_perf_dir',
                        type=str,
                        default="query_perf",
                        help='The directory path that store performance record of each query in search interation.')
    parser.add_argument('--clean',
                        type=bool,
                        default=True,
                        help='Whether clean the old result.')

    # argument about turning
    parser.add_argument('--max_iter',
                        type=int,
                        default=2000,
                        help='The number of max iterations.')
    parser.add_argument('--iter_start',
                        type=int,
                        default=0,
                        help='The start iteration point.')
    parser.add_argument('--type',
                        type=str,
                        default="option",
                        help='Turn type.')
    parser.add_argument('--mode',
                        type=str,
                        default="mix",
                        help='Turn mode.')

    # arguments about AI4C project
    parser.add_argument('--AI4C_plagin_lib',
                        type=str,
                        help="The path of plagin .so file generated by AI4C.",
                        default="/usr/local/lib/python3.9/site-packages/ai4c/lib/coarse_option_tuning_plugin_gcc12.so")
    parser.add_argument('--AI4C_option_file',
                        type=str,
                        help="The path of option file generated by AI4C.",
                        default="/usr/local/lib/python3.9/site-packages/ai4c/autotuner/yaml/coarse_options.yaml")
    parser.add_argument('--AI4C_yaml_dir',
                        type=str,
                        help="The directory path for input yaml file.",
                        default="yaml")
    parser.add_argument('--template_file',
                        type=str,
                        help="The template yaml optimization configuration file.",
                        default="function.yaml")

    args = parser.parse_args()

    assert args.doris_turning_path is not None
    assert args.fe_port is not None
    assert args.be_port is not None
    assert args.type in ['option', 'function', 'O3', 'mix']
    assert args.mode in ['mix', 'single']
    if args.type == 'O3':
        assert args.mode != 'mix'

    if args.clean:
        os.system(f"rm -f {args.opt_recorder} {args.opt_encode_recorder} {args.perf_recorder}")
        os.system(f"rm -rf {args.query_perf_dir}")
    os.system(f'mkdir -p {args.query_perf_dir}')

    main()
