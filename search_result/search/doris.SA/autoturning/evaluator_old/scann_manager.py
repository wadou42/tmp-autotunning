import warnings
warnings.warn("This code is the old version of doris manager. Please use managers in autotuning.manager", DeprecationWarning, stacklevel=2)

from concurrent.futures import ThreadPoolExecutor
import csv
import os
import subprocess
from typing import Dict, Optional, Union

import concurrent

# TODO 编译时线程控制
# TODO 不重新执行O3


class SCANNManager:
    def __init__(
        self,
        build_dir: str,
        env: str,
        ann_dir: str,
        repeat: int = 1,
        datasets: list[str] = None,
    ):
        self.build_dir = build_dir
        self.env = env
        self.ann_dir = ann_dir
        self.repeat = repeat
        self.datasets = datasets

    @staticmethod
    def transform_opt(opt: str):
        """
        bazel 中每一个cxxopt只能指定一个选项， 所以需要将多个选项拆分成多个--cxxopt
        """
        # opt = SCANNManager.replace_bad_config(opt)
        opt_list = opt.split(" ")
        copt_list = [f"--copt='{x}' " for x in opt_list if x]
        cxxopt_list = [f"--cxxopt='{x}' " for x in opt_list if x]
        return " ".join(copt_list) + " ".join(cxxopt_list)
    
    @staticmethod
    def replace_bad_config(opt_config:str) -> str:
        #如果启用"-flive-patching" 所有的-fipa都要禁用
        if "-flive-patching" in opt_config:
            opt_list = opt_config.split(" ")
            opt_list = [opt for opt in opt_list if not "-fipa" in opt]
            opt_config = " ".join(opt_list)
        
        # 有些优化选项是不存在  有些是会导致编译失败
        bad_config = ["-fsingle-precision-constant", "-fpack-struct", "-fno-rtti", "-fno-live-patching", "-fno-stack-protector-all", "-fno-stack-protector-explicit", "-fno-stack-protector-strong", "-gstatement-fno-rontiers", "-fno-no-threadsafe-statics"]
        for config in bad_config:
            opt_config = opt_config.replace(config, '')
        # 如果-ftoplevel-reorder是关闭的，-fsection-anchors必须关闭
        if "-fno-toplevel-reorder" in opt_config:
            opt_config = opt_config.replace("-fsection-anchors", "-fno-section-anchors")
        return opt_config

    def run_py_command_in_env(
        self,
        command_list: Union[list[str], str],
        env: str = None,
        cwd: str = None,
        silence: bool = True,
    ):
        """
        Run a command in a specific conda environment.
        :param command_list: The command to run, either as a string or a list of strings.
        :param env: The name of the conda environment to use.
        :param cwd: The working directory to run the command in.
        :param silence: If True, suppress output.
        """
        if silence:
            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
        else:
            stdout, stderr = None, None

        cwd = cwd or self.build_dir
        env = env or self.env

        if isinstance(command_list, str):
            command_list = [command_list]

        joined_commands = " && ".join(command_list)
        # print(joined_commands)
        finally_command = f"""conda run -n {env} bash -c "{joined_commands}" """

        p = subprocess.run(
            finally_command, stderr=stderr, stdout=stdout, shell=True, cwd=cwd
        )

        return p.returncode

    def build(self, opt_config: str = "-g -O3"):
        if self.clean() != 0:
            print("Clean failed, please check the environment.")
            return -1

        bazel_opt = self.transform_opt(
            opt_config + " -foptimize-sibling-calls -finline"
        )
        build_command = (
            "bazel clean && "
            "CC=gcc bazel build "
            "--cxxopt='-g' "
            "--copt='-g' "
            f"{bazel_opt} "
            "--cxxopt='-std=c++17' "
            "--copt=-fsized-deallocation "
            "--copt=-w "
            "--copt=-march=armv8.2-a+lse+sve+f64mm "
            "--cxxopt=-march=armv8.2-a+lse+sve+f64mm "
            "--copt=-msve-vector-bits=256 "
            "--cxxopt=-msve-vector-bits=256 "
            "--local_cpu_resources=48 "
            ":build_pip_pkg "
        )

        build_pip_pkg_command = "bazel-bin/build_pip_pkg"

        install_command = f"pip3 install scann-1.2.10-cp39-cp39-linux_aarch64.whl"

        check_command1 = f"pip list | grep scann"
        
        # NOTE 这个command2 有待商榷，并不一定有效
        check_command2 = f"python -c 'import scann'"

        command_list = [
            build_command,
            build_pip_pkg_command,
            install_command,
            check_command1,
            check_command2
        ]
        # command_list = [build_command, build_pip_pkg_command, install_command, check_command]

        for command in command_list:
            return_code = self.run_py_command_in_env(command)
            print(f'{command}: {return_code}')
            if return_code != 0:
                return -1
        return 0

    def clean(self):
        clean_command = (
            "bazel clean && "
            "rm -rf scann-1.2.10-cp39-cp39-linux_aarch64.whl && "
            "pip3 uninstall scann -y"
        )
        check_after = "pip3 list | grep scann"
        self.run_py_command_in_env(clean_command)
        scann_flag = self.run_py_command_in_env(check_after)
        return not scann_flag

    def run_benchmark(self) -> Optional[Dict[str, int]]:
        mapping = {}
        for dataset in self.datasets:
            result_file = f"{dataset}_results.csv"

            run_command = f"python run.py --force --threads 64 --algorithm scann --dataset {dataset} --local"
            get_result_command = (
                f"python data_export.py --datasets {dataset} --output {result_file}"
            )

            command_list = [run_command, get_result_command]
            for command in command_list:
                if self.run_py_command_in_env(command, cwd=self.ann_dir) != 0:
                    return -1

            qps = self.extract_qps_from_csv(os.path.join(self.ann_dir, result_file))
            if len(qps) == 0:
                continue
            mapping.update(
                self.extract_qps_from_csv(os.path.join(self.ann_dir, result_file))
            )
        return mapping

    @staticmethod
    def extract_qps_from_csv(file_path: str) -> dict[str, int]:
        mapping = {}
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if not all(col in reader.fieldnames for col in ["filename", "qps"]):
                raise ValueError("CSV文件必须包含'filename'和'qps'列")

            for row in reader:
                if float(row["k-nn"]) < 0.5:
                    return mapping
                filename = row["filename"]
                qps = float(row["qps"])  # 确保qps是数字
                mapping[filename] = qps
        return mapping

    @staticmethod
    def caculate_qps_acc(
        qps_dict: dict[str, int], qps_dict_base: dict[str, int]
    ) -> float:
        """
        计算qps加速比的均值
        """
        acc = 0
        if not (isinstance(qps_dict, dict) and isinstance(qps_dict_base, dict)):
            return -1
        if len(qps_dict_base) != len(qps_dict):
            return -1
        for key in qps_dict:
            acc += qps_dict[key] / qps_dict_base[key]
            print(
                f"flags / base = {qps_dict[key]} / {qps_dict_base[key]} = {qps_dict[key] / qps_dict_base[key]}",
                flush=True,
            )
        return acc / len(qps_dict)

    def test(self, num_repeats=1):
        result = self.run_benchmark()
        if not result or result == -1:
            return float('inf')
        if result==0:
            return float('inf')
        
        return result

    @staticmethod
    def test_together(manager1, manager2, flag1: str = None, flag2: str = None):
        """
        可以同时测试-O3和-flags
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交构建任务
            build1 = executor.submit(manager1.build, opt_config=f" {flag1} ")
            build2 = executor.submit(
                manager2.build,
                opt_config=f" {flag2} "
                # opt_config=" -O1 -fno-branch-count-reg -fcombine-stack-adjustments -fno-compare-elim -fcprop-registers -fdefer-pop -fforward-propagate -fno-guess-branch-probability -fif-conversion -fno-if-conversion2 -finline -fno-inline-functions-called-once -fipa-profile -fipa-pure-const -fipa-reference -fmerge-constants -fmove-loop-invariants -fomit-frame-pointer -freorder-blocks -fno-shrink-wrap -fno-split-wide-types -fssa-phiopt -fno-toplevel-reorder -fno-tree-bit-ccp -ftree-builtin-call-dce -ftree-ccp -ftree-ch -fno-tree-coalesce-vars -fno-tree-copy-prop -ftree-dce -fno-tree-dominator-opts -ftree-dse -ftree-fre -fno-tree-pta -fno-tree-sink -ftree-slsr -ftree-sra -fno-tree-ter -fno-align-labels -fcaller-saves -fcode-hoisting -fno-crossjumping -fcse-follow-jumps -fno-devirtualize -fdevirtualize-speculatively -fexpensive-optimizations -fgcse -fhoist-adjacent-loads -findirect-inlining -finline-small-functions -fno-ipa-bit-cp -fno-ipa-cp -fno-ipa-icf -fipa-icf-functions -fno-ipa-icf-variables -fno-ipa-ra -fno-ipa-sra -fipa-vrp -fno-isolate-erroneous-paths-dereference -flra-remat -fno-optimize-sibling-calls -fno-optimize-strlen -fpartial-inlining -fpeephole2 -free -fno-reorder-functions -frerun-cse-after-loop -fschedule-insns2 -fno-store-merging -fstrict-aliasing -fno-strict-overflow -fno-thread-jumps -fno-tree-pre -ftree-switch-conversion -fno-tree-tail-merge -ftree-vrp -fgcse-after-reload -fno-inline-functions -fno-ipa-cp-clone -fno-peel-loops -fpredictive-commoning -fno-split-loops -fno-split-paths -ftree-loop-distribute-patterns -ftree-loop-vectorize -ftree-partial-pre -ftree-slp-vectorize -funswitch-loops -fdce -fdelayed-branch -fdse -fcse-skip-blocks -fno-schedule-insns -falign-loops -fno-align-jumps -falign-functions -fno-reorder-blocks-and-partition",
            )

            concurrent.futures.wait([build1, build2])

            print(f'[test_together] test end')

            test1, test2 = None, None
            if not build1.cancelled() and build1.result() != -1:
                test1 = executor.submit(manager1.test)

            if not build2.cancelled() and build2.result() != -1:
                test2 = executor.submit(manager2.test)

            if test1 is None and test2 is None:
                return None, None

            if test2 is None:
                concurrent.futures.wait([test1])
                return test1.result(), None
            if test1 is None:
                concurrent.futures.wait([test2])
                return None, test2.result()

            concurrent.futures.wait([test1, test2])
            return test1.result(), test2.result()
            return SCANNManager.caculate_qps_acc(test1.result(), test2.result())


"""
build_dir是存放scann源码的位置

ann_dir是存放ann——benchmark的位置，为了方便使用，我略微修改了ann_benchmark，可以接受指定的dataset
所以最好复制/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks的直接使用

env是预留的python环境，为了可以并行需要多创建几个python环境
每个env需要执行下面的操作
1. 先在ann_dir执行 pip install -r requirements.txt
2. 然后再ann_dir执行 pip install -r requirements.txt
"""

if __name__ == "__main__":
    build_dir = "/home/scy/2025.05.23.turn.scann/src/scann/scann"
    build_dir_base = "/home/scy/2025.05.23.turn.scann/src/scann/scannO3"
    ann_dir = "/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks-flags"
    ann_dir_base = "/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks-base"
    env = "scanno3"
    env_base = "scannbase"

    manager = SCANNManager(
        build_dir=build_dir, env=env, ann_dir=ann_dir, datasets=["sift-128-euclidean"]
    )
    
    manager_base = SCANNManager(
        build_dir=build_dir_base,
        env=env_base,
        ann_dir=ann_dir_base,
        datasets=["sift-128-euclidean"],
    )
    print(SCANNManager.test_together(manager1=manager, manager2=manager_base))
