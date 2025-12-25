from concurrent.futures import ThreadPoolExecutor
import csv
import os
import shlex
import statistics
import subprocess
from typing import Optional, Union

import concurrent

FLOAT_MAX = float("inf")

# TODO 编译时线程控制
# TODO 不重新执行O3


class SCANNManager:
    def __init__(
        self,
        build_dir: str,
        env: str,
        ann_dir: str,
        repeat: int = 1,
        datasets: list[str] = [],
        num_repeat: int = 1,
    ):
        self.build_dir = build_dir
        self.env = env
        self.ann_dir = ann_dir
        self.repeat = repeat
        self.datasets = datasets
        self.num_repeat = num_repeat

    @staticmethod
    def transform_opt(opt: str):
        """
        bazel 中每一个cxxopt只能指定一个选项， 所以需要将多个选项拆分成多个--cxxopt
        """
        opt_list = opt.split(" ")
        copt_list = [f"--copt='{x}' " for x in opt_list if x]
        cxxopt_list = [f"--cxxopt='{x}' " for x in opt_list if x]
        return " ".join(copt_list) + " ".join(cxxopt_list)

    def run_py_command_in_env(
        self,
        command_list: Union[list[str], str],
        env: Optional[str] = None,
        cwd: Optional[str] = None,
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
        finally_command = f"taskset -c 192-319 conda run -n {env} bash -c {shlex.quote(joined_commands)}"

        p = subprocess.run(
            finally_command, stderr=stderr, stdout=stdout, shell=True, cwd=cwd, timeout=30*60
        )
        return p.returncode

    def build(self, opt_config: str = "-g -O3") -> int:
        if self.clean() != 0:
            print("Clean failed, please check the environment.")
            return -1

        bazel_opt = self.transform_opt(opt_config + " -falign-functions=32 -falign-jumps=4 -falign-loops=16 -ffp-contract=fast -fira-algorithm=CB -fira-region=mixed -fp-model=fast -freorder-blocks-algorithm=simple -fsched-stalled-insns-dep=4 -fsched-stalled-insns=12 -fsimd-cost-model=very-cheap -fstack-reuse=named_vars -fvect-cost-model=dynamic ")
        
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
            ":build_pip_pkg "
        )

        build_pip_pkg_command = "bazel-bin/build_pip_pkg"

        install_command = f"pip3 install scann-1.2.10-cp39-cp39-linux_aarch64.whl"

        check_command = f"pip list | grep scann"

        command_list = [
            build_command,
            build_pip_pkg_command,
            install_command,
            check_command,
        ]

        for command in command_list:
            return_code = self.run_py_command_in_env(command)
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

    def run_benchmark(self) -> dict[str, float]:
        mapping = {}
        for dataset in self.datasets:
            result_file = f"{dataset}_results.csv"

            run_command = f"python run.py --force --threads 128 --algorithm scann --dataset {dataset} --local"
            get_result_command = (
                f"python data_export.py --datasets {dataset} --output {result_file}"
            )

            command_list = [run_command, get_result_command]
            for command in command_list:
                if self.run_py_command_in_env(command, cwd=self.ann_dir) != 0:
                    return {}
            mapping.update(
                self.extract_qps_from_csv(os.path.join(self.ann_dir, result_file))
            )
        return mapping

    @staticmethod
    def extract_qps_from_csv(file_path: str) -> dict[str, int]:
        mapping = {}
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return mapping
            
            if not all(col in reader.fieldnames for col in ["filename", "qps"]):
                return mapping

            for row in reader:
                filename = row["filename"]
                qps = float(row["qps"])  # 将qps转换为浮点数
                mapping[filename] = qps
        return mapping

    @staticmethod
    def calculate_qps_acc(
        qps_dict: dict[str, float], qps_dict_base: dict[str, float]
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

    def run_single_test(self):
        # base_perf = base_perf = {
        #     # "600_NaN_2_squared_l2_35_240.hdf5": 124.12102978825011,
        #     "600_NaN_2_squared_l2_8_74.hdf5": 457.58623674802215,
        #     # "600_NaN_2_squared_l2_20_140.hdf5": 206.26822934105576,
        #     # "600_NaN_2_squared_l2_13_100.hdf5": 299.4451147754725,
        # }

        base_perf = {
            # "600_NaN_2_squared_l2_35_240.hdf5": 124.12102978825011,
            "600_NaN_2_squared_l2_8_74.hdf5": 1,
            # "600_NaN_2_squared_l2_20_140.hdf5": 206.26822934105576,
            # "600_NaN_2_squared_l2_13_100.hdf5": 299.4451147754725,
        }
        perf = self.run_benchmark()
        opt_perf = SCANNManager.calculate_qps_acc(perf, base_perf)

        base_perf = {
            # "600_NaN_2_squared_l2_35_240.hdf5": 124.12102978825011,
            "600_NaN_2_squared_l2_8_74.hdf5": 457.58623674802215,
            # "600_NaN_2_squared_l2_20_140.hdf5": 206.26822934105576,
            # "600_NaN_2_squared_l2_13_100.hdf5": 299.4451147754725,
        }
        acc_rate = SCANNManager.calculate_qps_acc(perf, base_perf)

        return opt_perf, acc_rate

    def test(self, num_repeat: int = -1) -> tuple[float, float]:
        """
        Run the Redis performance test multiple times and return the average time per request (in microseconds).

        :param num_repeat: Number of test repetitions, defaults to 1
        """
        if num_repeat < 0:
            num_repeat = self.num_repeat

        acc_rates = []
        opt_perfs = []
        for _ in range(num_repeat):
            opt_perf, acc_rate = self.run_single_test()
            if acc_rate <= 0 or acc_rate > 3:
                return FLOAT_MAX, FLOAT_MAX
            acc_rates.append(acc_rate)
            opt_perfs.append(opt_perf)
        median_opt_perf = statistics.median(opt_perfs) if opt_perfs else FLOAT_MAX
        median_acc_rate = statistics.median(acc_rates) if acc_rates else FLOAT_MAX
        return median_opt_perf, median_acc_rate


if __name__ == "__main__":
    build_dir = "/home/whq/scann/scann/scann"
    ann_dir = "/home/whq/scann/scann/ann-benchmarks-flags"
    env = "scanno3"
    
    manager = SCANNManager(
        build_dir=build_dir, env=env, ann_dir=ann_dir, datasets=["sift-128-euclidean"]
    )
    
    manager.build(opt_config="-g -O3")
    
    acc = 0
    for i in range(100):
        print(f"Running test {i + 1}", flush=True)
        acc += manager.test()
        
    print(f"Average acceleration: {acc / 100}")
    # build_dir = "/home/whq/scann/scann/scann"
    # build_dir_base = "/home/whq/scann/scann/scannO3"
    # ann_dir = "/home/whq/scann/scann/ann-benchmarks-flags"
    # ann_dir_base = "/home/whq/scann/scann/ann-benchmarks-base"
    # env = "scanno3"
    # env_base = "scannbase"

    # manager = SCANNManager(
    #     build_dir=build_dir, env=env, ann_dir=ann_dir, datasets=["sift-128-euclidean"]
    # )
    # manager_base = SCANNManager(
    #     build_dir=build_dir_base,
    #     env=env_base,
    #     ann_dir=ann_dir_base,
    #     datasets=["sift-128-euclidean"],
    # )
