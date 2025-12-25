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
        opt = SCANNManager.replace_bad_config(opt)
        opt_list = opt.split(" ")
        copt_list = [f"--copt='{x}' " for x in opt_list if x]
        cxxopt_list = [f"--cxxopt='{x}' " for x in opt_list if x]
        return " ".join(copt_list) + " ".join(cxxopt_list)

    @staticmethod
    def replace_bad_config(opt_config: str) -> str:
        # 如果启用"-flive-patching" 所有的-fipa都要禁用
        if "-flive-patching" in opt_config:
            opt_list = opt_config.split(" ")
            opt_list = [opt for opt in opt_list if not "-fipa" in opt]
            opt_config = " ".join(opt_list)

        # 有些优化选项是不存在  有些是会导致编译失败
        bad_config = ["-fsingle-precision-constant", "-fpack-struct", "-fno-rtti", "-fno-live-patching",
                      "-fno-stack-protector-all", "-fno-stack-protector-explicit", "-fno-stack-protector-strong",
                      "-gstatement-fno-rontiers", "-fno-no-threadsafe-statics"]
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
        if result == 0:
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
            # return SCANNManager.caculate_qps_acc(test1.result(), test2.result())

    @staticmethod
    def build_test(manager, flag: str = None):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交构建任务
            build1 = executor.submit(manager.build, opt_config=f" {flag} ")
            concurrent.futures.wait([build1])

            test1 = None
            if not build1.cancelled() and build1.result() != -1:
                test1 = executor.submit(manager.test)
            if test1 is None:
                return None
            concurrent.futures.wait([test1])
            return test1.result()



def reduceFlags(optList, build_path, add_part: str):
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
            # doris_manager = DorisManager(build_dir=build_path)
            # passed = doris_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part, jobs=128)

            scann_manager = SCANNManager(build_dir="/home/scy/2025.05.23.turn.scann/src/scann/scann.5",
                                         env='suo.scann5',
                                         ann_dir='/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks.5',
                                         datasets=['sift-128-euclidean'])
            scann_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part)

            test_time = 0
            avg_time = None
            while test_time < 5:
                avg_time = scann_manager.test()
                if isinstance(avg_time, dict):
                    break
                test_time += 1

            if not isinstance(avg_time, dict):
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

        # doris_manager = DorisManager(build_dir=doris_build_path)
        # passed = doris_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part, jobs=128)

        scann_manager = SCANNManager(build_dir="/home/scy/2025.05.23.turn.scann/src/scann/scann.5",
                                     env='suo.scann5',
                                     ann_dir='/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks.5',
                                     datasets=['sift-128-euclidean'])
        scann_manager.build(opt_config=' -g ' + level + ' ' + ' '.join(tmpOpt) + ' ' + add_part)

        test_time = 0
        avg_time = None
        while test_time < 5:
            avg_time = scann_manager.test()
            if isinstance(avg_time, dict):
                break
            test_time += 1

        if not isinstance(avg_time, dict):
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
        "-O3 -fno-align-functions -fno-align-labels -fno-align-loops -farray-widen-compare -fno-auto-inc-dec -fno-caller-saves -fno-code-hoisting -fno-combine-stack-adjustments -fno-compare-elim -fcrypto-accel-aes -fcx-fortran-rules -fcx-limited-range -fno-dce -fno-defer-pop -fexceptions -fno-expensive-optimizations -ffinite-loops -ffloat-store -fgcse-las -fno-gcse-lm -fgraphite -fno-guess-branch-probability -fno-hoist-adjacent-loads -ficp -fno-indirect-inlining -fno-inline-small-functions -fno-ipa-bit-cp -fno-ipa-cp -fno-ipa-cp-clone -fno-ipa-ic -fno-ipa-icf -fno-ipa-icf-variables -fipa-prefetch -fno-ipa-profile -fno-ipa-pure-const -fno-ipa-reference-addressable -fipa-reorder-fields -fipa-struct-reorg -fno-ira-hoist-pressure -fira-loop-pressure -fno-isolate-erroneous-paths-dereference -fno-ivopts -fno-jump-tables -fno-lifetime-dse -floop-crc -floop-elim -fno-loop-interchange -floop-nest-optimize -floop-parallelize-all -fno-math-errno -fmerge-mull -fmodulo-sched-allow-regmoves -fno-move-loop-invariants -fno-move-loop-stores -fnon-call-exceptions -fno-omit-frame-pointer -fopt-info -fno-optimize-strlen -fpack-struct -fno-partial-inlining -fno-plt -fno-printf-return-value -fprofile-reorder-functions -fno-reorder-blocks -freorder-blocks-and-partition -frounding-math -fno-sched-critical-path-heuristic -fno-sched-dep-count-heuristic -fno-sched-group-heuristic -fno-sched-rank-heuristic -fno-sched-spec -fsched-spec-load-dangerous -fno-sched-stalled-insns -fno-sched-stalled-insns-dep -fsched2-use-superblocks -fno-schedule-fusion -fno-section-anchors -fselective-scheduling -fno-short-enums -fno-shrink-wrap-separate -fno-signed-zeros -fsimdmath -fno-split-loops -fno-split-paths -fsplit-wide-types-early -fno-ssa-backprop -fstack-clash-protection -fstack-protector-strong -fno-store-merging -fno-strict-aliasing -fno-strict-volatile-bitfields -fno-thread-jumps -fno-trapping-math -ftrapv -fno-tree-bit-ccp -fno-tree-ccp -fno-tree-ch -fno-tree-dse -fno-tree-forwprop -fno-tree-loop-distribute-patterns -fno-tree-loop-distribution -fno-tree-loop-im -fno-tree-loop-ivcanon -fno-tree-partial-pre -fno-tree-pre -fno-tree-pta -fno-tree-reassoc -fno-tree-scev-cprop -fno-tree-sra -fno-tree-tail-merge -fno-tree-ter -ftree-vectorize -fno-tree-vrp -funroll-all-loops -fno-unroll-completely-grow-size -fno-unswitch-loops -fno-unwind-tables -fvar-tracking-assignments-toggle -fvar-tracking-uninit -fvpt -fweb -gstatement-frontiers -mcmlt-arith -mlow-precision-div -msimdmath-64 -fstrict-enums -fno-threadsafe-statics -fhandle-exceptions -ftree-loop-if-convert -fdelete-null-pointer-checks",
"-O3 -fno-aggressive-loop-optimizations -fno-align-loops -fassociative-math -fno-auto-inc-dec -fno-branch-count-reg -fno-caller-saves -fccmp2 -fno-crossjumping -fcrypto-accel-aes -fdelayed-branch -fdelete-dead-exceptions -fexceptions -ffinite-loops -ffloat-store -fno-gcse -fno-gcse-after-reload -fgcse-las -fgraphite -fgraphite-identity -fno-guess-branch-probability -fharden-conditional-branches -fno-hoist-adjacent-loads -fif-conversion-gimple -fno-if-conversion2 -fno-indirect-inlining -fno-inline-functions -fno-ipa-cp -fno-ipa-cp-clone -fno-ipa-profile -fno-ipa-pure-const -fno-ipa-reference-addressable -fno-ipa-sra -fno-ipa-strict-aliasing -fipa-struct-reorg -fira-loop-pressure -fno-isolate-erroneous-paths-dereference -fkeep-gc-roots-live -fkernel-pgo -floop-crc -fno-loop-interchange -fno-math-errno -fmerge-mull -fmodulo-sched -fnon-call-exceptions -fno-optimize-strlen -fpack-struct -fno-partial-inlining -fno-predictive-commoning -fprofile-partial-training -fprofile-reorder-functions -fno-reorder-blocks -fno-rerun-cse-after-loop -fno-sched-spec-insn-heuristic -fno-sched-stalled-insns-dep -fsel-sched-pipelining -fsel-sched-pipelining-outer-loops -fsel-sched-reschedule-pipelined -fselective-scheduling -fno-shrink-wrap-separate -fsignaling-nans -fsingle-precision-constant -fno-split-paths -fno-split-wide-types -fstack-protector -fstack-protector-all -fstack-protector-explicit -fstack-protector-strong -fno-stdarg-opt -fno-store-merging -fno-strict-volatile-bitfields -fno-thread-jumps -fno-toplevel-reorder -ftracer -fno-trapping-math -ftrapv -fno-tree-builtin-call-dce -fno-tree-copy-prop -fno-tree-dominator-opts -fno-tree-forwprop -fno-tree-loop-optimize -ftree-lrs -fno-tree-partial-pre -fno-tree-phiprop -fno-tree-pre -fno-tree-sink -fno-tree-slp-vectorize -fno-tree-slsr -fno-tree-switch-conversion -fno-tree-ter -fno-tree-vrp -funroll-loops -funsafe-math-optimizations -fvar-tracking -fvar-tracking-assignments-toggle -fno-version-loops-for-strides -mcmlt-arith -mlow-precision-div -mlow-precision-recip-sqrt -mlow-precision-sqrt -fnothrow-opt -flive-patching -fprefetch-loop-arrays -ftree-loop-if-convert",
"-O3 -fno-align-functions -fno-allocation-dce -farray-widen-compare -fassociative-math -fno-asynchronous-unwind-tables -fno-branch-count-reg -fbranch-probabilities -fno-caller-saves -fconvert-minmax -fno-cse-follow-jumps -fno-defer-pop -fdelete-dead-exceptions -fno-devirtualize-speculatively -fno-dse -fno-early-inlining -ffinite-loops -ffloat-store -fno-fp-int-builtin-inexact -fno-function-cse -fno-gcse -fno-gcse-after-reload -fgcse-las -fno-gcse-lm -fgraphite -fno-guess-branch-probability -fharden-conditional-branches -ficp -ficp-speculatively -fno-if-conversion -fno-if-conversion2 -fifcvt-allow-complicated-cmps -fno-inline -fno-inline-functions -fno-ipa-cp -fno-ipa-cp-clone -fno-ipa-icf -fno-ipa-modref -fno-ipa-pure-const -fno-ipa-reference -fipa-reorder-fields -fno-ipa-stack-alignment -fipa-struct-reorg -fno-ipa-vrp -fira-loop-pressure -fno-ira-share-save-slots -fno-isolate-erroneous-paths-dereference -fno-ivopts -fno-jump-tables -fkeep-gc-roots-live -fkernel-pgo -fno-lifetime-dse -floop-crc -floop-nest-optimize -floop-parallelize-all -fmerge-mull -fmodulo-sched-allow-regmoves -fno-optimize-strlen -fno-peephole2 -fno-predictive-commoning -fno-printf-return-value -fprofile-partial-training -fprofile-reorder-functions -fno-ree -frename-registers -frounding-math -fno-sched-group-heuristic -fno-sched-interblock -fno-sched-stalled-insns -fno-sched-stalled-insns-dep -fno-schedule-fusion -fno-schedule-insns -fsel-sched-pipelining -fsel-sched-reschedule-pipelined -fno-short-enums -fno-shrink-wrap -fsignaling-nans -fsingle-precision-constant -fsplit-ldp-stp -fno-split-loops -fno-split-wide-types -fno-ssa-backprop -fno-ssa-phiopt -fstack-clash-protection -fstack-protector -fstack-protector-all -fstack-protector-explicit -fno-strict-volatile-bitfields -fno-thread-jumps -ftracer -fno-trapping-math -ftrapv -fno-tree-ccp -fno-tree-coalesce-vars -ftree-cselim -fno-tree-dce -fno-tree-forwprop -fno-tree-loop-distribute-patterns -fno-tree-loop-ivcanon -fno-tree-partial-pre -fno-tree-pre -fno-tree-pta -fno-tree-reassoc -fno-tree-sink -ftree-slp-transpose-vectorize -fno-tree-slp-vectorize -fno-tree-sra -fno-tree-ter -ftree-vectorize -funconstrained-commons -funroll-all-loops -fno-unswitch-loops -fno-unwind-tables -fno-version-loops-for-strides -fweb -fwrapv -fwrapv-pointer -mlow-precision-div -mlow-precision-recip-sqrt -mlow-precision-sqrt -frtti -flive-patching -ftree-loop-if-convert",
"-O3 -fno-aggressive-loop-optimizations -fno-align-functions -fno-align-labels -fno-align-loops -fno-asynchronous-unwind-tables -fno-bit-tests -fno-caller-saves -fccmp2 -fno-code-hoisting -fno-compare-elim -fconvert-minmax -fcrypto-accel-aes -fno-cse-follow-jumps -fcx-fortran-rules -fcx-limited-range -fno-dce -fno-defer-pop -fdelayed-branch -fno-devirtualize-speculatively -fno-dse -fno-expensive-optimizations -ffinite-loops -ffloat-store -fno-forward-propagate -fno-function-cse -fno-gcse -fno-gcse-after-reload -fgcse-las -fno-gcse-lm -fgcse-sm -fgraphite-identity -fno-guess-branch-probability -fharden-compares -fharden-conditional-branches -ficp -fno-if-conversion -fif-conversion-gimple -fifcvt-allow-complicated-cmps -fno-indirect-inlining -fno-inline -fno-inline-functions -fno-inline-functions-called-once -fno-ipa-cp -fno-ipa-ic -fno-ipa-icf-variables -fno-ipa-modref -fipa-prefetch -fno-ipa-profile -fno-ipa-ra -fno-ipa-reference -fno-ipa-sra -fno-ipa-stack-alignment -fno-ipa-strict-aliasing -fno-ipa-vrp -fno-ira-hoist-pressure -fno-ira-share-save-slots -fno-ira-share-spill-slots -fno-isolate-erroneous-paths-dereference -fno-jump-tables -fkeep-gc-roots-live -fno-lifetime-dse -flimit-function-alignment -flive-range-shrinkage -floop-crc -fno-loop-interchange -floop-nest-optimize -fno-lra-remat -fno-math-errno -fmerge-mull -fmodulo-sched -fmodulo-sched-allow-regmoves -fno-optimize-sibling-calls -fno-optimize-strlen -fpack-struct -fno-plt -fno-predictive-commoning -fno-printf-return-value -fprofile-partial-training -fprofile-reorder-functions -frename-registers -fno-reorder-functions -fno-sched-last-insn-heuristic -fno-sched-spec -fno-sched-stalled-insns-dep -fsched2-use-superblocks -fno-schedule-fusion -fno-schedule-insns -fno-section-anchors -fsel-sched-pipelining -fsel-sched-reschedule-pipelined -fselective-scheduling2 -fno-short-enums -fshort-wchar -fno-shrink-wrap -fno-signed-zeros -fno-split-ivs-in-unroller -fsplit-ldp-stp -fno-split-loops -fno-split-paths -fno-split-wide-types -fstack-clash-protection -fstack-protector -fstack-protector-explicit -fno-stdarg-opt -fno-store-merging -fno-strict-volatile-bitfields -fno-thread-jumps -fno-toplevel-reorder -fno-trapping-math -ftrapv -fno-tree-bit-ccp -fno-tree-ccp -fno-tree-ch -fno-tree-coalesce-vars -fno-tree-copy-prop -fno-tree-dce -fno-tree-dse -fno-tree-forwprop -fno-tree-loop-distribute-patterns -fno-tree-loop-im -fno-tree-loop-ivcanon -fno-tree-loop-optimize -fno-tree-loop-vectorize -fno-tree-pta -fno-tree-reassoc -fno-tree-sink -ftree-slp-transpose-vectorize -fno-tree-slp-vectorize -fno-tree-sra -fno-tree-tail-merge -fno-unroll-completely-grow-size -funroll-loops -fno-unwind-tables -fvar-tracking -fvar-tracking-assignments-toggle -fno-version-loops-for-strides -fvpt -fwrapv -fwrapv-pointer -mcmlt-arith -mlow-precision-sqrt -ffold-simple-inlines -frtti -fno-threadsafe-statics -fhandle-exceptions -ftree-loop-if-convert",
"-O3 -fno-aggressive-loop-optimizations -fno-align-functions -fno-align-jumps -fno-align-labels -farray-widen-compare -fno-asynchronous-unwind-tables -fno-auto-inc-dec -fno-bit-tests -fbranch-probabilities -fno-caller-saves -fno-code-hoisting -fno-combine-stack-adjustments -fno-compare-elim -fconserve-stack -fconvert-minmax -fno-cse-follow-jumps -fcx-fortran-rules -fno-dce -fno-defer-pop -fdelayed-branch -fno-dse -fno-early-inlining -fexceptions -ffinite-loops -ffloat-store -fftz -fno-function-cse -fno-gcse-after-reload -fno-guess-branch-probability -fharden-compares -fno-hoist-adjacent-loads -ficp -fifcvt-allow-complicated-cmps -fno-indirect-inlining -fno-inline -fno-inline-small-functions -fno-ipa-cp -fno-ipa-ic -fno-ipa-icf -fno-ipa-icf-functions -fno-ipa-icf-variables -fno-ipa-modref -fipa-pta -fno-ipa-ra -fipa-reorder-fields -fno-ipa-sra -fno-ipa-stack-alignment -fno-ipa-strict-aliasing -fipa-struct-reorg -fno-ipa-vrp -fno-ira-hoist-pressure -fira-loop-pressure -fno-ira-share-save-slots -fno-ira-share-spill-slots -fno-isolate-erroneous-paths-dereference -fno-ivopts -fkernel-pgo -fno-lifetime-dse -flimit-function-alignment -flive-range-shrinkage -floop-crc -fno-loop-interchange -floop-parallelize-all -fno-loop-unroll-and-jam -fno-lra-remat -fmodulo-sched -fmodulo-sched-allow-regmoves -fno-move-loop-invariants -fnon-call-exceptions -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-optimize-strlen -fno-peel-loops -fno-peephole2 -fno-plt -fno-predictive-commoning -fno-printf-return-value -fprofile-partial-training -fprofile-reorder-functions -frename-registers -fno-reorder-functions -frounding-math -fno-sched-pressure -fno-sched-rank-heuristic -fno-sched-spec -fsched-spec-load-dangerous -fno-sched-stalled-insns-dep -fsched2-use-superblocks -fno-schedule-fusion -fno-schedule-insns -fno-section-anchors -fsel-sched-pipelining -fsel-sched-reschedule-pipelined -fselective-scheduling -fselective-scheduling2 -fno-semantic-interposition -fno-short-enums -fshort-wchar -fno-shrink-wrap -fno-signed-zeros -fno-split-ivs-in-unroller -fsplit-ldp-stp -fno-split-loops -fno-split-paths -fno-split-wide-types -fno-ssa-phiopt -fstack-clash-protection -fstack-protector -fstack-protector-explicit -fstack-protector-strong -fno-stdarg-opt -fno-store-merging -fno-strict-aliasing -fno-strict-volatile-bitfields -fno-trapping-math -fno-tree-coalesce-vars -fno-tree-copy-prop -fno-tree-dce -fno-tree-dominator-opts -fno-tree-dse -fno-tree-forwprop -fno-tree-loop-distribute-patterns -fno-tree-loop-im -fno-tree-loop-ivcanon -fno-tree-loop-optimize -fno-tree-loop-vectorize -ftree-lrs -fno-tree-pta -fno-tree-reassoc -fno-tree-sink -fno-tree-sra -fno-tree-switch-conversion -fno-tree-tail-merge -funroll-all-loops -fno-unroll-completely-grow-size -funsafe-math-optimizations -fno-unswitch-loops -fno-unwind-tables -fvar-tracking -fvar-tracking-assignments-toggle -fvpt -fwrapv-pointer -gstatement-frontiers -mlow-precision-div -mlow-precision-recip-sqrt -mlow-precision-sqrt -ffold-simple-inlines -frtti -fstrict-enums -fno-threadsafe-statics -flive-patching -fdelete-null-pointer-checks",
    ]

    add_parts = [""] * len(optlist)

    # optlist = ["-O3"] * 100

    f = open('result.txt', "a+")

    for idx in range(len(optlist)):
        print(f'[main] {idx}-th opt')

        # doris_manager = DorisManager(build_dir=doris_build_path, fe_port=9030, be_port=9050)
        # passed = doris_manager.build(opt_config=' -g ' + optlist[idx], jobs=128)

        scann_manager = SCANNManager(build_dir="/home/scy/2025.05.23.turn.scann/src/scann/scann.5",
                                     env='suo.scann5',
                                     ann_dir='/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks.5',
                                     datasets=['sift-128-euclidean'])
        scann_manager.clean()
        scann_manager.build(opt_config=' -g ' + optlist[idx])

        test_time = 0
        avg_time = None
        while test_time < 5:
            avg_time = scann_manager.test()
            if isinstance(avg_time, dict):
                break
            test_time += 1

        if isinstance(avg_time, dict):
            f.write('success\n')
            f.flush()
            continue

        o = optlist[idx].split(' ')

        o = reduceFlags(o, doris_build_path, add_parts[idx])
        o = reduceMore(o, doris_build_path, add_parts[idx])

        f.write(' '.join(o) + '\n')
        f.flush()
    f.close()


