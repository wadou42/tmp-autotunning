from autoturning.manager.manager import Manager
import subprocess
import re

import os

class ZstdManager(Manager):
    def __init__(self, benchmark_core=312, build_dir="/home/whq/workspace/zstd-instance1", num_repeat: int = 1):
        self.benchmark_core = benchmark_core
        self.build_dir = build_dir
        self.num_repeat = num_repeat

    def build(self, opt_config: str = "-g -O3") -> int:
        os.sched_setaffinity(0, set(range(128,256)))
        self.clean()
        build_commands = f"make -j32 MOREFLAGS='{opt_config}'"
        p = subprocess.run(
            build_commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=self.build_dir,
        )
        if p.returncode != 0:
            return -1
        return 0

    def parser_results(self, raw_output: str) -> tuple[float, float]:

        for line in raw_output.splitlines():
            if match := re.search(
                r"zstd [\d\.]+ -\d+\s+[\d\.]+\s+([\d\.]+) MB/s\s+([\d\.]+) us\s+([\d\.]+) MB/s\s+([\d\.]+) us", line
            ):
                compress_speed = float(match.group(1))
                decompress_speed = float(match.group(3))
                return compress_speed, decompress_speed
        return -1, -1

    def test(self, num_repeat: int = -1) -> float:
        os.sched_setaffinity(0, set(range(self.benchmark_core ,self.benchmark_core + 1)))
        if num_repeat == -1:
            num_repeat = self.num_repeat
        compress_results: list[float] = []
        decompress_results: list[float] = []
        for i in range(1, num_repeat + 1):
            p = subprocess.run(
                f"ZSTD_HOME='{self.build_dir}' bash run.sh",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.build_dir,
            )
            compress, decompress = self.parser_results(
                p.stdout.decode() if p.stdout else ""
            )
            if p.returncode != 0:
                return -1
            compress_results.append(compress)
            decompress_results.append(decompress)

        def _median(lst):
            lst_copy = lst[:]
            n = len(lst_copy)
            if n == 0:
                return -1
            lst_copy.sort()
            mid = n // 2
            if n % 2 == 0:
                return (lst_copy[mid - 1] + lst_copy[mid]) / 2.0
            else:
                return lst_copy[mid]

        print(f"Compress speeds: {compress_results}")
        print(f"Decompress speeds: {decompress_results}")
        print(f"Median compress speed: {_median(compress_results)} MB/s")
        print(f"Median decompress speed: {_median(decompress_results)} MB/s")
        return _median(compress_results)

    def clean(self) -> int:
        clean_commands = "make clean"
        p = subprocess.run(clean_commands, shell=True,
                           stderr=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                           cwd=self.build_dir)
        if p.returncode != 0:
            return -1
        return 0


if __name__ == "__main__":
    manager = ZstdManager(build_dir="/home/whq/dataset/zstd/lzbench-2.2", num_repeat=1)
    manager.build(opt_config="-O3")
    manager.test()
    manager.clean()


# Compress speeds: [173.0, 176.0, 176.0, 176.0, 177.0, 183.0, 175.0, 173.0, 178.0, 181.0]
# Decompress speeds: [1036.0, 1037.0, 1034.0, 1038.0, 1036.0, 1036.0, 1036.0, 1036.0, 1034.0, 1032.0]
