import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Union


class RocksDBManager:
    def __init__(
        self,
        test_home: str = "/home/whq/YCSB",
    ):
        self.test_home = test_home

    def parser_exec_time(self, stdout: list[str]) -> float:
        """
        Parse the output of db_bench to extract the wall-clock time in seconds.
        Returns float('inf') if parsing fails.
        """
        """[OVERALL], Throughput(ops/sec), 1234.56"""
        pattern = re.compile(r"\[OVERALL\],\s+Throughput\(ops/sec\),\s+([\d.]+)")
        for line in stdout:
            match = pattern.search(line)
            if match:
                micros_per_op = float(match.group(1))
                return micros_per_op
        return float("inf")

    def build(self, opt_config: str = " -g -O3 ", jobs: int = 128) -> int:
        """
        Configure, build and install RocksDB with the given compiler flags.
        Returns 0 on success, -1 on failure.
        """

        build_script = "compile.sh"
        try:
            subprocess.run(
                [
                    "bash",
                    build_script,
                    "--opt_config",
                    opt_config,
                ],
                cwd=self.test_home,
                check=True,
                text=True,
                # stdout=subprocess.PIPE,
                # stderr=subprocess.PIPE,
                timeout=1800,
            )
        except Exception as e:
            print(f"[run] Buile failed, message: {e}")
            return -1
        return 0

    def test(self, num_repeat: int = 1) -> float:
        build_script = "run.sh"
        try:
            p = subprocess.run(
                [
                    "bash",
                    build_script,
                ],
                cwd=self.test_home,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1800,
            )
        except Exception as e:
            return -1
        perf = self.parser_exec_time(p.stdout.splitlines())
        if perf > 0:
            return perf
        return -1
       
    def clean(self):
        pass

if __name__ == "__main__":
    test_home = "/home/whq/dataset/rocksdb.6.26.1"
    manager = RocksDBManager(test_home=test_home)
    # manager.build(opt_config="-g -O3", jobs=32)
    result = manager.test()
    print(f"RocksDB test result: {result} ops/sec")
    pass
