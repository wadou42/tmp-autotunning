#!/usr/bin/env python3
import os
import re
import queue
import socket
import shutil
import subprocess
from pathlib import Path
import threading
import time



def parse_qps(stdouts: list[str]) -> float:
    for out in stdouts:
        pattern = r"queries:\s+(\d+)\s+\(([\d\.]+)\sper\s+sec.\)"
        if match := re.search(pattern, out):
            qps = float(match.group(2))
            return qps
    return -1


class MySQLManager:
    """
    The MySQLManager class encapsulates methods to build and test a MySQL server instance.
    """

    def __init__(
        self,
        build_home: str = "/home/whq/mysql/build",
        install_home: str = "/home/whq/bin/mysql",
        data_home: str = "/data/instance0",
        test_home: str = "/home/whq/bin/sysbench/share/sysbench",
        cnf_file: str = "/home/whq/my.cnf",
    ):
        """
        Initialize the MySQL environment paths and configurations.

        Args:
            build_home (str): Path to MySQL build directory. Defaults to "/home/whq/dataset/mysql/mysql/build".
            install_home (str): Path to MySQL installation directory. Defaults to "/home/whq/bin/mysql".
            data_home (str): Path to MySQL data storage directory. Defaults to "/home/whq/data/mysql".
            test_home (str): Path to sysbench test directory. Defaults to "/home/whq/dataset/mysql/sysbench".
            cnf_file (str): Path to MySQL configuration file. Defaults to "/etc/my.cnf".

        Notes:
            - All paths will be converted to `Path` objects internally.
            - `data_subdirs` is initialized as ["data", "run", "tmp", "log"].
        """
        self.build_home = Path(build_home)
        self.install_home = Path(install_home)
        self.data_home = Path(data_home)
        self.test_home = Path(test_home)
        self.cnf_file = cnf_file
        self.data_subdirs = ["data", "run", "tmp", "log"]

    def clean_directory(self, directory: Path) -> None:
        if not os.path.exists(directory):
            return
        for item in os.listdir(directory):
            path = os.path.join(directory, item)

            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            else:
                shutil.rmtree(path)



    def clean(self) -> None:
        """
        Stop any running MySQL instances and clean build, install, and data directories.
        """
        subprocess.run(["pkill", "-9", "mysqld"], stderr=subprocess.DEVNULL)
        self.clean_directory(self.data_home)
        for subdir in self.data_subdirs:
            (self.data_home / subdir).mkdir(parents=True, exist_ok=True)
        try:
            # 清理主目录
            self.clean_directory(self.build_home)
            self.clean_directory(self.install_home)

            self.build_home.mkdir(parents=True, exist_ok=True)
            self.install_home.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Directory preparation failed: {e}")

    def clean_cache(self):
        """
        Clean the data directory and recreate necessary subdirectories.
        """
        self.clean_directory(self.data_home / "data")
        (self.data_home / "data").mkdir(parents=True, exist_ok=True)
        for subdir in self.data_subdirs:
            (self.data_home / subdir).mkdir(parents=True, exist_ok=True)

    def run_cmake(self, opt_config: str = "-O3") -> int:
        """
        Execute the CMake configuration command with specified optimization flags.
        Args:
            opt_config (str): Compiler optimization flags (e.g. '-O3 -march=native').
        Returns:
            int: 0 on success, -1 on failure.
        """
        cmake_cmd = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.install_home}",
            f"-DMYSQL_DATADIR={self.data_home}/data",
            f"-DWITH_BOOST={self.build_home.parent / 'boost/boost_1_73_0'}",
            f"-DCMAKE_C_FLAGS_RELEASE='-g -DNDEBUG {opt_config}'",
            f"-DCMAKE_CXX_FLAGS_RELEASE='-g -DNDEBUG {opt_config}'",
        ]
        try:
            p = subprocess.run(
                cmake_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=self.build_home,
                timeout=30 * 60,
            )
        except Exception as e:
            return -1
        if p.returncode != 0:
            return -1
        return 0

    def compile_and_install(self) -> int:
        """
        Execute the compilation and installation commands.
        """
        compile_cmds = [["make", "-j128"], ["make", "install"]]

        for cmd in compile_cmds:
            try:
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    cwd=self.build_home,
                    timeout=60 * 60,
                )
            except Exception as e:
                return -1
            if p.returncode != 0:
                return -1
        return 0

    def set_permissions(self) -> None:
        """
        Set file permissions for MySQL server scripts and data directories.
        For testing purposes, file permissions are set to be fully open.
        777 for executable scripts and 755 for data directories.
        """
        try:
            mysql_server = self.install_home / "support-files" / "mysql.server"
            if mysql_server.exists():
                mysql_server.chmod(0o777)

            data_dir = self.data_home / "data"
            if data_dir.exists():
                data_dir.chmod(0o755)
        except Exception as e:
            raise RuntimeError(f"Permission setting failed: {e}")

    def build(self, opt_config: str) -> int:
        os.sched_setaffinity(0, set(range(0, 128)))
        opt_config=opt_config.replace("O2", "O3").replace("O1", "O3")
        """
        Build and install MySQL server with specified optimization configuration.
        Args:
            opt_config (str): Compiler optimization flags (e.g. '-O3 -march=native').
        Returns:
            int: 0 on success, -1 on failure.
        """
        self.clean()

        if self.run_cmake(opt_config=opt_config) != 0:
            print("CMake configuration failed", flush=True)
            return -1

        if self.compile_and_install() != 0:
            print("Compilation failed")
            return -1

        self.set_permissions()

        return 0

    def prepare_test(self):
        """Initialize and start MySQL server"""
        self.clean_cache()
        try:
            subprocess.run(
                ["./mysqld", f"--defaults-file={self.cnf_file}", "--initialize"],
                cwd=self.install_home / "bin",
                check=True,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                timeout=30 * 60,
            )
        except Exception as e:
            print(f"Error {e}")
            return -1

        start_mysql_cmd = f"""./mysqld --defaults-file={self.cnf_file} --datadir={self.data_home}/data --socket={self.data_home}/run/mysql.sock --skip-grant-tables"""
        try:
            subprocess.run(
                start_mysql_cmd,
                shell=True,
                cwd=self.install_home / "bin",
                check=True,
                # stderr=subprocess.PIPE,
                # stdout=subprocess.PIPE,
                timeout=1 * 60 * 60,  # 设置超时时间为1小时
            )
        except Exception as e:
            print(f"Error {e}")
            return -1

        return 0


    def run_benchmark(self, result_queue):
        mysql_bin = self.install_home / "bin"
        mysql_cmd = str(mysql_bin / "mysql")
        mysql_sock = str(self.data_home / "run/mysql.sock")
        mysqladmin_cmd = str(mysql_bin / "mysqladmin")

        if not self.wait_mysql_start():
            return -1

        try:
            # stage1: initialize test database
            subprocess.run(
                [
                    mysql_cmd,
                    "-u",
                    "root",
                    f"--socket={mysql_sock}",
                    "-e",
                    "DROP DATABASE IF EXISTS test; CREATE DATABASE test;",
                ],
                cwd=self.install_home / "bin",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30 * 60,
            )
            # stage2: run performance benchmark
            p = subprocess.run(
                f"bash _run_all.sh -d {self.data_home};",
                cwd=self.test_home,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30 * 60,
            )

            perf = parse_qps(p.stdout.decode().split("\n"))

            subprocess.run(
                [mysqladmin_cmd, "-u", "root", f"--socket={mysql_sock}", "shutdown"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                timeout=35,
                check=False,
            )

            # stage3: return execution time
            result_queue.put(perf)
            return perf

        except Exception as e:
            print(f"Benchmark error: {str(e)}")
            result_queue.put(-1.0)

        finally:
            # stage4: cleanup (independent exception handling)
            try:
                subprocess.run(
                    [
                        mysqladmin_cmd,
                        "-u",
                        "root",
                        f"--socket={mysql_sock}",
                        "shutdown",
                    ],
                    timeout=30,
                    check=False,  # 允许关闭失败
                )
            except Exception as e:
                print(f"Cleanup warning: {str(e)}")


    def wait_mysql_start(self, timeout: int = 300) -> bool:
        def check_by_socket():
            return (self.data_home / "run/mysql.sock").exists()

        def check_by_mysqladmin():
            try:
                cmd = [
                    "./mysqladmin",
                    "ping",
                    f"--socket={self.data_home}/run/mysql.sock",
                ]
                result = subprocess.run(
                    cmd,
                    cwd=self.install_home / "bin",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                )
                return "mysqld is alive" in result.stdout.decode()
            except:
                return False

        # wait for mysql to start
        start_time = time.time()
        while time.time() - start_time < timeout:
            if check_by_socket() and check_by_mysqladmin():
                time.sleep(2)
                return True
            time.sleep(1)
        return False

    def test(self):
        """主测试方法"""
        cmd = "echo 3 > /proc/sys/vm/drop_caches"
        os.sched_setaffinity(0, set(range(0, 16)))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            timeout=100,
        )
        time.sleep(5)
        subprocess.run(["pkill", "-9", "mysqld"], stderr=subprocess.DEVNULL)

        result_queue = queue.Queue()
        # 启动线程
        mysql_thread = threading.Thread(target=self.prepare_test)
        bench_thread = threading.Thread(target=self.run_benchmark, args=(result_queue,))

        mysql_thread.start()
        bench_thread.start()

        # 等待测试完成
        mysql_thread.join()
        bench_thread.join(timeout=1 * 60 * 60)

        if not result_queue.empty():
            return float(result_queue.get())
        return -1

  

    @staticmethod
    def analysis_report(management1: "MySQLManager", management2: "MySQLManager"):
        func_file1 = management1.build_home.parent / "project-func-info-databaase.json"
        func_file2 = management2.build_home.parent / "project-func-info-databaase.json"
        report_file1 = management1.install_home / "bin/report.txt"
        report_file2 = management2.install_home / "bin/report.txt"

        analysis1 = LineAnalysis(func_file=func_file1, report_file=report_file1)  # type: ignore
        analysis2 = LineAnalysis(func_file=func_file2, report_file=report_file2)  # type: ignore
        result1 = analysis1.parse_report()
        result2 = analysis2.parse_report()
        if not result1 or not result2:
            return []
        for key in result1:
            modified_key = (key[0].replace("mysql/mysql", "mysql/mysqlO3"),) + key[1:]
            result1[key].extend(result2.get(modified_key, [0, 0]))
        result_list = []

        for key, value in result1.items():
            result_list.append(list(key) + value)
        return result_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MySQL Build System")
    parser.add_argument(
        "--optimize",
        "-o",
        type=str,
        default="-g -O3",
        help="Compiler optimization flags (e.g. '-O3 -march=native')",
    )

    args = parser.parse_args()
    try:
        builder = MySQLManager(
        )
        print(f"build with {args.optimize}")
        # builder.build(f" -g  {args.optimize}")
        perfs = []
        for _ in range(1):
            # builder.build(" -g -O3 ")
            # print(f"Build with -O3 completed.", flush=True)
            result = builder.test()
            perfs.append(result)
            print(f"The {_+1}-th test result:", flush=True)
            print(f"perf: {result}", flush=True)

        median_perf = sorted(perfs)[len(perfs) // 2]
        print(f"Median perf: {median_perf}", flush=True)
        avg_perf = sum(perfs) / len(perfs)
        print(f"Average perf: {avg_perf}", flush=True)
        print(f"All perfs: {perfs}", flush=True)
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
