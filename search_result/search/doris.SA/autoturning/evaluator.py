from abc import ABC, abstractmethod
from typing import Optional
import statistics
import random

# from autoturning.evaluator.doris_manager import DorisManager
# from autoturning.evaluator.redis.redis_manager import RedisManager
# from autoturning.evaluator.scann_manager import SCANNManager
from autoturning.manager.redis_manager import RedisManager
from autoturning.manager.doris_manager import DorisManager
from autoturning.manager.scann_manager import SCANNManager
from autoturning.recorder import Recorder

from autoturning.utils import calculate_acc_rate
import autoturning.script_args


class Evaluator(ABC):
    """
    A base evaluator class for providing an evaluation interface to the search framework.
    """
    def __init__(self, o3_project_path: str, opt_project_path: str, repeat_time: int, rebuild_o3: bool = True):
        self.o3_project_path = o3_project_path
        self.opt_project_path = opt_project_path
        self.rebuild_o3 = rebuild_o3
        self.repeat_time = repeat_time
        self.built_o3 = False

    @abstractmethod
    def evaluate(self, opt: str) -> tuple[float, float, float]:
        """
        Interface function provide to search framework.
        :param opt: the command line optimization
        :return: a tuple, contains o3_perf, opt_perf, and acc_rate
        """
        pass

    @abstractmethod
    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):
        """
        Evaluate the optimization multiple times with build once.
        :param opt: the optimization sequence.
        :param recorder: the recorder object
        :param repeat_time: test time
        :return: None
        """

        pass


class DorisEvaluator(Evaluator):
    """
    The evaluator class for doris.
    """
    def __init__(self, o3_project_path: str, opt_project_path: str, repeat_time: int, rebuild_o3: bool = True,
                 fe_port: int = 9030, be_port: int = 9050):
        """
        Create a doris evaluator
        :param o3_project_path: project path that will be built under o3
        :param opt_project_path: project path that will be built under turned optimization settings.
        :param repeat_time: the repeat time that conduct to evaluate the performance
        :param rebuild_o3: whether repeatedly build the project under O3
        :param fe_port: doris default port, do not change
        :param be_port: doris default port, do not change
        """
        super().__init__(o3_project_path, opt_project_path, repeat_time, rebuild_o3)
        self.fe_port = fe_port
        self.be_port = be_port

    @DeprecationWarning
    def evaluate_old(self, opt: str) -> tuple[float, float, float]:
        mode = 'opt'

        if mode == 'normal':
            opt_manager = DorisManager(build_dir=self.opt_project_path, repeat=20, fe_port=self.fe_port,
                                       be_port=self.be_port)
            o3_manager = DorisManager(build_dir=self.o3_project_path, repeat=20, fe_port=self.fe_port,
                                      be_port=self.be_port)
            opt_all_times: Optional[list[float]] = None
            opt_all_query_times: dict[str, list[float]]
            o3_all_times: Optional[list[float]] = None
            o3_all_query_times: dict[str, list[float]]

            try:
                if self.rebuild_o3 or not self.built_o3 or self.opt_project_path == self.o3_project_path:
                    o3_manager.build(opt_config="-g -Ofast", jobs=128)
                o3_all_times, o3_all_query_times = o3_manager.test()

                opt_manager.build(opt_config=opt, jobs=128)
                opt_all_times, opt_all_query_times = opt_manager.test()
            except KeyboardInterrupt:
                o3_manager.stop_services()
                opt_manager.stop_services()

            if o3_all_times is None or len(o3_all_times) == 0:
                o3_all_times = [-1]
            if opt_all_times is None or len(opt_all_times) == 0:
                opt_all_times = [-1]

            o3_avg_time = sum(o3_all_times) / len(o3_all_times)
            opt_avg_time = sum(opt_all_times) / len(opt_all_times)

            o3_mean_time = statistics.mean(o3_all_times)
            opt_mean_time = statistics.mean(opt_all_times)

            return o3_mean_time, opt_mean_time, calculate_acc_rate(o3_mean_time, opt_mean_time)
        elif mode == 'opt':
            opt_all_times: Optional[list[float]] = None
            opt_manager = DorisManager(build_dir=self.opt_project_path, repeat=15, fe_port=self.fe_port,
                                       be_port=self.be_port)
            try:
                opt_manager.build(opt_config=opt, jobs=128)
                opt_all_times, opt_all_query_times = opt_manager.test()
            except KeyboardInterrupt:
                opt_manager.stop_services()

            if opt_all_times is None or len(opt_all_times) == 0:
                opt_all_times = [-1]

            opt_mean_time = statistics.mean(opt_all_times)
            return -1, opt_mean_time, -1
        elif mode == 'O3':
            opt_all_times: Optional[list[float]] = None
            opt_manager = DorisManager(build_dir=self.opt_project_path, repeat=20, fe_port=self.fe_port,
                                       be_port=self.be_port)
            try:
                opt_manager.build(jobs=128)
                opt_all_times, opt_all_query_times = opt_manager.test()
            except KeyboardInterrupt:
                opt_manager.stop_services()

            if opt_all_times is None or len(opt_all_times) == 0:
                opt_all_times = [-1]

            opt_mean_time = statistics.mean(opt_all_times)
            return -1, opt_mean_time, -1

    def evaluate(self, opt: str) -> tuple[float, float, float]:
        doris_manager_base = DorisManager(build_dir=self.opt_project_path, repeat=self.repeat_time)

        o3_time: Optional[float] = None
        opt_time: Optional[float] = None
        if autoturning.script_args.evaluate_search:
            doris_manager_base.clean()
            doris_manager_base.build(opt_config=opt)
            opt_time = doris_manager_base.test(interval=0, task='hot')
            o3_time = autoturning.script_args.default_o3_perf
        if autoturning.script_args.evaluate_o3:
            doris_manager_base.clean()
            doris_manager_base.build(opt_config='-O3')
            o3_time = doris_manager_base.test(interval=0, task='hot')

        assert o3_time is not None
        assert opt_time is not None

        return o3_time, opt_time, calculate_acc_rate(o3_perf=o3_time, opt_perf=opt_time, project='doris')

    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):  # TODO
        manager = DorisManager(build_dir=self.opt_project_path, repeat=15)
        manager.clean()
        manager.build(opt_config=opt)

        o3_perf = autoturning.script_args.default_o3_perf

        all_acc_rate = []
        all_o3_perf = []
        all_opt_perf = []

        for idx in range(repeat_time):
            opt_perf = manager.test(interval=0, task='hot')
            acc_rate = calculate_acc_rate(o3_perf, opt_perf, project="redis")
            recorder.record(f'iter={idx}, o3_perf={o3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate}\n')
            if acc_rate != float('inf') and acc_rate > -1:
                all_acc_rate.append(acc_rate)
                all_opt_perf.append(opt_perf)
                all_o3_perf.append(o3_perf)

        median_acc_rate = statistics.median(all_acc_rate)
        median_o3_perf = statistics.median(all_o3_perf)
        median_opt_perf = statistics.median(all_opt_perf)
        recorder.record(f'Final, acc_rate={median_acc_rate}')
        recorder.record(f'Final, o3_perf={median_o3_perf}')
        recorder.record(f'Final, opt_perf={median_opt_perf}')

        print(f'[main] The best acc rate is: {median_acc_rate}')
        print(f'[main] The o3_perf is: {median_o3_perf}')
        print(f'[main] The best opt perf is: {median_opt_perf}')
        pass


class FastDebugDorisEvaluator(DorisEvaluator):

    def __init__(self, o3_project_path: str, opt_project_path: str, repeat_time: int, rebuild_o3: bool = True,
                 fe_port: int = 9030, be_port: int = 9050):
        super().__init__(o3_project_path, opt_project_path, repeat_time, rebuild_o3, fe_port, be_port)

    def evaluate(self, opt: str) -> tuple[float, float, float]:
        opt_manager = DorisManager(build_dir=self.opt_project_path, repeat=20, fe_port=self.fe_port,
                                   be_port=self.be_port)
        build_res: int = opt_manager.build_be(opt_config=opt, jobs=128)
        if build_res != 0:
            return -1, -1, -1
        start_res: int = opt_manager.start_doris_be()
        if start_res != 0:
            return -1, -1, -1
        return 100, 100, 100

    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):  # TODO
        pass


class RedisEvaluator(Evaluator):
    """
    The evaluator class for redis
    """
    def __init__(self, o3_project_path: str, opt_project_path: str, repeat_time: int, rebuild_o3: bool = True, port: int=6379):
        """
        Create a redis evaluator
        :param o3_project_path: project path that will be built under o3
        :param opt_project_path: project path that will be built under turned optimization settings.
        :param repeat_time: the repeat time that conduct to evaluate the performance
        :param rebuild_o3: whether repeatedly build the project under O3
        """
        super().__init__(o3_project_path, opt_project_path, repeat_time, rebuild_o3)
        self.port = port
        self.redis_start_core = autoturning.script_args.redis_start_core
        self.redis_bench_core = autoturning.script_args.redis_bench_core


    @DeprecationWarning
    def evaluate_old(self, opt: str) -> tuple[float, float, float]:
        mode = 'opt'

        assert mode in ['opt', 'normal', 'O3']
        if mode == 'O3':
            o3_manager = RedisManager(redis_home=self.o3_project_path, enable_gperftools=False, port=50009)
            opt_manager = RedisManager(redis_home=self.opt_project_path, enable_gperftools=False, port=50008)

            o3_manager.clean()
            opt_manager.clean()

            if self.rebuild_o3 or not self.built_o3:
                o3_manager.build("-g -O3")
            opt_manager.build(opt)

            opt_avg_time, opt_avg_query_time = opt_manager.test()
            o3_avg_time, o3_avg_query_time = o3_manager.test()
            return o3_avg_time, opt_avg_time, calculate_acc_rate(o3_avg_time, opt_avg_time)
        elif mode == "opt":
            pass
        elif mode == 'O3':
            pass
        assert False, "Unreachable code!"

    def evaluate(self, opt: str) -> tuple[float, float, float]:
        redis_manager = RedisManager(
            redis_home=self.opt_project_path,
            enable_gperftools=False,
            num_repeat=self.repeat_time,
            port=self.port,
            redis_start_core=self.redis_start_core,
            redis_bench_core=self.redis_bench_core,
        )

        o3_time: Optional[float] = None
        opt_time: Optional[float] = None
        if autoturning.script_args.evaluate_search:
            redis_manager.build(opt_config=opt)
            opt_time = redis_manager.test()
            o3_time = autoturning.script_args.default_o3_perf
        if autoturning.script_args.evaluate_o3:
            redis_manager.build(opt_config='-O3')
            o3_time = redis_manager.test()
        assert o3_time is not None
        assert opt_time is not None
        return o3_time, opt_time, calculate_acc_rate(o3_perf=o3_time, opt_perf=opt_time, project='redis')

    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):  # TODO
        manager = RedisManager(
            redis_home=self.opt_project_path,
            enable_gperftools=False,
            num_repeat=1,
            port=self.port,
            redis_start_core=self.redis_start_core,
            redis_bench_core=self.redis_bench_core,
        )
        manager.build(opt_config=opt)

        o3_perf = autoturning.script_args.default_o3_perf

        all_acc_rate = []
        all_o3_perf = []
        all_opt_perf = []

        for idx in range(repeat_time):
            opt_perf = manager.test()
            acc_rate = calculate_acc_rate(o3_perf, opt_perf, project="redis")
            recorder.record(f'iter={idx}, o3_perf={o3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate}\n')
            if acc_rate != float('inf') and acc_rate > -1:
                all_acc_rate.append(acc_rate)
                all_opt_perf.append(opt_perf)
                all_o3_perf.append(o3_perf)

        median_acc_rate = statistics.median(all_acc_rate)
        median_o3_perf = statistics.median(all_o3_perf)
        median_opt_perf = statistics.median(all_opt_perf)
        recorder.record(f'Final, acc_rate={median_acc_rate}\n')
        recorder.record(f'Final, o3_perf={median_o3_perf}\n')
        recorder.record(f'Final, opt_perf={median_opt_perf}\n')

        print(f'[main] The best acc rate is: {median_acc_rate}')
        print(f'[main] The o3_perf is: {median_o3_perf}')
        print(f'[main] The best opt perf is: {median_opt_perf}')


class ScannEvaluator(Evaluator):
    """
    The evaluator class for scann
    """
    def __init__(self,
                 o3_project_path: str,
                 opt_project_path: str,
                 repeat_time: int,
                 o3_ann_dir: str,  # should be "/home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks-flags",
                 opt_ann_dir: str,  # should be /home/scy/2025.05.23.turn.scann/src/scann/ann-benchmarks-base
                 opt_env_name: str = "scannbase",
                 o3_env_name: str = "scanno3",
                 dataset: str = 'sift-128-euclidean',
                 rebuild_o3: bool = True):
        """
        Create a redis evaluator
        :param o3_project_path: project path that will be built under o3
        :param opt_project_path: project path that will be built under turned optimization settings.
        :param repeat_time: the repeat time that conduct to evaluate the performance
        :param rebuild_o3: whether repeatedly build the project under O3
        """
        super().__init__(o3_project_path, opt_project_path, repeat_time, rebuild_o3)
        self.o3_ann_dir: str = o3_ann_dir
        self.opt_ann_dir: str = opt_ann_dir
        self.opt_env_name: str = opt_env_name
        self.o3_env_name: str = o3_env_name
        self.dataset: str = dataset

    @DeprecationWarning
    def evaluate_old(self, opt: str) -> tuple[float, float, float]:
        o3_manager = SCANNManager(build_dir=self.o3_project_path,
                                  env=self.o3_env_name,
                                  ann_dir=self.o3_ann_dir,
                                  datasets=[self.dataset])
        opt_manager = SCANNManager(build_dir=self.opt_project_path,
                                   env=self.opt_env_name,
                                   ann_dir=self.opt_ann_dir,
                                   datasets=[self.dataset])
        print('[evaluate] build and test start')

        mode = 'opt'
        o3_result: Optional[dict[str, int]] = None
        opt_result: Optional[dict[str, int]] = None
        if mode == 'normal':
            o3_result, opt_result = SCANNManager.test_together(manager1=o3_manager, manager2=opt_manager, flag1="-O3",
                                                               flag2=opt)
        elif mode == 'opt':
            opt_manager.build(opt)
            opt_result = opt_manager.test()
            if opt_result is None or not isinstance(opt_result, dict):
                o3_result = None
            else:
                o3_result = dict()
                for key in opt_result:
                    o3_result.setdefault(key, 460)
        elif mode == 'O3':
            o3_manager.build("-O3")
            o3_result = o3_manager.test()
            opt_result = o3_result[:]

        print(f'[evaluate] build and test end')
        print(f'[evaluate] o3_result={o3_result}')
        print(f'[evaluate] opt_result={opt_result}')

        o3_performance: float = -1
        opt_performance: float = -1
        acc: float = 0

        if o3_result is not None and isinstance(o3_result, dict):
            o3_performance = sum(o3_result.values())
        if opt_result is not None and isinstance(opt_result, dict):
            opt_performance = sum(opt_result.values())

        print(f'[evaluate] {[o3_performance, opt_performance]}')

        if o3_result is not None and opt_result is not None and \
                isinstance(o3_result, dict) and \
                isinstance(opt_result, dict) and \
                len(o3_result) == len(opt_result):
            for key in o3_result:
                acc += opt_result[key] / o3_result[key]

        acc = acc / (len(o3_result) if o3_result is not None and len(o3_result) != 0 else 1)

        print(f'[scann.evaluate] {o3_performance, opt_performance, acc}')
        return o3_performance, opt_performance, acc

    def evaluate(self, opt: str) -> tuple[float, float, float]:
        manager = SCANNManager(build_dir=self.opt_project_path,
                               env=self.opt_env_name,
                               ann_dir=self.opt_ann_dir,
                               datasets=[self.dataset],
                               num_repeat=self.repeat_time)
        opt_perf: Optional[float] = None
        o3_perf: Optional[float] = None
        acc_rate: Optional[float] = None
        if autoturning.script_args.evaluate_search:
            manager.build(opt_config=opt)
            opt_perf, acc_rate = manager.test()
            o3_perf = autoturning.script_args.default_o3_perf
        if autoturning.script_args.evaluate_o3:
            manager.build(opt_config='-O3')
            o3_perf, _ = manager.test()
        assert opt_perf is not None
        assert o3_perf is not None
        assert acc_rate is not None
        return o3_perf, opt_perf, acc_rate

    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):
        manager = SCANNManager(build_dir=self.opt_project_path,
                               env=self.opt_env_name,
                               ann_dir=self.opt_ann_dir,
                               datasets=[self.dataset])
        manager.build(opt_config=opt)

        o3_perf = autoturning.script_args.default_o3_perf

        all_acc_rate = []
        all_o3_perf = []
        all_opt_perf = []

        for idx in range(repeat_time):
            opt_perf, acc_rate = manager.test()
            recorder.record(f'iter={idx}, o3_perf={o3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate}\n')
            if acc_rate != float('inf') and acc_rate > -1:
                all_acc_rate.append(acc_rate)
                all_opt_perf.append(opt_perf)
                all_o3_perf.append(o3_perf)

        median_acc_rate = statistics.median(all_acc_rate)
        median_o3_perf = statistics.median(all_o3_perf)
        median_opt_perf = statistics.median(all_opt_perf)
        recorder.record(f'Final, acc_rate={median_acc_rate}')
        recorder.record(f'Final, o3_perf={median_o3_perf}')
        recorder.record(f'Final, opt_perf={median_opt_perf}')

        print(f'[main] The best acc rate is: {median_acc_rate}')
        print(f'[main] The o3_perf is: {median_o3_perf}')
        print(f'[main] The best opt perf is: {median_opt_perf}')


class RandomEvaluator(Evaluator):

    def __init__(self, o3_project_path: str, opt_project_path: str, repeat_time: int):
        super().__init__(o3_project_path, opt_project_path, repeat_time)

    def evaluate(self, opt: str) -> tuple[float, float, float]:
        """
        Randomly return values from [-1, 1]
        :param opt:
        :return:
        """
        o3_perf: float = -1
        opt_perf: float = -1
        if random.randint(0, 9) <= 8:
            o3_perf = float(random.randint(1, 100))
        if random.randint(0, 9) <= 8:
            opt_perf = float(random.randint(1, 100))
        return o3_perf, opt_perf, calculate_acc_rate(o3_perf, opt_perf, project='doris')

    def evaluate_build_once(self, opt: str, recorder: Recorder, repeat_time: int):  # TODO
        pass

