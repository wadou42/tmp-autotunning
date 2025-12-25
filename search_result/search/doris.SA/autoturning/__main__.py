"""
Created by chenyaosuo
This is the main file of the universal auto-turning tool.
"""
import statistics
import sys
import argparse

import autoturning.script_args
from autoturning.config_parser import *
from autoturning.searcher import *
from autoturning.script_args import *
from autoturning.metadata_parser import *
from autoturning.evaluator import *


if __name__ == '__main__':

    SCRIPT_START_TIME = time.time()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    def trace_calls(frame, event, arg):
        if event == 'call':
            co = frame.f_code
            func_name = co.co_name
            # if co.co_filename.startswith('<') or \
            #         'site-packages' in co.co_filename or \
            #         '__main__' in sys.modules and sys.modules['__main__'].__package__ not in co.co_filename:
            #     return

            file_name = os.path.basename(co.co_filename)

            class_name = None
            if 'self' in frame.f_locals:
                self_obj = frame.f_locals['self']
                if hasattr(self_obj, '__class__'):
                    class_name = self_obj.__class__.__name__
            elif 'cls' in frame.f_locals:
                cls_obj = frame.f_locals['cls']
                if isinstance(cls_obj, type):
                    class_name = cls_obj.__name__

            start_time = time.time()
            func_iden = f'{file_name}:{class_name}.{func_name}'
            if True:
                print(f'[{"{:.2f}".format(start_time - SCRIPT_START_TIME)}][{func_iden}] start')
            else:
                arg_names = frame.f_code.co_varnames[:frame.f_code.co_argcount]
                arg_values = [frame.f_locals[name] for name in arg_names]
                arg_types = [type(val).__name__ for val in arg_values]
                # do not output parameter info
                params = f'({", ".join([f"{_[0]} {_[1]}={_[2]}" for _ in zip(arg_types, arg_names, arg_values)])})'
                print(f'[{"{:.2f}".format(start_time - SCRIPT_START_TIME)}][{func_iden}{params}] start')

            def trace_returns(frame, event, arg):
                if event == 'return':
                    end_time = time.time()
                    print(f'[{"{:.2f}".format(end_time - SCRIPT_START_TIME)}][{file_name}:{class_name}.{func_name}] end, run {"{:.2f}".format(end_time - start_time)} seconds.')
                elif event == 'exception':
                    sys.settrace(None)
                    return None
                return trace_returns

            return trace_returns
        return None

    sys.settrace(trace_calls)

    EXISTING_SEARCHER = ['GA', 'RANDOM', 'PSO', 'SA', 'DE']
    MUTATION_BASED_SEARCHER = ['GA', 'RANDOM', 'SA', 'DE']
    GA_BASED_SEARCHER = ['GA', 'DE']

    parser = argparse.ArgumentParser(
        description='An unified compiler auto-turning tool.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # arguments relate to metadata
    parser.add_argument('--option_meta_config',
                        type=str,
                        default="config/option.txt",
                        help='The path of optimization option metadata configuration.')
    parser.add_argument('--ai4c_meta_config',
                        type=str,
                        default="config/ai4c.txt",
                        help='The path of ai4c metadata configuration.')
    parser.add_argument('--param_meta_config',
                        type=str,
                        default="config/param.important.txt",
                        help='The path of gcc parameter metadata configuration.')
    parser.add_argument('--existing_ai4c_config',
                        type=str,
                        default='config/function.yaml',
                        help='The path of existing ai4c config')
    parser.add_argument('--existing_constrains',
                        type=str,
                        default=None,
                        help='The path of existing constrains file')

    # arguments relate to existing configurations.
    parser.add_argument('--existing_option_config_file',
                        type=str,
                        default=None,
                        help='The path of existing optimization option config')
    # TODO: add existing ai4c configuration dir, add the logic to specify the starting configuration.

    # arguments relate to search
    parser.add_argument('--searcher',
                        type=str_to_searcher_type,
                        default=None,
                        help=f"The name of searcher, either: {[_.name for _ in SearcherType]}")
    parser.add_argument('--evaluation_cnt_limit',
                        type=int,
                        default=50,
                        help="The limit of evaluation number.")
    parser.add_argument("--time_limit",
                        type=int,
                        default=60*60*24*5,
                        help="The limit of time cost.")
    parser.add_argument("--mutation_bit_rate",
                        type=float,
                        default=None,
                        help=f"The percentage of mutation bits when using mutation-based searchers, i.e., searchers in {MUTATION_BASED_SEARCHER}.")
    parser.add_argument("--crossover_bit_rate",
                        type=float,
                        default=None,
                        help=f"The percentage of crossover bits when using GA-baesd searchers, i.e., searchers in {GA_BASED_SEARCHER}")
    parser.add_argument("--init_num",
                        type=int,
                        default=10,
                        help=f"The number of initial configurations.")
    parser.add_argument("--population",
                        type=int,
                        default=5,
                        help=f"Population.")
    parser.add_argument("--default_o3_perf",
                        type=float,
                        help="Default O3 performance.")
    parser.add_argument("--pso_w",
                        type=float,
                        default=0.7,
                        help="The w parameter of PSO algorithm.")
    parser.add_argument("--pso_c1",
                        type=float,
                        default=1.4,
                        help="The c1 parameter of PSO algorithm.")
    parser.add_argument("--pso_c2",
                        type=float,
                        default=1.4,
                        help="The c2 parameter of PSO algorithm.")
    parser.add_argument("--pso_v_rate",
                        type=float,
                        default=0.1,
                        help="The v_rate parameter of PSO algorithm.")
    parser.add_argument("--sa_temperature",
                        type=float,
                        default=100.0,
                        help="The temperature parameter of SA algorithm.")
    parser.add_argument("--sa_cooling_rate",
                        type=float,
                        default=0.95,
                        help="The cooling_rate parameter of SA algorithm.")

    # arguments relate to project under turning
    parser.add_argument('--project',
                        type=str_to_evaluator_type,
                        default=None,
                        help=f"The project name under turning, either {[_.name for _ in EvaluatorType]}")

    # arguments for all project
    parser.add_argument('--o3_project_path',
                        type=str,
                        default=None,
                        help='The path of project that will be built under O3.')
    parser.add_argument('--opt_project_path',
                        type=str,
                        default=None,
                        help="The path of project that will be built under optimization setting.")
    parser.add_argument('--repeat',
                        type=int,
                        default=15,
                        help="The limit of evaluation number.")

    # arguments for redis
    parser.add_argument('--redis_port',
                        type=int,
                        default=6379,
                        help=f"The port of redis project.")
    parser.add_argument('--redis_start_core',
                        type=str,
                        help="The taskset core for redis. For 1.2, it should be 319.")
    parser.add_argument('--redis_bench_core',
                        type=str,
                        help="The taskset core for redis. For 1.2, it should be 311.")

    # arguments for doris
    parser.add_argument('--doris_fe_port',
                        type=int,
                        default=9030,
                        help=f"The port of the front end.")
    parser.add_argument('--doris_be_port',
                        type=int,
                        default=9050,
                        help=f"The port of the back end.")

    # arguments for scann
    parser.add_argument("--scann_o3_ann_dir",
                        type=str,
                        default=None,
                        help="The path of o3 benchmark directory")
    parser.add_argument("--scann_opt_ann_dir",
                        type=str,
                        default=None,
                        help="The path of opt benchmark directory")
    parser.add_argument("--scann_o3_env",
                        type=str,
                        default=None,
                        help="The o3 conda environment")
    parser.add_argument("--scann_opt_env",
                        type=str,
                        default=None,
                        help="The opt conda environment")

    # arguments relate to record
    parser.add_argument('--final_eva_cnt',
                        type=int,
                        default=50,
                        help="The number of evaluation count.")
    parser.add_argument('--ai4c_config_dir',
                        type=str,
                        default='ai4c_yamls',
                        help="The folder that stores all of the AI4C configurations.")
    parser.add_argument("--opt_recorder",
                        type=str,
                        default="opt.recorder.txt",
                        help="The file that record the optimization during searching process.")
    parser.add_argument("--perf_recorder",
                        type=str,
                        default="perf.recorder.txt",
                        help="The file that record the performance during searching process.")
    parser.add_argument("--final_perf_recorder",
                        type=str,
                        default="final.perf.recorder.txt",
                        help="The file that record the performance during searching process.")

    # arguments relate to ai4c project
    parser.add_argument("--ai4c_option_file",
                        type=str,
                        default='/usr/local/lib/python3.9/site-packages/ai4c/autotuner/yaml/coarse_options.yaml',
                        help="One of a ai4c configuration file, search this file according to file name of the default value.")
    parser.add_argument("--ai4c_plagin_lib",
                        type=str,
                        default='/usr/local/lib/python3.9/site-packages/ai4c/lib/coarse_option_tuning_plugin_gcc12.so',
                        help="The compiled lib of AI4C project, you can find this file according to the file name of the default setting. "
                             "Or according to the official installation help")

    # arguments controls turning and result
    parser.add_argument("--clean",
                        type=bool,
                        default=True,
                        help="Whether clean existing result.")

    parser.add_argument("--turn_option",
                        type=str,
                        default="False",
                        help="Whether search gcc option.")
    parser.add_argument("--turn_func",
                        type=str,
                        default="False",
                        help="Whether search function-level turning.")
    parser.add_argument("--turn_param",
                        type=str,
                        default="False",
                        help="Whether search gcc parameters.")

    parser.add_argument("--evaluate_search",
                        type=str,
                        default="False",
                        help="Whether evaluate the search result.")
    parser.add_argument("--evaluate_o3",
                        type=str,
                        default="False",
                        help="Whether evaluate o3.")

    args = parser.parse_args()

    # set global arguments
    autoturning.script_args.turn_option = args.turn_option in ['TRUE', 'true', 'True']
    autoturning.script_args.turn_func = args.turn_func in ['TRUE', 'true', 'True']
    autoturning.script_args.turn_param = args.turn_param in ['TRUE', 'true', 'True']

    autoturning.script_args.evaluate_search = args.evaluate_search in ['TRUE', 'true', 'True']
    autoturning.script_args.evaluate_o3 = args.evaluate_o3 in ['TRUE', 'true', 'True']

    autoturning.script_args.default_o3_perf = args.default_o3_perf

    # check searcher arguments
    assert args.searcher is not None
    assert isinstance(args.searcher, SearcherType)
    if args.searcher is SearcherType.RANDOM or \
            args.searcher is SearcherType.SA or \
            args.searcher is SearcherType.DE or \
            args.searcher is SearcherType.GA:
        assert args.mutation_bit_rate is not None
    if args.searcher is SearcherType.DE or \
            args.searcher is SearcherType.GA:
        assert args.crossover_bit_rate is not None
    assert args.init_num > args.population

    # check global arguments
    assert args.project is not None
    assert isinstance(args.project, EvaluatorType)

    # check common project arguments
    assert args.opt_project_path is not None
    assert os.path.exists(args.opt_project_path)
    if autoturning.script_args.evaluate_o3:
        assert args.o3_project_path is not None
        assert os.path.exists(args.o3_project_path)

    # check scann arguments
    if args.project is EvaluatorType.SCANN:
        assert args.scann_opt_ann_dir is not None
        assert args.scann_opt_env is not None
        assert os.path.exists(args.scann_opt_ann_dir)
        if autoturning.script_args.evaluate_o3:
            assert args.scann_o3_ann_dir is not None
            assert args.scann_o3_env is not None
            assert os.path.exists(args.scann_o3_ann_dir)
    if args.project is EvaluatorType.REDIS:
        assert args.redis_start_core is not None
        assert args.redis_bench_core is not None
        autoturning.script_args.redis_start_core = args.redis_start_core
        autoturning.script_args.redis_bench_core = args.redis_bench_core

    # clean
    if args.clean:
        os.system(f'rm -rf {args.ai4c_config_dir}')
        os.system(f'rm -f {args.opt_recorder}')
        os.system(f'rm -f {args.perf_recorder}')
    if not os.path.exists(args.ai4c_config_dir):
        os.system(f'mkdir -p {args.ai4c_config_dir}')

    # parse metadata
    option_metadata: Optional[OptionMetadata] = None
    if autoturning.script_args.turn_option:
        option_parser: OptionMetaDataParser = OptionMetaDataParser(file=args.option_meta_config)
        option_metadata = option_parser.parse()
        print(f'[main] parsed option metadata: {option_metadata}')
        if args.existing_constrains is not None:
            assert os.path.exists(args.existing_constrains)
            option_metadata.filter(args.existing_constrains)
    # else:
        

    ai4c_metadata: Optional[AI4CMetaData] = None
    if autoturning.script_args.turn_func:
        ai4c_parser = AI4CMetaDataParser(metadata_file=args.ai4c_meta_config,
                                         config_file=args.existing_ai4c_config)
        ai4c_metadata = ai4c_parser.parse()
        print(f'[main] parsed ai4c metadata: {ai4c_metadata}')

    gcc_param_metadata: Optional[GCCParamMetadata] = None
    if autoturning.script_args.turn_param:
        gcc_param_metadata_parser = GCCParamMetadataParser(file=args.param_meta_config)
        gcc_param_metadata = gcc_param_metadata_parser.parse()
        print(f'[main] parsed gcc parameter metadata: {gcc_param_metadata}')

    # parse config
    opt_settings: list[list[int]] = []
    option_config_parser: Optional[OptionConfigParser] = None
    random_ai4c_config_parser: Optional[RandomAI4CConfigParser] = None
    random_gcc_parameter_parser: Optional[RandomGCCParamConfigParser] = None
    if autoturning.script_args.turn_option:
        if args.existing_option_config_file is not None:
            option_config_parser = ExistingOptionConfigParser(metadata=option_metadata,
                                                              config_file=args.existing_option_config_file)
        else:
            option_config_parser = RandomOptionConfigParser(option_metadata)
    if autoturning.script_args.turn_func:
        random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    if autoturning.script_args.turn_param:
        random_gcc_parameter_parser = RandomGCCParamConfigParser(gcc_param_metadata)

    # encode
    for _ in range(args.init_num):
        random_option_config: dict[str, Union[int, str, bool, list]] = dict()
        random_ai4c_config: dict[str, dict[str, Union[int, str, bool]]] = dict()
        random_gcc_parameter_config: dict[str, int] = dict()
        if autoturning.script_args.turn_option:
            random_option_config = option_config_parser.parse()
        if autoturning.script_args.turn_func:
            random_ai4c_config = random_ai4c_config_parser.parse()
        if autoturning.script_args.turn_param:
            random_gcc_parameter_config = random_gcc_parameter_parser.parse()
        c: list[int] = encode(ai4c_config=random_ai4c_config,
                              ai4c_metadata=ai4c_metadata,
                              option_config=random_option_config,
                              option_metadata=option_metadata,
                              gcc_param_config=random_gcc_parameter_config,
                              gcc_param_metadata=gcc_param_metadata)
        opt_settings.append(c)

    # random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    # opt_settings = []
    # for _ in range(5):
    #     random_ai4c_config = random_ai4c_config_parser.parse()
    #     random_option_config = random_option_config_parser.parse()
    #     c = encode(ai4c_config=random_ai4c_config,
    #                ai4c_metadata=ai4c_metadata,
    #                option_config=random_option_config,
    #                option_metadata=option_metadata,
    #                mode=args.turn_what)
    #     opt_settings.append(c)

    # set the evaluator.
    evaluator: Evaluator
    if args.project is EvaluatorType.DORIS:
        evaluator = DorisEvaluator(o3_project_path=args.o3_project_path,
                                   opt_project_path=args.opt_project_path,
                                   repeat_time=args.repeat,
                                   fe_port=args.doris_fe_port,
                                   be_port=args.doris_be_port)
    elif args.project is EvaluatorType.REDIS:
        evaluator = RedisEvaluator(o3_project_path=args.o3_project_path,
                                   opt_project_path=args.opt_project_path,
                                   repeat_time=args.repeat,
                                   port=args.redis_port)
    elif args.project is EvaluatorType.SCANN:
        evaluator = ScannEvaluator(o3_ann_dir=args.scann_o3_ann_dir,
                                   o3_project_path=args.o3_project_path,
                                   opt_ann_dir=args.scann_opt_ann_dir,
                                   opt_project_path=args.opt_project_path,
                                   repeat_time=args.repeat,
                                   o3_env_name=args.scann_o3_env,
                                   opt_env_name=args.scann_opt_env)
    elif args.project is EvaluatorType.RANDOM:
        evaluator = RandomEvaluator(o3_project_path=args.o3_project_path,
                                    opt_project_path=args.opt_project_path,
                                    repeat_time=args.repeat)
    elif args.project is EvaluatorType.FAST_DORIS:
        evaluator = FastDebugDorisEvaluator(o3_project_path=args.o3_project_path,
                                            opt_project_path=args.opt_project_path,
                                            repeat_time=args.repeat,
                                            fe_port=args.doris_fe_port,
                                            be_port=args.doris_be_port)
    else:
        assert False, "Unreachable code!"
    assert evaluator is not None

    # create the search config
    search_config: SearchConfig = SearchConfig(ai4c_config_dir=f"{os.getcwd()}/{args.ai4c_config_dir}",
                                               ai4c_option_file=args.ai4c_option_file,
                                               ai4c_plagin_lib=args.ai4c_plagin_lib,
                                               opt_record_file=args.opt_recorder,
                                               perf_record_file=args.perf_recorder)

    # set the searcher.
    searcher: Searcher
    if args.searcher is SearcherType.RANDOM:
        searcher = RandomSearcher(ai4c_metadata=ai4c_metadata,
                                  evaluation_cnt_limit=args.evaluation_cnt_limit,
                                  evaluator=evaluator,
                                  mutation_bit_rate=args.mutation_bit_rate,
                                  opt_settings=opt_settings,
                                  option_metadata=option_metadata,
                                  gcc_param_metadata=gcc_param_metadata,
                                  population=args.population,
                                  search_config=search_config,
                                  constrains_file=args.existing_constrains,
                                  time_limit=args.time_limit)
    elif args.searcher is SearcherType.SA:
        searcher = SASearcher(ai4c_metadata=ai4c_metadata,
                              evaluation_cnt_limit=args.evaluation_cnt_limit,
                              evaluator=evaluator,
                              mutation_bit_rate=args.mutation_bit_rate,
                              opt_settings=opt_settings,
                              option_metadata=option_metadata,
                              gcc_param_metadata=gcc_param_metadata,
                              population=args.population,
                              search_config=search_config,
                              constrains_file=args.existing_constrains,
                              time_limit=args.time_limit,
                              temperature=args.sa_temperature,
                              cooling_rate=args.sa_cooling_rate)
    elif args.searcher is SearcherType.DE:
        searcher = DESearcher(ai4c_metadata=ai4c_metadata,
                              evaluation_cnt_limit=args.evaluation_cnt_limit,
                              evaluator=evaluator,
                              opt_settings=opt_settings,
                              option_metadata=option_metadata,
                              population=args.population,
                              crossover_bit_rate=args.crossover_bit_rate,
                              mutation_bit_rate=args.mutation_bit_rate,
                              constrains_file=args.existing_constrains,
                              search_config=search_config,
                              time_limit=args.time_limit,
                              gcc_param_metadata=gcc_param_metadata)
    elif args.searcher is SearcherType.GA:
        searcher = GASearcher(ai4c_metadata=ai4c_metadata,
                              evaluation_cnt_limit=args.evaluation_cnt_limit,
                              evaluator=evaluator,
                              mutation_bit_rate=args.mutation_bit_rate,
                              opt_settings=opt_settings,
                              option_metadata=option_metadata,
                              gcc_param_metadata=gcc_param_metadata,
                              population=args.population,
                              search_config=search_config,
                              crossover_bit=args.crossover_bit_rate,
                              constrains_file=args.existing_constrains,
                              time_limit=args.time_limit)
    elif args.searcher is SearcherType.PSO:
        searcher = PSOSearcher(ai4c_metadata=ai4c_metadata,
                               evaluation_cnt_limit=args.evaluation_cnt_limit,
                               evaluator=evaluator,
                               opt_settings=opt_settings,
                               option_metadata=option_metadata,
                               gcc_param_metadata=gcc_param_metadata,
                               population=args.population,
                               search_config=search_config,
                               constrains_file=args.existing_constrains,
                               time_limit=args.time_limit,
                               w=args.pso_w,
                               c1=args.pso_c1,
                               c2=args.pso_c2,
                               v_rate=args.pso_v_rate)
    else:
        assert False, "Unreachable branch."
    assert searcher is not None

    # conduct searching process.
    searcher.search()

    # search end, find the best configuration, and run multiple times.
    print("[main] search ends, find the best configuration")
    assert os.path.exists(args.perf_recorder)
    assert os.path.exists(args.opt_recorder)
    perf_infos: list[list[Union[int, float]]] = []
    opt_infos: dict[int, str] = dict()
    with open(args.perf_recorder) as perf_f, open(args.opt_recorder) as opt_f:
        # pattern of perf recorder: 0: O3_perf=xxx, opt_perf=xxx, acc_rate=xxx
        # pattern of opt recorder: iter: opt
        perf_lines = perf_f.readlines()
        opt_lines = opt_f.readlines()

        for p in perf_lines:
            iter_num = int(p.split(':')[0])
            o3_perf = float(p.strip().split('O3_perf=')[1].split(',')[0])
            opt_perf = float(p.strip().split('opt_perf=')[1].split(',')[0])
            acc_rate = float(p.strip().split('acc_rate=')[1].split(',')[0])
            if acc_rate != float('inf'):
                perf_infos.append([acc_rate, iter_num])

        opt_infos = {int(_.strip().split(':')[0]): _.strip().split(':')[1] for _ in opt_lines}

    best_config = opt_infos[sorted(perf_infos, reverse=True)[0][1]]

    final_recorder: Recorder = Recorder(file=args.final_perf_recorder)
    final_recorder.record(best_config + '\n')

    # all_acc_rate = []
    # all_o3_perf = []
    # all_opt_perf = []
    # for idx in range(args.final_eva_cnt):
    #     o3_perf, opt_perf, acc_rate = evaluator.evaluate(best_config)
    #     final_recorder.record(f'iter={idx}, o3_perf={o3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate}\n')
    #     if acc_rate != float('inf') and acc_rate > -1:
    #         all_acc_rate.append(acc_rate)
    #         all_opt_perf.append(opt_perf)
    #         all_o3_perf.append(o3_perf)
    # median_acc_rate = statistics.median(all_acc_rate)
    # median_o3_perf = statistics.median(all_o3_perf)
    # median_opt_perf = statistics.median(all_opt_perf)
    # final_recorder.record(f'Final, acc_rate={median_acc_rate}')
    # final_recorder.record(f'Final, o3_perf={median_o3_perf}')
    # final_recorder.record(f'Final, opt_perf={median_opt_perf}')
    #
    # print(f'[main] The best acc rate is: {median_acc_rate}')
    # print(f'[main] The o3_perf is: {median_o3_perf}')
    # print(f'[main] The best opt perf is: {median_opt_perf}')
    evaluator.evaluate_build_once(best_config, final_recorder, args.final_eva_cnt)
