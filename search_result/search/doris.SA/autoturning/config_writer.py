import math
from typing import Union

from autoturning.utils import get_opt_negation
from autoturning.metadata import OptionMetadata, AI4CMetaData, DataType, GCCParamMetadata


class OptionConfigWriter:
    """
    This class aims to convert option config in memory to optimization option command lines.
    """
    def __init__(self, option_metadata: OptionMetadata):
        self.option_metadata = option_metadata

    def write(self, config: dict[str, Union[str, int, bool, list]], mode: str="absolute") -> str:
        assert mode in ['absolute', 'relative']

        if config is None or len(config.keys()) == 0:
            return ''

        options = []
        for param_name in config:
            if self.option_metadata.elems[param_name].type == DataType.BOOL:
                if mode == 'relative':
                    if not config[param_name] and self.option_metadata.elems[param_name].default:
                        options.append(get_opt_negation(param_name))
                    elif config[param_name] and not self.option_metadata.elems[param_name].default:
                        options.append(param_name)
                elif mode == 'absolute':
                    if not config[param_name]:
                        options.append(get_opt_negation(param_name))
                    elif config[param_name]:
                        options.append(param_name)
                else:
                    assert False, "Unreachable code!"
            elif self.option_metadata.elems[param_name].type == DataType.ENUM:
                if config[param_name] != 'NONE':
                    options.append(f'{param_name}={config[param_name]}')
            elif self.option_metadata.elems[param_name].type == DataType.NUM:
                real_value = config[param_name]
                if param_name == '-fpack-struct':
                    real_value = int(math.pow(2, real_value))
                options.append(f'{param_name}={real_value}')
            elif self.option_metadata.elems[param_name].type == DataType.NM:
                options.append(f'{param_name}={config[param_name][0]},{config[param_name][1]}')
            elif self.option_metadata.elems[param_name].type == DataType.NMNM:
                options.append(
                    f'{param_name}={config[param_name][0]}:{config[param_name][1]}:{config[param_name][2]}:{config[param_name][3]}')
        for o in options:
            print(o)
        return ' '.join(options)


class GCCParamConfigWriter:
    def __init__(self, gcc_param_metadata: GCCParamMetadata):
        self.gcc_param_metadata = gcc_param_metadata

    def write(self, config: dict[str, int]) -> str:
        params = []
        if self.gcc_param_metadata is None or len(config.keys()) == 0:
            return ''
        for param_name in self.gcc_param_metadata.keys:
            param = f'--param {param_name}={config[param_name]}'
            params.append(param)
        return ' '.join(params)


class AI4CConfigWriter:
    def __init__(self, ai4c_metadata: AI4CMetaData, ai4c_yaml_dir: str):
        self.ai4c_metadata = ai4c_metadata
        self.iter = 0
        self.ai4c_yaml_dir = ai4c_yaml_dir

    def write(self, config: dict[str, dict[str, Union[int, bool]]]) -> str:

        def write_args(single_conf: dict[str, Union[int, bool]]) -> str:
            # Args: [
            # {flag_prefetch_loop_arrays: '1'}, {param_prefetch_latency: 1087}, {param_simultaneous_prefetches: 61},
            # {flag_tree_vrp: '0'}, {flag_inline_small_functions: '0'}, {flag_tree_tail_merge: '0'},
            # {flag_tree_slp_vectorize: '0'}, {flag_tree_pre: '0'}, {flag_tree_builtin_call_dce: '1'},
            # {flag_finite_loops: '0'}, {flag_optimize_sibling_calls: '1'}, {flag_schedule_insns: '1'},
            # {flag_schedule_insns_after_reload: '0'}, {flag_unroll_loops: '0'}, {param_max_inline_insns_auto: 75},
            # {param_inline_unit_growth: 32}, {param_max_inline_recursive_depth_auto: 7},
            # {param_large_function_insns: 47}, {param_large_unit_insns: 10214}, {param_max_unrolled_insns: 1598},
            # {param_max_average_unrolled_insns: 392}, {param_max_unroll_times: 28},
            # {param_min_insn_to_prefetch_ratio: 4}, {param_prefetch_min_insn_to_mem_ratio: 1}
            # ]
            args = []
            for param_name in self.ai4c_metadata.param_metadata.keys:
                if self.ai4c_metadata.param_metadata.elems[param_name].type == DataType.NUM:
                    args.append('{' + f"{param_name}: {single_conf[param_name]}" + '}')
                elif self.ai4c_metadata.param_metadata.elems[param_name].type == DataType.BOOL:
                    if single_conf[param_name]:
                        args.append('{' + f"{param_name}: '1'" + '}')
                    else:
                        args.append('{' + f"{param_name}: '0'" + '}')
            return f'Args: [{", ".join(args)}]'

        """
        --- !AutoTuning {
            Args: [
                {flag_prefetch_loop_arrays: '1'}, {param_prefetch_latency: 1087}, {param_simultaneous_prefetches: 61}, 
                {flag_tree_vrp: '0'}, {flag_inline_small_functions: '0'}, {flag_tree_tail_merge: '0'}, 
                {flag_tree_slp_vectorize: '0'}, {flag_tree_pre: '0'}, {flag_tree_builtin_call_dce: '1'}, 
                {flag_finite_loops: '0'}, {flag_optimize_sibling_calls: '1'}, {flag_schedule_insns: '1'}, 
                {flag_schedule_insns_after_reload: '0'}, {flag_unroll_loops: '0'}, {param_max_inline_insns_auto: 75}, 
                {param_inline_unit_growth: 32}, {param_max_inline_recursive_depth_auto: 7}, 
                {param_large_function_insns: 47}, {param_large_unit_insns: 10214}, {param_max_unrolled_insns: 1598}, 
                {param_max_average_unrolled_insns: 392}, {param_max_unroll_times: 28}, 
                {param_min_insn_to_prefetch_ratio: 4}, {param_prefetch_min_insn_to_mem_ratio: 1}
            ], 
            CodeRegionHash: 1903313707217807969, 
            CodeRegionType: function, 
            DebugLoc: {
                Column: 6, 
                File: /home/scy/2025.02.13.attempt.fuction.tuning/src/redisopt1/redis-6.0.20/src/module.c, 
                Line: 6763
            }, 
            Invocation: 0, 
            Name: RM_ScanCursorDestroy, 
            Pass: coarse_option_generate
        }
        """

        if len(config.keys()) == 0:
            return ''

        yaml_file = f'{self.ai4c_yaml_dir}/{self.iter}.txt'
        self.iter += 1

        all_contents = []
        for func_hash in self.ai4c_metadata.func_metadata.keys:
            args_part = write_args(config[func_hash])
            code_region_hash_part = f'CodeRegionHash: {func_hash}'
            code_region_type_part = f'CodeRegionType: {self.ai4c_metadata.func_metadata.elems[func_hash].code_region_type}'
            debug_loc_part = str(self.ai4c_metadata.func_metadata.elems[func_hash].debug_loc)
            invocation_part = f'Invocation: {self.ai4c_metadata.func_metadata.elems[func_hash].invocation}'
            name_part = f'Name: {self.ai4c_metadata.func_metadata.elems[func_hash].name}'
            pass_part = f'Pass: {self.ai4c_metadata.func_metadata.elems[func_hash].pass_name}'
            oneline_content = [args_part, code_region_hash_part, code_region_type_part, debug_loc_part, invocation_part,
                               name_part, pass_part]
            oneline = '!AutoTuning {' + f'{", ".join(oneline_content)}' + '}'
            all_contents.append(oneline + '\n--- ')
        f = open(yaml_file, 'a+')
        f.writelines(all_contents)
        f.close()
        return yaml_file


if __name__ == '__main__':
    from metadata_parser import OptionMetaDataParser
    from config_parser import RandomOptionConfigParser
    from autoturning.script_args import TurnWhat

    option_parser = OptionMetaDataParser(file='../config/option.txt')
    option_metadata = option_parser.parse()
    random_option_config_parser = RandomOptionConfigParser(option_metadata)
    random_option_config = random_option_config_parser.parse()

    from metadata_parser import AI4CMetaDataParser
    from config_parser import RandomAI4CConfigParser

    ai4c_parser = AI4CMetaDataParser(metadata_file='../config/ai4c.inline.txt',
                                     config_file='../config/doris/important.doris.function.yaml')
    ai4c_metadata = ai4c_parser.parse()
    random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    random_ai4c_config = random_ai4c_config_parser.parse()

    turn_mode: TurnWhat = TurnWhat.FUNC

    print(turn_mode)
    print(turn_mode is TurnWhat.FUNC)

    from codec import encode, decode
    c = encode(ai4c_config=random_ai4c_config,
               ai4c_metadata=ai4c_metadata,
               option_config=random_option_config,
               option_metadata=option_metadata,
               mode=turn_mode)

    option_config, ai4c_config = decode(ai4c_metadata=ai4c_metadata,
                                        code=c,
                                        option_metadata=option_metadata,
                                        mode=turn_mode)

    option_config_writer = OptionConfigWriter(option_metadata)
    ai4c_config_writer = AI4CConfigWriter(ai4c_metadata, '../yamls')

    opt = option_config_writer.write(option_config)
    ai4c = ai4c_config_writer.write(ai4c_config)

    print('aaa')
    print(opt)
    print('aaa')
    print(ai4c)
