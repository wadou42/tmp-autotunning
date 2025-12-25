import random
from typing import Union, Optional

import autoturning.script_args
from autoturning.metadata import OptionMetadata, AI4CMetaData, DataType, ParamMetadataItem, GCCParamMetadata
from autoturning.utils import formalize_code_bit


def encode(option_config: dict[str, Union[str, bool, int, list]],
           ai4c_config: dict[str, dict[str, Union[int, bool]]],
           gcc_param_config: dict[str, int],
           option_metadata: OptionMetadata,
           ai4c_metadata: AI4CMetaData,
           gcc_param_metadata: GCCParamMetadata) -> list[int]:
    """
    Encode the given configuration.

    The input configuration contains two pars:
    1. option_config: which is the optimization option configuration.
    2. ai4c_config: which is the parameter setting for each function in the project under auto-turning.
    :param option_config:
    :param ai4c_config:
    :param option_metadata:
    :param ai4c_metadata:
    :return: a big big big vector.
    """
    def encode_single(val: Union[str, bool, int, list],
                      t: DataType,
                      val_range: list[Union[str, bool, int]]) -> list[int]:
        if isinstance(val, list):
            return val
        else:
            if t in [DataType.BOOL, DataType.ENUM]:
                if t in [DataType.BOOL, DataType.ENUM] and val == 'OFF':
                    idx = random.randint(0, len(val_range)-1)
                else:
                    idx = val_range.index(val)
                return [(idx + 1) * 100]  # e.g., [False, True] => [(0, 100], (100, 200]]
            else:
                return [val]

    code: list[int] = []

    if autoturning.script_args.turn_option:
        for param_name in option_metadata.keys:
            # Options removed during constraint processing (such as mutual exclusion, dependency, or forced closure)
            # will not be encoded or appear in subsequent optimization steps.

            if param_name not in option_config:
                continue
            c = encode_single(val=option_config[param_name],
                              t=option_metadata.elems[param_name].type,
                              val_range=option_metadata.elems[param_name].val_range)
            code.extend(c)
    if autoturning.script_args.turn_func:
        for code_hash in ai4c_metadata.func_metadata.keys:
            for param_name in ai4c_metadata.param_metadata.keys:
                c = encode_single(val=ai4c_config[code_hash][param_name],
                                  t=ai4c_metadata.param_metadata.elems[param_name].type,
                                  val_range=ai4c_metadata.param_metadata.elems[param_name].val_range)
                code.extend(c)
    if autoturning.script_args.turn_param:
        for param_name in gcc_param_metadata.keys:
            c = encode_single(val=gcc_param_config[param_name],
                              t=gcc_param_metadata.elems[param_name].type,
                              val_range=gcc_param_metadata.elems[param_name].val_range)
            code.extend(c)
    return code


def decode(code: list[int],
           option_metadata: OptionMetadata,
           ai4c_metadata: AI4CMetaData,
           gcc_param_metadata: GCCParamMetadata) -> \
        tuple[dict[str, Union[str, bool, int, list]], dict[str, dict[str, Union[int, bool]]], dict[str, int]]:
    """
    Decode the vector to option config and ai4c config.
    :param code: a big big big vector
    :param option_metadata:
    :param ai4c_metadata:
    :return: option config and ai4c config, each of which is represented as a dict
    """

    def decode_single(code: list[int], idx: int, param_metadata: ParamMetadataItem) -> \
            tuple[int, Union[str, bool, int, list]]:
        if param_metadata.width != 1:  # i.e., param_metadata.type in [DataType.NMNM, DataType.NM]
            offset = 0
            ret_val = []
            while offset < param_metadata.width:
                # here we use the knowledge that DataType.NMNM, DataType.NM always has special value,
                # and the last elements of the last value range is always the MAX value of this parameter.
                code[idx + offset] = formalize_code_bit(code[idx + offset], param_metadata)
                ret_val.append(code[idx + offset])
                offset += 1
            idx += offset
            return idx, ret_val
        elif param_metadata.type in [DataType.BOOL, DataType.ENUM]:
            code[idx] = formalize_code_bit(code[idx], param_metadata)
            ret_idx = int((code[idx] - 1) / 100)
            ret_val = param_metadata.val_range[ret_idx]
            return idx + 1, ret_val
        elif param_metadata.type in [DataType.NUM]:
            code[idx] = formalize_code_bit(code[idx], param_metadata)
            ret_val = code[idx]
            return idx + 1, ret_val
        else:
            assert False, "Unreachable code"

    # def is_open(metadata: ParamMetadataItem,
    #             option_config: dict[str, Union[int, str, list[Union[int, str]]]]):
    #     if metadata.type is DataType.BOOL:
    #         return option_config[metadata.name]
    #     elif metadata.type is DataType.ENUM:
    #         return option_config[metadata.name] != 'NONE'
    #     else:
    #         assert False
    #
    # def solve_must_close_constrains(code: list[int],
    #                                 idx: int,
    #                                 metadata: ParamMetadataItem,
    #                                 option_config: dict[str, Union[int, str, list[Union[int, str]]]]):
    #     assert metadata.type in [DataType.BOOL, DataType.ENUM]
    #     if metadata.type is DataType.BOOL:
    #         option_config[metadata.name] = False
    #         code[idx] = (metadata.val_range.index(False) + 1) * 100
    #     elif metadata.type is DataType.ENUM:
    #         option_config[metadata.name] = 'NONE'
    #         code[idx] = len(metadata.val_range) * 100
    #
    # def solve_must_open_constrains(code: list[int],
    #                                idx: int,
    #                                metadata: ParamMetadataItem,
    #                                option_config: dict[str, Union[int, str, list[Union[int, str]]]]):
    #     assert metadata.type in [DataType.BOOL, DataType.ENUM]
    #     if metadata.type is DataType.BOOL:
    #         option_config[metadata.name] = True
    #         code[idx] = (metadata.val_range.index(True) + 1) * 100
    #     elif metadata.type is DataType.ENUM:
    #         rand_idx = random.randint(0, len(metadata.val_range)-1)
    #         option_config[metadata.name] = metadata.val_range[rand_idx]
    #         code[idx] = (rand_idx+1) * 100
    #
    # def solve_coexists_constrains(code: list[int],
    #                               idx1: int,
    #                               idx2: int,
    #                               metadata1: ParamMetadataItem,
    #                               metadata2: ParamMetadataItem,
    #                               option_config: dict[str, Union[int, str, list[Union[int, str]]]]):
    #
    #     correct_use = metadata1.type in [DataType.BOOL, DataType.ENUM] or \
    #                   metadata2.type in [DataType.BOOL, DataType.ENUM]
    #     assert correct_use
    #
    #     if is_open(metadata1, option_config) and is_open(metadata2, option_config):    # if coexists.
    #         if random.randint(0, 1) == 0:   # close the bool
    #             solve_must_close_constrains(code, idx1, metadata1, option_config)
    #         else:  # close the enum, we choose the last element of the value range is the close one.
    #             solve_must_close_constrains(code, idx2, metadata2, option_config)
    #
    # def solve_must_close_latter_when_close_former(code: list[int],
    #                                               idx1: int,
    #                                               idx2: int,
    #                                               metadata1: ParamMetadataItem,
    #                                               metadata2: ParamMetadataItem,
    #                                               option_config: dict[str, Union[int, str, list[Union[int, str]]]]):
    #     correct_use = metadata1.type is DataType.BOOL and metadata2.type is DataType.BOOL
    #     assert correct_use
    #     if not is_open(metadata1, option_config):
    #         solve_must_close_constrains(code, idx2, metadata2, option_config)

    option_config: dict[str, Union[str, bool, int, list]] = dict()
    ai4c_config: dict[str, dict[str, Union[int, bool]]] = dict()
    gcc_param_config: dict[str, int] = dict()
    idx = 0

    # f_ipf_pta_idx: Optional[int] = None
    # f_ipf_pta_metadata: Optional[ParamMetadataItem] = None
    # f_live_patching_idx: Optional[int] = None
    # f_live_patching_metadata: Optional[ParamMetadataItem] = None
    # f_toplevel_reorder_idx: Optional[int] = None
    # f_toplevel_reorder_metadata: Optional[ParamMetadataItem] = None
    # f_section_anchors_idx: Optional[int] = None
    # f_section_anchors_metadata: Optional[ParamMetadataItem] = None
    # f_inline_automics_idx: Optional[int] = None
    # f_inline_automics_metadata: Optional[ParamMetadataItem] = None
    # # -fno-reg-struct-return
    # f_reg_struct_return_idx: Optional[int] = None
    # f_reg_struct_return_metadata: Optional[ParamMetadataItem] = None

    if autoturning.script_args.turn_option:
        for param_name in option_metadata.keys:
            # if param_name == '-fipa-pta':
            #     f_ipf_pta_idx = idx
            #     f_ipf_pta_metadata = option_metadata.elems[param_name]
            # if param_name == '-flive-patching':
            #     f_live_patching_idx = idx
            #     f_live_patching_metadata = option_metadata.elems[param_name]
            # if param_name == '-ftoplevel-reorder':
            #     f_toplevel_reorder_idx = idx
            #     f_toplevel_reorder_metadata = option_metadata.elems[param_name]
            # if param_name == '-fsection-anchors':
            #     f_section_anchors_idx = idx
            #     f_section_anchors_metadata = option_metadata.elems[param_name]
            # if param_name == '-finline-atomics':
            #     f_inline_automics_idx = idx
            #     f_inline_automics_metadata = option_metadata.elems[param_name]
            # if param_name == '-freg-struct-return':
            #     f_reg_struct_return_idx = idx
            #     f_reg_struct_return_metadata = option_metadata.elems[param_name]

            idx, ret_val = decode_single(code=code, idx=idx, param_metadata=option_metadata.elems[param_name])
            option_config.setdefault(param_name, ret_val)

        # tar_part1 = f'-finline-atomics={option_config["-finline-atomics"]}, -finline-atomics={option_config["-finline-atomics"]}'
        # tar_part2 = f'code[-finline-atomics]={code[f_inline_automics_idx]}, code[-finline-atomics]={code[f_inline_automics_idx]}'
        # print(f'[decode] before `solve_coexists_constrains`: {tar_part1}, {tar_part2}')

        # if f_ipf_pta_metadata is not None and f_live_patching_metadata is not None:
        #     solve_coexists_constrains(code=code,
        #                               idx1=f_ipf_pta_idx,
        #                               idx2=f_live_patching_idx,
        #                               metadata1=f_ipf_pta_metadata,
        #                               metadata2=f_live_patching_metadata,
        #                               option_config=option_config)
        # if f_toplevel_reorder_metadata is not None and f_section_anchors_metadata is not None:
        #     solve_must_close_latter_when_close_former(code=code,
        #                                               idx1=f_toplevel_reorder_idx,
        #                                               idx2=f_section_anchors_idx,
        #                                               metadata1=f_toplevel_reorder_metadata,
        #                                               metadata2=f_section_anchors_metadata,
        #                                               option_config=option_config)
        # if f_inline_automics_metadata is not None:
        #     solve_must_close_constrains(code=code,
        #                                 idx=f_inline_automics_idx,
        #                                 metadata=f_inline_automics_metadata,
        #                                 option_config=option_config)
        #
        # if f_reg_struct_return_metadata is not None:
        #     solve_must_open_constrains(code=code,
        #                                idx=f_reg_struct_return_idx,
        #                                metadata=f_reg_struct_return_metadata,
        #                                option_config=option_config)

        # tar_part1 = f'-finline-atomics={option_config["-finline-atomics"]}, -finline-atomics={option_config["-finline-atomics"]}'
        # tar_part2 = f'code[-finline-atomics]={code[f_inline_automics_idx]}, code[-finline-atomics]={code[f_inline_automics_idx]}'
        # print(f'[decode] after `solve_coexists_constrains`: {tar_part1}, {tar_part2}')

    if autoturning.script_args.turn_func:
        for code_hash in ai4c_metadata.func_metadata.keys:
            for param_name in ai4c_metadata.param_metadata.keys:
                idx, ret_val = decode_single(code=code, idx=idx, param_metadata=ai4c_metadata.param_metadata.elems[param_name])
                ai4c_config.setdefault(code_hash, dict()).setdefault(param_name, ret_val)

    if autoturning.script_args.turn_param:
        for param_name in gcc_param_metadata.keys:
            idx, ret_val = decode_single(code=code, idx=idx, param_metadata=gcc_param_metadata.elems[param_name])
            gcc_param_config.setdefault(param_name, ret_val)

    if autoturning.script_args.turn_option and autoturning.script_args.turn_func:
        assert idx == len(code)
    return option_config, ai4c_config, gcc_param_config


if __name__ == '__main__':
    from metadata_parser import OptionMetaDataParser
    from config_parser import RandomOptionConfigParser

    option_parser = OptionMetaDataParser(file='../config/option.txt')
    option_metadata = option_parser.parse()
    random_option_config_parser = RandomOptionConfigParser(option_metadata)
    random_option_config = random_option_config_parser.parse()

    from metadata_parser import AI4CMetaDataParser
    from config_parser import RandomAI4CConfigParser

    ai4c_parser = AI4CMetaDataParser(metadata_file='../config/ai4c.txt', config_file='../config/function.yaml')
    ai4c_metadata = ai4c_parser.parse()
    random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    random_ai4c_config = random_ai4c_config_parser.parse()

    c = encode(ai4c_config=random_ai4c_config,
               ai4c_metadata=ai4c_metadata,
               option_config=random_option_config,
               option_metadata=option_metadata)

    option_config, ai4c_config = decode(ai4c_metadata=ai4c_metadata, code=c, option_metadata=option_metadata)

    for param_name in random_option_config:
        assert random_option_config[param_name] == option_config[
            param_name], f'{random_option_config[param_name]} neq to {option_config[param_name]}'
    for code_hash in random_ai4c_config:
        for param_name in random_ai4c_config[code_hash]:
            assert random_ai4c_config[code_hash][param_name] == ai4c_config[code_hash][param_name]
    assert len(random_option_config.keys()) == len(option_config.keys())
    assert len(random_ai4c_config.keys()) == len(ai4c_config.keys())
