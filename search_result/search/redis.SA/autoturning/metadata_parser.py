import os
import time
import json
from multiprocessing import Queue, Process

from autoturning.metadata import *
from autoturning.utils import execmd
# from autoturning.multiprocess_utils import multiprocessing_wrapper


class OptionMetaDataParser:

    """
    This class aims to parse the optimization metadata into memory, and finally generates Metadata object.
    """
    def __init__(self, file: str, base_level: str = '-O3'):
        """
        Create an option metadata parser
        :param file: the optimization option metadata configuration file
        :param base_level: the base optimization level, which is used to calculate the default status of options.
        """
        self.file = file
        self.base_level = base_level

        self.base_level_help_lines: list[str] = []
        if True:  # for debug
            def get_original_optimization_status(level):
                res = execmd(f'gcc -Q --help=optimizers {level}').split('\n')
                return res
            self.base_level_help_lines = get_original_optimization_status(self.base_level)

    def parse(self) -> OptionMetadata:  # reviewed in 2025.09.02
        """
        Parse the optimization option metadata configuration file and return the option metadata object.
        :return: option metadata object
        """
        f = open(self.file)
        lines = f.readlines()
        f.close()

        option_metadata = OptionMetadata()
        for l in lines:
            print(l)
            # if l.startswith('//'):
            #     continue
            if not l.strip().startswith('-'):
                continue
            metadata_item: GCCOptionParamMetadataItem
            name: str
            if '=' not in l:  # DataType.BOOL
                name = l.strip()
                default_open = False
                for line in self.base_level_help_lines:
                    if name in line and '[enabled]' in line:
                        default_open = True
                        break
                metadata_item = GCCOptionParamMetadataItem(item_type=DataType.BOOL, name=name, val_range=[False, True], width=1,
                                                           default=default_open)
            elif 'n:m:n2:m2' in l:  # DataType.NMNM
                # 0, 1, 2-65536
                name = l.split('=')[0]
                metadata_item = GCCOptionParamMetadataItem(item_type=DataType.NMNM, name=name, val_range=[0, 1, [2, 65536]],
                                                           width=4)
            elif 'n,m' in l:  # DataType.NM
                # 0, 1-65535
                name = l.split('=')[0]
                metadata_item = GCCOptionParamMetadataItem(item_type=DataType.NM, name=name, val_range=[0, [1, 65535]], width=2)
            elif '[' in l:  # DataType.ENUM
                name = l.split('=')[0]
                val_range = l.split('[')[1].strip().split(']')[0].split('|')
                metadata_item = GCCOptionParamMetadataItem(item_type=DataType.ENUM, name=name, val_range=val_range, width=1)
            elif '<number>' in l:  # DataType.NUM
                name = l.split('=')[0]
                val_range = [0, 65536]
                if name == '-fpack-struct':
                    val_range = [0, 4]
                metadata_item = GCCOptionParamMetadataItem(item_type=DataType.NUM, name=name, val_range=val_range, width=1)
            else:
                assert False, "Unreachable code"

            assert name is not None
            assert metadata_item is not None
            option_metadata.keys.append(name)
            option_metadata.elems.setdefault(name, metadata_item)

        return option_metadata


def parse_ai4c_line(line: str) -> list[str, AI4CFunctionMetadataItem]:
    code_region_hash = line.split('CodeRegionHash: ')[1].split(',')[0]
    code_region_type = line.split('CodeRegionType: ')[1].split(',')[0]

    column = line.split('Column: ')[1].split(',')[0].strip()
    file = line.split('File: ')[1].split(',')[0].strip()
    line_num = line.split('Line: ')[1].split(',')[0].strip()
    name = line.split('Name: ')[1].split(',')[0].strip()
    pass_name = line.split('Pass: ')[1].split(',')[0].strip()

    ai4c_func_metadata_elem = AI4CFunctionMetadataItem(code_region_hash=code_region_hash,
                                                       code_region_type=code_region_type,
                                                       column=column, file=file, line=line_num, name=name,
                                                       pass_name=pass_name)
    return [code_region_hash, ai4c_func_metadata_elem]


class AI4CMetaDataParser:
    """
    This class aims to parse the metadata information of AI4C search space from a metadata configuration file,
    and a template(but real) AI4C configuration yaml file.
    """
    def __init__(self, metadata_file: str, config_file):
        """
        Create an ai4c metadata parser.
        :param metadata_file: the metadata configuration file for ai4c search space(contains parameter information).
        :param config_file: the ai4c configuration yaml file.
        """
        self.metadata_file = metadata_file
        self.config_file = config_file

    def parse(self) -> AI4CMetaData:
        """
        Parse the metadata file and ai4c configuration yaml file, and return the metadata object
        :return: ai4c metadata object
        """
        f = open(self.metadata_file)
        lines = f.readlines()
        f.close()

        ai4c_param_metadata = AI4CParamMetadata()
        for l in lines:
            name = l.split('name=')[1].split(', ')[0].strip()
            item_type = l.split('type=')[1].split(', ')[0].strip()
            val_range = None
            if item_type == 'bool':
                val_range = [False, True]
                item_type = DataType.BOOL
            elif item_type == 'int':
                lb = int(l.split('lb=')[1].split(',')[0].strip())
                ub = int(l.split('ub=')[1].split(',')[0].strip())
                val_range = [lb, ub]
                item_type = DataType.NUM
            assert val_range is not None
            ai4c_metadata_item = AI4CParamMetadataItem(item_type=item_type, name=name, val_range=val_range, width=1)
            ai4c_param_metadata.keys.append(name)
            ai4c_param_metadata.elems.setdefault(name, ai4c_metadata_item)

        ai4c_func_metadata = AI4CFunctionMetadata()
        f = open(self.config_file)
        lines = f.readlines()
        f.close()

        # The old version of parse.
        for line in lines:
            code_region_hash = line.split('CodeRegionHash: ')[1].split(',')[0]
            code_region_type = line.split('CodeRegionType: ')[1].split(',')[0]

            column = line.split('Column: ')[1].split(',')[0].strip()
            file = line.split('File: ')[1].split(',')[0].strip()
            line_num = line.split('Line: ')[1].split(',')[0].strip()
            name = line.split('Name: ')[1].split(',')[0].strip()
            pass_name = line.split('Pass: ')[1].split(',')[0].strip()

            ai4c_func_metadata_elem = AI4CFunctionMetadataItem(code_region_hash=code_region_hash,
                                                               code_region_type=code_region_type,
                                                               column=column, file=file, line=line_num, name=name,
                                                               pass_name=pass_name)
            assert code_region_hash not in ai4c_func_metadata.keys
            ai4c_func_metadata.keys.append(code_region_hash)
            ai4c_func_metadata.elems.setdefault(code_region_hash, ai4c_func_metadata_elem)

        """
        multiprocessing version, 
        we think that directly turning all of the function parameters in doris in not a good idea,
        thus the multiprocessing parse is unnecessary.
        """
        # request = Queue()
        # result = Queue()
        #
        # for line in lines:
        #     request.put(line)
        #
        # ps = []
        # for _ in range(os.cpu_count()):
        #     # print(f'gen {_}-th process')
        #     p = Process(target=multiprocessing_wrapper, args=(request, result, len(lines), parse_ai4c_line, ), daemon=True)
        #     ps.append(p)
        #     p.start()
        #
        # while len(ai4c_func_metadata.keys) != len(lines):
        #     # print(f'done size: {len(ai4c_func_metadata.keys)}, total: {len(lines)}')
        #     results = result.get()
        #     code_region_hash = results[0]
        #     ai4c_func_metadata_elem = results[1]
        #     ai4c_func_metadata.keys.append(code_region_hash)
        #     ai4c_func_metadata.elems.setdefault(code_region_hash, ai4c_func_metadata_elem)
        #
        # for p in ps:
        #     p.terminate()

        ai4c_metadata = AI4CMetaData(ai4c_param_metadata, ai4c_func_metadata)
        return ai4c_metadata


class GCCParamMetadataParser:
    def __init__(self, file: str):
        self.file: str = file

    def parse(self) -> GCCParamMetadata:
        gcc_param_metadata = GCCParamMetadata()
        with open(self.file) as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('#') or line.strip().startswith('//'):
                    continue
                # Pattern: {"name": "max-inline-insns-auto", "default": 15, "min": 10, "max": 300}
                line_info = json.loads(line.strip())
                val_range = [line_info['min'], line_info['max']]
                assert "name" in line_info
                assert "default" in line_info
                assert 'min' in line_info
                assert 'max' in line_info

                gcc_param_metadata_item = GCCOptionParamMetadataItem(item_type=DataType.NUM,
                                                                    name=line_info['name'],
                                                                    val_range=val_range,
                                                                    width=1,
                                                                    default=line_info['default'])
                gcc_param_metadata.keys.append(line_info['name'])
                gcc_param_metadata.elems.setdefault(line_info['name'], gcc_param_metadata_item)
        return gcc_param_metadata


# test of parser
if __name__ == '__main__':
    # option_parser = OptionMetaDataParser(file='../config/option.txt')
    # option_metadata = option_parser.parse()
    # print(option_metadata)

    ai4c_parser = AI4CMetaDataParser(metadata_file='../config/ai4c.inline.txt', config_file='../config/doris/important.doris.function.yaml')
    ai4c_metadata = ai4c_parser.parse()
    print(ai4c_metadata)
