from enum import Enum, auto
from typing import Union, Optional


class DataType(Enum):
    BOOL = auto()  # [True, False]
    NMNM = auto()  # [0, 1, [2, 65536]]
    NM = auto()  # -fpatchable-function-entry=n,m only, [0, [1, 65535]]
    ENUM = auto()  # [val1, val2, val3]
    NUM = auto()  # [lb, ub]


class ParamMetadataItem:
    """
    This class is a basic data class, which is used to record the metadata of parameters.
    The information of this class includes:
    1. type: the data type, which is defined in DataType
    2. name: the parameter name.
    3. val_range: the value range.
    4. width: the position required by this parameter when encoding it to a part of vector.
    """

    def __init__(self, item_type: DataType, name: str, val_range: list[Union[int, list[int], bool, str]], width: int):
        """
        Create a parameter metadata item.
        :param item_type: data type, defined in DataType
        :param name: parameter name
        :param val_range: value range
        :param width: the position required by this parameter when encoding it to a part of vector.
        """
        self.type: DataType = item_type
        self.name: str = name
        self.val_range: list[Union[int, list[int], bool, str]] = val_range
        self.width: int = width

    def __str__(self):
        return f'name={self.name}, type={self.type}, val_range={self.val_range}'


class OptionMetadata:
    """
    This class is used to represent the "metadata" of each optimization option, including:
    keys: which records all available optimization option name,
        and the order of these names represents the iteration order for encoding and decoding these settings.
    elems: which contains the specific metadata of corresponding option name, and it is a dict.
    base_level: which is the level that the turning process based.
    """

    def __init__(self, base_level='-O3'):
        """
        Construct an option metadata object
        :param base_level: the level that turning process based.
        """
        self.keys: list[str] = []
        self.elems: dict[str, GCCOptionParamMetadataItem] = dict()
        self.base_level = base_level

    def __str__(self):
        content = ''
        for k in self.keys:
            content += str(self.elems[k]) + '\n'
        return content

    def filter(self, constrains_file: str):
        """
        Remove the option information from the metadata, if both -f and -fno forms are specified in the constraints file.
        This function is used to provide a convenient way to directly remove the option via constraints file.
        BTW, I still think that the removing of an option should directly remove this option from the metadata file,
        rather than remove this option in this function.
        :param constrains_file: The path of constraint file.
        :return: None
        """
        conflict_constrains = []
        with open(constrains_file) as f:
            lines = f.readlines()
            constrains = [l.strip() for l in lines if l.startswith('-') and len(l) > 2]
            for c in constrains:
                if ' ' in c.strip():  # for single constrains only
                    continue
                if '-fno-' in c:
                    positive_form = c.replace('-fno-', '-f', 1)
                    if positive_form in constrains:
                        if c in self.keys:
                            conflict_constrains.append(c)
                        elif positive_form in self.keys:
                            conflict_constrains.append(positive_form)
        conflict_constrains = list(set(conflict_constrains))

        for co in conflict_constrains:
            print(f'[OptionMetadata.filter] Remove {co} from metadata.')
            assert co in self.keys
            self.keys.remove(co)
            assert co not in self.keys
            del self.elems[co]


class GCCOptionParamMetadataItem(ParamMetadataItem):
    """
    Concrete data class of optimization option metadata, each instance of this class record the metadata of one option.
    The metadata information includes:
    1. item_type: the configuration type, which is defined in DataType.
    2. name: the name of option
    3. val_range: which is a list of values, whose elements are usually int and str.
        When the elements of val_range is a list, this means there are some values have special meanings.
        For example, 1 is usually means open and use default value, 0 means close.
    4. width: which means the element with for this option when encode this optimization option.
    5. default: which records the default open/close status in base_level, only works when item_type is DataType.BOOL.
    """

    def __init__(self, item_type: DataType, name: str, val_range: list[Union[int, str, list[int]]],
                 width: int, default: Union[bool, int] = False):
        """
        Generates an option metadata item.
        :param item_type: the data type of item
        :param name: the name of option
        :param val_range: the value range
        :param width: the width that this option required during encoding process.
        :param default: the default status (open/close) under base_level, only works when item_type is DataType.BOOL.
        """
        super().__init__(item_type=item_type, name=name, val_range=val_range, width=width)
        self.default: Union[bool, int] = default


class AI4CParamMetadata:
    """
    This class is a data class, which records ai4c parameter metadata, includes:
    1. keys: which contains all parameter names that can be turned by ai4c.
        The order in keys controls the order of encoding.
    2. elems: which contains the concrete metadata information of parameter.
    """

    def __init__(self):
        self.keys: list[str] = []
        self.elems: dict[str, ParamMetadataItem] = dict()

    def __str__(self):
        content = ''
        for key in self.keys:
            content += str(self.elems[key]) + '\n'
        return content


class DebugCodeLocation:
    def __init__(self, file, line, column):
        self.file = file
        self.line = line
        self.column = column

    def __str__(self):
        content = f'Column: {self.column}, File: {self.file}, Line: {self.line}'
        return 'DebugLoc: {' + content + '}'


class AI4CFunctionMetadata:
    """
    A data class that records the function related ai4c metadata.
    The keys in this class are hash code of code region.
    The elems in this class are concrete information of the corresponding function.
    """

    def __init__(self):
        """
        Create an ai4c metadata configuration.
        """
        self.keys = []
        self.elems: dict[str, AI4CFunctionMetadataItem] = dict()

    def __str__(self):
        content = ''
        for key in self.keys:
            content += str(self.elems[key]) + '\n'
        return content


class AI4CFunctionMetadataItem:
    """
    This function is a data function, which is used to record the concrete metadata information of one function.
    The information includes:
    1. pass name: useless setting
    2. code region hash: the hash code of this function, which is used to identify the function under turning.
    3. code region type: useless setting.
    4. name: the function name
    5. file: the file that function belongs to
    6. line: the line number that function starts with.
    7. column: the column that function starts with.
    """

    def __init__(self, pass_name: str, code_region_hash: str, code_region_type: str, name: str, file: str, line: str,
                 column: str):
        """
        Create an ai4c function metadata object
        :param pass_name: useless setting
        :param code_region_hash: the hash code of this function, which is used to identify the function under turning.
        :param code_region_type: useless setting.
        :param name: the function name
        :param file: the file that function belongs to
        :param line: the line number that function starts with.
        :param column: the column that function starts with.
        """
        self.code_region_hash = code_region_hash
        self.code_region_type = code_region_type
        self.debug_loc = DebugCodeLocation(file, line, column)
        self.invocation = 0
        self.name = name
        self.pass_name = pass_name

    def __str__(self):
        code_region_hash_part = f'CodeRegionHash: {self.code_region_hash}'
        code_region_type_part = f'CodeRegionType: {self.code_region_type}'
        debug_log_part = str(self.debug_loc)
        invocation_part = f'Invocation: {self.invocation}'
        if '::' in self.name or "[" in self.name:
            self.name = f"'{self.name}'"
        name_part = f'Name: {self.name}'
        pass_part = f'Pass: {self.pass_name}'
        content = ', '.join([code_region_hash_part, code_region_type_part, debug_log_part,
                             invocation_part, name_part, pass_part])
        return '!AutoTuning {' + content + '}'


class AI4CParamMetadataItem(ParamMetadataItem):
    """
    This class is a data class, which refers to the ai4c parameter metadata.
    """

    def __init__(self, item_type: DataType, name: str, val_range: list, width: int):
        """
        Create a ai4c parameter metadata item.
        :param item_type: data type, defined in DataType
        :param name: parameter name
        :param val_range: value range
        :param width: the position required by this parameter when encoding it to a part of vector.
        """
        super().__init__(item_type=item_type, name=name, val_range=val_range, width=width)


class AI4CMetaData:
    """
    The top-level data class for recording the metadata information of ai4c, which contains:
    1. param_metadata: which record the parameter metadata that ai4c project can turn.
    2. function_metadata: which record all function basic information of project under auto-turning.
    """

    def __init__(self, param_metadata: AI4CParamMetadata, func_metadata: AI4CFunctionMetadata):
        self.param_metadata: AI4CParamMetadata = param_metadata
        self.func_metadata: AI4CFunctionMetadata = func_metadata

    def __str__(self):
        content = 'Param Part:\n'
        content += str(self.param_metadata)
        content += str(self.func_metadata)
        return content


class GCCParamMetadata:
    """
    This class represents the GCC params that using --param name=xxx to specify.
    """
    def __init__(self):
        """
        Create a metadata set of GCC parameters that can be turned by this project.
        """
        self.keys: list[str] = []
        self.elems: dict[str, GCCOptionParamMetadataItem] = dict()

    def __str__(self):
        content = ''
        for key in self.keys:
            content += str(self.elems[key]) + '\n'
        return content
