import random

from autoturning.metadata import ParamMetadataItem, DataType


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    try:
        pipe = os.popen(cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"


def random_single_numeric_val(val_range: list) -> int:
    """
    Generate random numeric value according to the given value range.
    The value range may be constructed with two layer, the first layer may contain some special value.
    Such as 0, 1, which are the value that close the optimization or use the default value of this configuration.
    :param val_range: the value range
    :return: a random numberic value
    """
    rand_idx = random.randint(0, len(val_range) - 1)
    rand_val = val_range[rand_idx]  # the special value.
    if isinstance(rand_val, list):
        assert len(rand_val) == 2
        lb = rand_val[0]
        ub = rand_val[1]
        rand_val = random.randint(lb, ub)
    return rand_val


def random_val(metadata: ParamMetadataItem) -> any:
    """
    Generate a random config value according to the provided metadata.
    :param metadata: the metadata information of the configuration item.
    :return: any
    """
    if metadata.type == DataType.NUM:
        assert len(metadata.val_range) == 2
        lb = metadata.val_range[0]
        ub = metadata.val_range[1]
        return random.randint(lb, ub)
    elif metadata.type == DataType.BOOL:
        return random.randint(0, 1) == 0
    elif metadata.type == DataType.NM:
        # for only -fpatchable-function-entry=n,m
        val1 = random_single_numeric_val(metadata.val_range)
        val2 = random_single_numeric_val(metadata.val_range)
        return [val1, val2]
    elif metadata.type == DataType.ENUM:
        return random_single_numeric_val(metadata.val_range)
    elif metadata.type == DataType.NMNM:
        n1 = random_single_numeric_val(metadata.val_range)
        m1 = random_single_numeric_val(metadata.val_range)
        n2 = random_single_numeric_val(metadata.val_range)
        m2 = random_single_numeric_val(metadata.val_range)
        return [n1, m1, n2, m2]


def get_opt_negation(flag) -> str:
    """
    Get negative version of the given optimization flag
    :param flag: the given optimization flag
    :return: The negative version of the given flag.
    """
    if '-fno-' not in flag:
        if '-fweb-' not in flag:
            return flag[:2] + 'no-' + flag[2:]
    else:
        return flag[:2] + flag[5:]


def random_code_bit(metadata: ParamMetadataItem) -> int:
    """
    Generate random code bit, which is used for code in search, according to the given param metadata.
    :param metadata: the given metadata
    :return: random code bit.
    """
    if metadata.type in [DataType.BOOL, DataType.ENUM]:
        return 100 * random.randint(1, len(metadata.val_range))
    elif metadata.type in [DataType.NMNM, DataType.NM]:
        return random_single_numeric_val(metadata.val_range)
    elif metadata.type == DataType.NUM:
        lb = metadata.val_range[0]
        ub = metadata.val_range[1]
        return random.randint(lb, ub)


def mutate_to_new_code(val: int, metadata: ParamMetadataItem) -> int:
    """
    Randomly generate a new value according to the metadat,
    this function ensures that the new value is different from the original one.
    :param val: the original value, which is never the original one.
    :param metadata: metadata
    :return: the new value.
    """

    # def inner_value_generator():
    #     if metadata.type in [DataType.BOOL, DataType.ENUM]:
    #         return 100 * random.randint(1, len(metadata.val_range))
    #     elif metadata.type in [DataType.NMNM, DataType.NM]:
    #         return random_single_numeric_val(metadata.val_range)
    #     elif metadata.type == DataType.NUM:
    #         lb = metadata.val_range[0]
    #         ub = metadata.val_range[1]
    #         return random.randint(lb, ub)

    new_val = random_code_bit(metadata)
    while new_val == val:
        new_val = random_code_bit(metadata)
    return new_val


def formalize_code_bit(val: int, param_metadata: ParamMetadataItem) -> int:
    """
    Formalize the given value according to the param metadata, note that this function returns the formalized "code",
    not a config value.
    :param val: the given code value.
    :param param_metadata: the given param metadata
    :return: the value.
    """
    if param_metadata.type in [DataType.BOOL, DataType.ENUM]:
        return int(max(min(val, len(param_metadata.val_range) * 100), 0))
    elif param_metadata.type in [DataType.NMNM, DataType.NM]:
        return int(max(min(val, param_metadata.val_range[-1][-1]), param_metadata.val_range[0]))
    elif param_metadata.type is DataType.NUM:
        return int(max(min(val, param_metadata.val_range[-1]), param_metadata.val_range[0]))
    else:
        assert False, "Unreachable code!"
    pass


def calculate_acc_rate(o3_perf: float, opt_perf: float, project: str) -> float:
    if o3_perf <= 0 or opt_perf <= 0 or o3_perf == float("inf") or opt_perf == float("inf"):
        return -1

    assert project in ['redis', 'doris', 'scann']
    if project in ['redis', 'doris']:
        return o3_perf / opt_perf
    elif project == 'scann':
        return opt_perf / o3_perf
