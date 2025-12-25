from enum import Enum


class SearcherType(Enum):
    GA: str = 'GA'
    RANDOM: str = 'RANDOM'
    PSO: str = 'PSO'
    SA: str = 'SA'
    DE: str = 'DE'


def str_to_searcher_type(s: str) -> SearcherType:
    try:
        return SearcherType[s]
    except KeyError:
        raise "Invalid value"


class EvaluatorType(Enum):
    DORIS: str = 'doris'
    REDIS: str = 'redis'
    SCANN: str = 'scann'
    ROCKSDB: str = 'rocksdb'
    RANDOM: str = 'random'
    FAST_DORIS: str = 'fast_doris'
    ZSTD: str = 'zstd'
    MYSQL: str = 'mysql'


def str_to_evaluator_type(s: str) -> EvaluatorType:
    try:
        return EvaluatorType[s]
    except KeyError:
        raise "Invalid value"


class TurnWhat(Enum):
    OPT_FUNC: str = 'opt_func'
    OPT: str = 'opt'
    FUNC: str = 'func'


def str_to_turn_what(s: str) -> TurnWhat:
    try:
        return TurnWhat[s]
    except KeyError:
        raise "Invalid value"


class TurnOpt(Enum):
    OPT_O3: str = 'OPT_O3'
    OPT: str = 'OPT'
    O3: str = 'O3'


def str_to_turn_opt(s: str) -> TurnOpt:
    try:
        return TurnOpt[s]
    except KeyError:
        raise "Invalid value"


turn_option: bool
turn_param: bool
turn_func: bool

evaluate_search: bool
evaluate_o3: bool

default_o3_perf: float

redis_start_core: str
redis_bench_core: str
