from typing import Union
from abc import ABC, abstractmethod

from autoturning.metadata import OptionMetadata, AI4CMetaData, DataType, GCCParamMetadata
from autoturning.utils import random_val, get_opt_negation


class OptionConfigParser(ABC):
    @abstractmethod
    def parse(self) -> dict[str, Union[int, str, bool, list]]:
        pass


class ExistingOptionConfigParser(OptionConfigParser):
    def __init__(self, metadata: OptionMetadata, config_file: str):
        super().__init__()
        self.metadata: OptionMetadata = metadata
        self.config_file: str = config_file
        self.config_lines: list[str] = []
        with open(self.config_file) as conf_f:
            self.config_lines = conf_f.readlines()
        self.conf_line_idx = 0

    def parse(self) -> dict[str, Union[int, str, bool, list]]:
        """
        Parse the given option configuration file.
        :return: the option configuration.
        """
        # Early return if we are already at the end of the file (using random initial value).
        if self.conf_line_idx >= len(self.config_lines):
            return RandomOptionConfigParser(self.metadata).parse()

        # Initialization
        config: dict[str, Union[int, str, bool, list]] = dict()
        for option in self.metadata.keys:
            if self.metadata.elems[option].type is DataType.BOOL:
                config[option] = self.metadata.elems[option].default
                # config[option] = True
                # config[option] = False
            else:
                # directly re-use the random logic to initialize the config.
                # TODO: this should be default value, not random value.
                rand_val = random_val(self.metadata.elems[option])
                config.setdefault(option, rand_val)

        # Ignore the empty lines.
        conf_line = self.config_lines[self.conf_line_idx]
        while conf_line.startswith('#') or not conf_line.strip() and self.conf_line_idx < len(self.config_lines):
            self.conf_line_idx += 1
            conf_line = self.config_lines[self.conf_line_idx]

        # Early return if we are already at the end of the file (using random initial value).
        if self.conf_line_idx >= len(self.config_lines):
            return RandomOptionConfigParser(self.metadata).parse()

        # Parse
        assert '=' not in conf_line, "Currently we only support 01 existing options."
        for option in self.metadata.keys:
            # The original value if false, thus the only thing we do is to assign a true value.
            if f" {option} " in conf_line or conf_line.strip().endswith(option):
                config[option] = True
            if f" {get_opt_negation(option)} " in conf_line or conf_line.strip().endswith(get_opt_negation(option)):
                config[option] = False

        self.conf_line_idx += 1
        return config


class RandomOptionConfigParser(OptionConfigParser):
    """
    This class is used to generate an option configuration randomly
    """

    def __init__(self, metadata: OptionMetadata):
        super().__init__()
        self.metadata: OptionMetadata = metadata

    def parse(self) -> dict[str, Union[int, str, bool, list]]:
        """
        This function generates a set of random value according to the global option metadata.
        :return: a dict contains a random optimization option configuration.
        """
        option_config = dict()
        for key in self.metadata.keys:
            rand_val = random_val(self.metadata.elems[key])
            option_config.setdefault(key, rand_val)
        return option_config


class RandomAI4CConfigParser:
    """
    This class aims to randomly generate ai4c config.
    """
    def __init__(self, metadata: AI4CMetaData):
        self.metadata: AI4CMetaData = metadata

    def parse(self) -> dict[str, dict[str, Union[int, str, bool]]]:
        """
        Randomly generate AI4C config.
        :return:
        """
        rand_val = dict()
        for code_hash in self.metadata.func_metadata.keys:
            for param_name in self.metadata.param_metadata.keys:
                rv = random_val(self.metadata.param_metadata.elems[param_name])
                rand_val.setdefault(code_hash, dict()).setdefault(param_name, rv)
        return rand_val


class RandomGCCParamConfigParser:
    """
    This class aims to randomly generate gcc param config.
    """
    def __init__(self, metadata: GCCParamMetadata):
        self.metadata: GCCParamMetadata = metadata
        pass

    def parse(self) -> dict[str, int]:
        """
        This function generates a set of random value according to the global gcc param metadata.
        :return: a dict contains a random gcc parameter option configuration.
        """
        gcc_param_config = dict()
        for key in self.metadata.keys:
            rand_val = random_val(self.metadata.elems[key])
            gcc_param_config.setdefault(key, rand_val)
        return gcc_param_config


if __name__ == '__main__':
    # test code for RandomOptionConfigParser
    from metadata_parser import OptionMetaDataParser

    option_parser = OptionMetaDataParser(file='../config/option.txt')
    option_metadata = option_parser.parse()
    random_option_config_parser = RandomOptionConfigParser(option_metadata)
    random_config = random_option_config_parser.parse()
    for name in random_config:
        print(f'{name}: {random_config[name]}')

    # test code for RandomAI4COptionConfigParser
    from metadata_parser import AI4CMetaDataParser

    ai4c_parser = AI4CMetaDataParser(metadata_file='../config/ai4c.txt', config_file='../config/function.yaml')
    ai4c_metadata = ai4c_parser.parse()
    random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    random_ai4c_config = random_ai4c_config_parser.parse()
    for code_hash in random_ai4c_config:
        print(f'{code_hash}: {random_ai4c_config[code_hash]}')
