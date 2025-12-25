# import os
# import typing
# from typing import Optional, Union
# import random

# from z3 import *

# from autoturning.utils import get_opt_negation


# seed = 42
# set_param("smt.random_seed", seed)
# set_param("sat.random_seed", seed)


# class ConstrainsSolver:
#     """
#     This class is used solve the constraints of the options. Currently, it only targets on bool options.
#     """
#     def __init__(self, constrains_file: str):
#         self.option_constrains: list[list[str]] = []
#         if os.path.exists(constrains_file):
#             f = open(constrains_file)
#             constrains_file_lines = f.readlines()
#             f.close()
#             for line in constrains_file_lines:
#                 if line.startswith('-'):
#                     self.option_constrains.append(line.strip().split(' '))
    
#     def add_constrain(self, constrain: list[str]):
#         self.option_constrains.append(constrain)

#     def solve(self,
#               opt_config: dict[str, typing.Union[str, bool, int, list]],
#               ai4c_config: dict[str, dict[str, typing.Union[int, bool]]]):
#         """
#         Solve the constraints using Z3 solver.
#         :param opt_config: the option configuration.
#         :param ai4c_config: the ai4c configuration (currently not supported).
#         :return: None
#         """
#         # solve the option constrains

#         all_cond: list[BoolRef] = []
#         constrained_option_bool_refs: dict[str, BoolRef] = dict()
#         for opt_constrain in self.option_constrains:
#             single_cond: list[BoolRef] = []

#             # we should ensure that all options are in config.
#             options_in_config = [1 for option in opt_constrain
#                                  if option in opt_config or get_opt_negation(option) in opt_config]
#             if len(options_in_config) != len(opt_constrain):
#                 continue

#             # Convert constraints to Z3 expression.
#             for option in opt_constrain:
#                 if option not in constrained_option_bool_refs:  # z3 boolref not created
#                     if option not in opt_config:  # close in constrains
#                         assert get_opt_negation(option) in opt_config
#                         option = get_opt_negation(option)
#                         z3_bv = Bool(option)
#                         single_cond.append(Not(z3_bv))
#                     else:  # open in constrains
#                         z3_bv = Bool(option)
#                         single_cond.append(z3_bv)
#                     constrained_option_bool_refs.setdefault(option, z3_bv)
#                 else:
#                     if option not in opt_config:
#                         option = get_opt_negation(option)
#                         single_cond.append(Not(constrained_option_bool_refs[option]))
#                     else:
#                         single_cond.append(constrained_option_bool_refs[option])
#             all_cond.append(And(single_cond))

#         # record the constraints that have only one option.
#         single_constrains: list[str] = []
#         for opt_constrain in self.option_constrains:
#             if len(opt_constrain) == 1:
#                 single_constrains.extend(opt_constrain)
#         single_constrains = [get_opt_negation(_) if '-fno-' in _ else _ for _ in single_constrains]

#         # Add randomness to solver
#         max_iter = 1000
#         rand_iter = 1
#         while True:
#             random_cond: list[BoolRef] = []
#             iter_single_constrains = single_constrains[:]
#             for opt_constrain in self.option_constrains:

#                 # we should ensure that all options are in config.
#                 options_in_config = [1 for option in opt_constrain
#                                      if option in opt_config or get_opt_negation(option) in opt_config]
#                 if len(options_in_config) != len(opt_constrain):
#                     continue

#                 length_cond = len(set([get_opt_negation(_) if '-fno-' in _ else _
#                                        for _ in opt_constrain]) - set(iter_single_constrains)) > 1
#                 if length_cond:
#                     for option in opt_constrain:
#                         if option in iter_single_constrains or get_opt_negation(option) in iter_single_constrains:
#                             continue
#                         if random.randint(0, 1) == 1:
#                             if option not in opt_config:
#                                 option = get_opt_negation(option)
#                             if random.randint(0, 1) == 1:
#                                 random_cond.append(constrained_option_bool_refs[option])
#                             else:
#                                 random_cond.append(Not(constrained_option_bool_refs[option]))
#                             iter_single_constrains.append(option)  # add already fixed option into single constrains
#             random_cond = And(random_cond)
#             solver = Solver()
#             solver.add(And(Not(Or(all_cond)), random_cond))
#             if solver.check() == sat:
#                 break

#             if max_iter <= rand_iter:
#                 assert False
#             rand_iter += 1

#         model = solver.model()
#         for option in constrained_option_bool_refs:
#             opt_config[option] = model[constrained_option_bool_refs[option]]
#             print(f'{option}={model[constrained_option_bool_refs[option]]}')


# if __name__ == '__main__':
#     from autoturning.metadata_parser import *
#     from autoturning.config_parser import *

#     option_parser = OptionMetaDataParser(file='../config/option.all.01.txt')
#     option_metadata = option_parser.parse()
#     random_option_config_parser = RandomOptionConfigParser(option_metadata)
#     random_option_config = random_option_config_parser.parse()

#     from metadata_parser import AI4CMetaDataParser
#     from config_parser import RandomAI4CConfigParser

#     ai4c_parser = AI4CMetaDataParser(metadata_file='../config/ai4c.txt', config_file='../config/redis/redis.function.yaml')
#     ai4c_metadata = ai4c_parser.parse()
#     random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
#     random_ai4c_config = random_ai4c_config_parser.parse()

#     print('*'*50)
#     seed = random.randint(1, 1000)
#     set_param("smt.random_seed", seed)
#     set_param("sat.random_seed", seed)
#     c = ConstrainsSolver(constrains_file="../config/constrains.txt")
#     c.solve(ai4c_config=random_ai4c_config, opt_config=random_option_config)


import os
import typing
from z3 import *


def get_opt_negation(option: str) -> str:
    option = option.strip()
    if not option:
        return ""
    
    if '=' in option:
        return ''

    # Define common negation rules
    # Note: Longer prefixes (like "-fno-") must be checked before shorter ones (like "-f")
    negation_rules = [
        ("-fno-", "-f"),
        ("-f", "-fno-"),
        ("-mno-", "-m"),
        ("-m", "-mno-"),
        ("-Wno-", "-W"),
        ("-W", "-Wno-"),
    ]

    for prefix, negated_prefix in negation_rules:
        if option.startswith(prefix):
            return negated_prefix + option[len(prefix):]
    return ""


class ConstrainsSolver:
    def __init__(self, constrains_file: str):
        set_param("smt.random_seed", 42)
        set_param("sat.random_seed", 42)
        self.option_constrains: list[list[str]] = []
        if os.path.exists(constrains_file):
            f = open(constrains_file)
            constrains_file_lines = f.readlines()
            f.close()
            for line in constrains_file_lines:
                line = line.strip()
                if line.startswith("-"):
                    self.option_constrains.append(line.split(" "))

    def add_constrain(self, constrain: list[str]):
        self.option_constrains.append(constrain)
    def solve(self, opt_config: dict[str, typing.Union[str, bool, int, list]], ai4c_config=None):
        # Convert to second format (key=value:true)
        second_format = {}
        for key, value in opt_config.items():
            if isinstance(value, bool):
                second_format[key] = value
            else:
                second_format[f"{key}={value}"] = True
        
        # Process constraints on second format
        all_single_constrain = set()
        for constrain_list in self.option_constrains:
            if len(constrain_list) > 1:
                continue
            constrain = constrain_list[0]
            all_single_constrain.add(constrain)
            if get_opt_negation(constrain) in all_single_constrain:
                second_format.pop(constrain, None)
                second_format.pop(get_opt_negation(constrain), None)

        all_cond: list[BoolRef] = []
        constrained_option_bool_refs: dict[str, BoolRef] = dict()

        for opt_constrain in self.option_constrains:
            single_cond: list[BoolRef] = []
            options_in_config = [
                1 for option in opt_constrain
                if option in second_format or get_opt_negation(option) in second_format
            ]
            if len(options_in_config) != len(opt_constrain):
                continue

            for option in opt_constrain:
                if option not in constrained_option_bool_refs:
                    if option not in second_format:
                        assert get_opt_negation(option) in second_format
                        option = get_opt_negation(option)
                        z3_bv = Bool(option)
                        single_cond.append(Not(z3_bv))
                    else:
                        z3_bv = Bool(option)
                        single_cond.append(z3_bv)
                    constrained_option_bool_refs.setdefault(option, z3_bv)
                else:
                    if option not in second_format:
                        option = get_opt_negation(option)
                        single_cond.append(Not(constrained_option_bool_refs[option]))
                    else:
                        single_cond.append(constrained_option_bool_refs[option])
            all_cond.append(And(single_cond))
        print(constrained_option_bool_refs)
        solver = Solver()
        solver.add(Not(Or(all_cond)))
        if solver.check() == sat:
            model = solver.model()
            # Update second format with results
            # second_format.clear()
            for option in constrained_option_bool_refs:
                second_format[option] = bool(model[constrained_option_bool_refs[option]])
            
            # Convert back to first format and update original opt_config
            opt_config.clear()
            for key, value in second_format.items():
                if '=' in key:
                    original_key, original_value = key.split('=', 1)
                    if value is False:
                        opt_config[original_key] = 'OFF'
                    else:
                        opt_config[original_key] = original_value
                else:
                    opt_config[key] = value
            print(f"[ConstrainsSolver.solve] solved opt_config: {opt_config}")
            return opt_config
        else:
            unsat_constraints = []
            for cond in all_cond:
                solver_unsat = Solver()
                solver_unsat.add(cond)
                if solver_unsat.check() == unsat:
                    unsat_constraints.append(cond)

            print("No valid configuration found under given constraints.")
            print("Unsatisfiable constraints:")
            for unsat in unsat_constraints:
                print(unsat)
            raise RuntimeError("No valid configuration found under given constraints.")
    # def solve(self, opt_config: dict[str, str]):
        
    #     all_single_constrain = set()
    #     for constrain_list in self.option_constrains:
    #         if len(constrain_list) > 1:
    #             continue
    #         constrain = constrain_list[0]
    #         all_single_constrain.add(constrain)
    #         if get_opt_negation(constrain) in all_single_constrain:
    #             opt_config.pop(constrain, None)
    #             opt_config.pop(get_opt_negation(constrain), None)

    #     all_cond: list[BoolRef] = []
    #     constrained_option_bool_refs: dict[str, BoolRef] = dict()

    #     for opt_constrain in self.option_constrains:
    #         single_cond: list[BoolRef] = []
    #         options_in_config = [
    #             1 for option in opt_constrain
    #             if option in opt_config or get_opt_negation(option) in opt_config
    #         ]
    #         if len(options_in_config) != len(opt_constrain):
    #             continue

    #         for option in opt_constrain:
    #             if option not in constrained_option_bool_refs:
    #                 if option not in opt_config:
    #                     assert get_opt_negation(option) in opt_config
    #                     option = get_opt_negation(option)
    #                     z3_bv = Bool(option)
    #                     single_cond.append(Not(z3_bv))
    #                 else:
    #                     z3_bv = Bool(option)
    #                     single_cond.append(z3_bv)
    #                 constrained_option_bool_refs.setdefault(option, z3_bv)
    #             else:
    #                 if option not in opt_config:
    #                     option = get_opt_negation(option)
    #                     single_cond.append(Not(constrained_option_bool_refs[option]))
    #                 else:
    #                     single_cond.append(constrained_option_bool_refs[option])
    #         all_cond.append(And(single_cond))

    #     solver = Solver()
    #     solver.add(Not(Or(all_cond)))

    #     if solver.check() == sat:
    #         # print("Found a valid configuration under given constraints.")
    #         model = solver.model()
    #         for option in constrained_option_bool_refs:
    #             opt_config[option] = bool(model[constrained_option_bool_refs[option]])
    #         return opt_config
    #     else:
    #         # 打印无法满足的约束
    #         unsat_constraints = []
    #         for cond in all_cond:
    #             solver_unsat = Solver()
    #             solver_unsat.add(cond)
    #             if solver_unsat.check() == unsat:
    #                 unsat_constraints.append(cond)

    #         print("No valid configuration found under given constraints.")
    #         print("Unsatisfiable constraints:")
    #         for unsat in unsat_constraints:
    #             print(unsat)
    #         raise RuntimeError("No valid configuration found under given constraints.")
        

if __name__ == "__main__":
    opt_config = {
        '-funroll-all-loops': False,
        '-funroll-completely-grow-size': False,
        '-funroll-loop': False,
        '-fa=8': True,
        '-fb=1024': True,
    }
    
    solver = ConstrainsSolver("reduce_constrains/constrains/fast.txt")

    # opt_config = {
    #     "-ftree-parallelize-loops": "16",   # ❌ 不合法
    #     "-fschedule-insns": "after",        # ✅ 合法
    # }

    print("Before:", opt_config)
    fixed_config = solver.solve(opt_config)
    print("After:", fixed_config)