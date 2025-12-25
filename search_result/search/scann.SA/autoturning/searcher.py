# Created by Chenyao Suo
# This file is a simple framework to conduct local search from several given optimization settings,
# using different algorithm, i.e., searcher class in this file.
# Some algorithms are already implemented in this file, including:
# Genetic Algorithm
# Random Search
# Particle Swarm Optimization
# Simulated Annealing
# Differential Evolution

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

import autoturning.script_args
from autoturning.codec import decode, encode
from autoturning.config_writer import OptionConfigWriter, AI4CConfigWriter, GCCParamConfigWriter
from autoturning.evaluator import *
from autoturning.metadata import OptionMetadata, AI4CMetaData, ParamMetadataItem, GCCParamMetadata
from autoturning.recorder import Recorder
from autoturning.utils import mutate_to_new_code, random_code_bit, formalize_code_bit
from autoturning.constrains_solver import ConstrainsSolver


class SearchConfig:

    def __init__(self,
                 ai4c_plagin_lib: str,
                 ai4c_option_file: str,
                 ai4c_config_dir: str,
                 perf_record_file: str,
                 opt_record_file: str):
        """
        A simple data class, stores some ai4c configuration, and files to record the result.
        :param ai4c_plagin_lib:
        :param ai4c_option_file:
        :param ai4c_config_dir:
        :param perf_record_file:
        :param opt_record_file:
        """
        self.ai4c_plagin_lib: str = ai4c_plagin_lib
        self.ai4c_option_file: str = ai4c_option_file
        self.ai4c_config_dir: str = ai4c_config_dir
        self.perf_record_file: str = perf_record_file
        self.opt_record_file: str = opt_record_file


class Searcher(ABC):
    def __init__(self,
                 opt_settings: list,
                 population: int,
                 evaluation_cnt_limit: int,
                 time_limit: int,
                 evaluator: Evaluator,
                 option_metadata: OptionMetadata,
                 ai4c_metadata: AI4CMetaData,
                 gcc_param_metadata: GCCParamMetadata,
                 search_config: SearchConfig,
                 constrains_file: str):
        """
        Basic class for all algorithm searchers.
        Every searcher will firstly evaluate the given optimization settings,
        and record valid optimization settings (i.e., the return value of self.evaluation() is bigger than 0).
        self.opt_settings will always hold the valid optimization settings, and its length is always population.
        :param opt_settings: the given optimization settings.
        :param population: the number of start point.
        :param evaluation_cnt_limit: the number of evaluation. the search process will be stopped when evaluation cnt is zero
        """
        self.opt_settings: list[list[int]]
        self.population: int
        self.best_setting: Optional[list[int]]
        self.best_performance: float
        self.evaluation_cnt_limit: int
        self.time_limit: int
        self.evaluator: Evaluator
        self.option_metadata: OptionMetadata
        self.ai4c_metadata: AI4CMetaData
        self.gcc_param_metadata: GCCParamMetadata
        self.param_metadatas: list[ParamMetadataItem]
        self.option_writer: OptionConfigWriter
        self.ai4c_writer: AI4CConfigWriter
        self.perf_recorder: Recorder
        self.opt_recorder: Recorder
        self.search_config: SearchConfig

        self.opt_settings = opt_settings
        self.population = population
        self.best_setting = None
        self.best_performance = 0.0
        self.evaluation_cnt_limit = evaluation_cnt_limit
        self.time_limit = time_limit
        self.evaluator = evaluator
        self.option_metadata = option_metadata
        self.ai4c_metadata = ai4c_metadata
        self.gcc_param_metadata = gcc_param_metadata
        self.param_metadatas = []
        self.option_writer = OptionConfigWriter(option_metadata)
        self.ai4c_writer = AI4CConfigWriter(
            ai4c_metadata, search_config.ai4c_config_dir
        )
        self.gcc_param_writer = GCCParamConfigWriter(gcc_param_metadata)
        self.perf_recorder = Recorder(search_config.perf_record_file)
        self.opt_recorder = Recorder(search_config.opt_record_file)
        self.search_config = search_config

        self.constrains_solver: Optional[ConstrainsSolver] = None
        if constrains_file is not None:
            self.constrains_solver: ConstrainsSolver = ConstrainsSolver(constrains_file)

        if option_metadata is not None:
            for param_name in option_metadata.keys:
                for _ in range(option_metadata.elems[param_name].width):
                    self.param_metadatas.append(option_metadata.elems[param_name])

        if ai4c_metadata is not None:
            for func_hash in ai4c_metadata.func_metadata.keys:
                for param_name in ai4c_metadata.param_metadata.keys:
                    for _ in range(
                        ai4c_metadata.param_metadata.elems[param_name].width
                    ):
                        self.param_metadatas.append(
                            ai4c_metadata.param_metadata.elems[param_name]
                        )

        if gcc_param_metadata is not None:
            for param_name in gcc_param_metadata.keys:
                for _ in range(gcc_param_metadata.elems[param_name].width):
                    self.param_metadatas.append(gcc_param_metadata.elems[param_name])
        assert len(self.param_metadatas) == len(opt_settings[0])

    @abstractmethod
    def search(self) -> None:
        """
        Conduct the search process.
        :return:
        """
        pass

    def init_evaluate(self) -> list[float]:
        """
        Evaluate all given optimization settings to eliminate invalid optimization settings,
        and init the best performance and best optimization settings.
        :return: None
        """
        opt_performance: list[list[Union[float, list[int]]]] = []
        for opt_setting in self.opt_settings:
            performance: float = self.evaluate(opt_setting)
            # When there are a lot of failed configuration, this line will cause the following assertion crash.
            # Do not remove this line, because this condition is useful.
            # In other words, crash is the expect behavior in this case.
            # if self.evaluation_cnt_limit <= 0:
            #     assert False, "The evaluation count is 0"
            if performance > 0:
                opt_performance.append([performance, opt_setting])
            if len(opt_performance) == self.population:
                break
        assert (
            len(opt_performance) == self.population
        ), "Please give more optimization settings, current optimization settings are not enough"
        opt_performance = sorted(opt_performance, reverse=True)
        self.opt_settings = [_[1] for _ in opt_performance]
        performances = [_[0] for _ in opt_performance]
        self.best_performance = opt_performance[0][0]
        self.best_setting = opt_performance[0][1]
        return performances

    def evaluate(self, opt: list) -> float:
        """
        Evaluate the given optimization setting
        :param opt: the given optimization setting, i.e., a list of 101010(int).
        :return: the accelerate rate, the bigger this rate is, the better the given opt is.
        Negative(or 0) accelerate rate means compile failed or test failed.(invalid optimization setting)
        """
        option_config: dict[str, Union[str, bool, int, list]]
        ai4c_config: dict[str, dict[str, Union[int, bool]]]
        option_config, ai4c_config, gcc_param_config = decode(ai4c_metadata=self.ai4c_metadata,
                                                              option_metadata=self.option_metadata,
                                                              gcc_param_metadata=self.gcc_param_metadata,
                                                              code=opt)
        if self.constrains_solver is not None:
            self.constrains_solver.solve(
                opt_config=option_config, ai4c_config=ai4c_config
            )
            new_c: list[int] = encode(
                ai4c_config=ai4c_config,
                ai4c_metadata=self.ai4c_metadata,
                option_config=option_config,
                option_metadata=self.option_metadata,
                gcc_param_config=gcc_param_config,
                gcc_param_metadata=self.gcc_param_metadata,
            )
            opt[:] = new_c
        yaml_file: str = self.ai4c_writer.write(ai4c_config)
        option_config_filted: dict[str, Union[str, bool, int, list]] = {
            k: v
            for k, v in option_config.items()
            if not (isinstance(v, str) and v == "OFF")
        }

        option_cmd: str = self.option_writer.write(option_config_filted)
        gcc_param: str = self.gcc_param_writer.write(gcc_param_config)

        ai4c_part: str
        ai4c_part = ""
        if yaml_file != "":
            ai4c_part = (
                f"-fplugin={self.search_config.ai4c_plagin_lib} "
                f"-fplugin-arg-coarse_option_tuning_plugin_gcc12-yaml={self.search_config.ai4c_option_file} "
                f"-fplugin-arg-coarse_option_tuning_plugin_gcc12-autotune={yaml_file}"
            )

        # opt_cmd = '-g ' + self.option_metadata.base_level
        if self.option_metadata is not None:
            opt_cmd = f"-g {self.option_metadata.base_level}"
        else:
            opt_cmd = "-g -O3"
        if autoturning.script_args.turn_option:
            opt_cmd += " " + option_cmd
        if autoturning.script_args.turn_param:
            opt_cmd += " " + gcc_param
        if autoturning.script_args.turn_func:
            opt_cmd += " " + ai4c_part

        # record the evaluation.
        self.opt_recorder.record(f"{self.opt_recorder.iter}: {opt_cmd}\n")

        o3_perf: float
        opt_perf: float
        acc_rate: float
        o3_perf, opt_perf, acc_rate = self.evaluator.evaluate(opt_cmd)
        self.evaluation_cnt_limit -= 1

        # record the evaluation.
        self.perf_recorder.record(
            f"{self.perf_recorder.iter}: O3_perf={o3_perf}, opt_perf={opt_perf}, acc_rate={acc_rate}\n"
        )

        return acc_rate


class MutateBasedSearcher(Searcher, ABC):
    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        mutation_bit_rate: float,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        ai4c_metadata: AI4CMetaData,
        gcc_param_metadata: GCCParamMetadata,
        search_config: SearchConfig,
        constrains_file: str,
        gcc_param_bit_rate: float = 1,
    ):
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            constrains_file=constrains_file,
            time_limit=time_limit,
        )
        self.option_size = len(option_metadata) if option_metadata is not None else 0
        self.gcc_param_size = len(gcc_param_metadata) if gcc_param_metadata is not None else 0
        self.option_mutation_bits = int(mutation_bit_rate * self.option_size)
        self.gcc_param_mutation_bits = int(gcc_param_bit_rate * self.gcc_param_size)
        self.mutation_bits = int(mutation_bit_rate * len(opt_settings[0]))

    def mutation(self, opt: list[int]) -> list[int]:
        mutate_idx: list[int]
        new_values: list[int]
        option_mutate_idx = random.sample(
            range(len(opt[0 : self.option_size])), self.option_mutation_bits
        )
        gcc_param_mutate_idx = random.sample(
            range(self.option_size, self.option_size + len(opt[self.option_size :])),
            self.gcc_param_mutation_bits,
        )
        mutate_idx = option_mutate_idx + gcc_param_mutate_idx
        new_values = [
            (
                mutate_to_new_code(opt[_], self.param_metadatas[_])
                if _ in mutate_idx
                else opt[_]
            )
            for _ in range(len(opt))
        ]
        return new_values


class RandomSearcher(MutateBasedSearcher):

    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        mutation_bit_rate: float,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        ai4c_metadata: AI4CMetaData,
        gcc_param_metadata: GCCParamMetadata,
        search_config: SearchConfig,
        constrains_file: str,
        gcc_param_bit_rate: float = 1,
    ):
        """
        Random Searcher that starts from given optimization settings.
        It searches better optimization settings via mutation all given start optimization settings in a polling manner.
        This searcher has 2 parameters:
        (1) the number of mutation count(for whole search process)
        (2) the number of mutation bits(in each mutation)
        :param opt_settings: the given optimization settings.
        :param population: the number of start point.
        :param evaluation_cnt_limit: the number of evaluation. the search process will be stopped when evaluation cnt is zero
        :param mutation_bit_rate: the number of bit to be flipped in each mutation.
        """
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            mutation_bit_rate=mutation_bit_rate,
            constrains_file=constrains_file,
            time_limit=time_limit,
            gcc_param_bit_rate=gcc_param_bit_rate,
        )

    def search(self):
        cur_search_cnt: int
        cur_opt_idx: int

        self.init_evaluate()
        cur_search_cnt = 0
        cur_opt_idx = 0
        while self.evaluation_cnt_limit != 0 and self.time_limit > 0:
            start_time = time.time()
            print(
                f"[RandomSearcher.search] remain evaluation count: {self.evaluation_cnt_limit}"
            )
            mutated_opt: list[int] = self.mutation(self.opt_settings[cur_opt_idx])
            performance: float = self.evaluate(mutated_opt)
            if performance > 0:
                self.opt_settings[cur_opt_idx] = mutated_opt
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_setting = mutated_opt
            cur_opt_idx = (cur_opt_idx + 1) % len(self.opt_settings)
            cur_search_cnt += 1
            end_time = time.time()
            self.time_limit -= end_time - start_time
        # return self.best_setting, self.best_performance


class GASearcher(MutateBasedSearcher):

    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        crossover_bit: int,
        mutation_bit_rate: int,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        ai4c_metadata: AI4CMetaData,
        gcc_param_metadata: GCCParamMetadata,
        search_config: SearchConfig,
        constrains_file: str,
        gcc_param_bit_rate: float = 1,
    ):
        """
        Searcher that use Genetic Algorithm.
        :param opt_settings: the given optimization settings
        :param population: the number of starting point, as well as the population size of Genetic Algorithm.
        :param evaluation_cnt_limit: the number of evaluation count, the search process will be stopped when evaluation cnt is reached.
        :param crossover_bit: the number of bit to be crossover each time.
        """
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            mutation_bit_rate=mutation_bit_rate,
            constrains_file=constrains_file,
            time_limit=time_limit,
            gcc_param_bit_rate=gcc_param_bit_rate,
        )
        self.crossover_bit: int = int(crossover_bit * len(opt_settings[0]))

    def selection(
        self, opt_settings, performances
    ) -> tuple[list[list[int]], list[float]]:
        """
        We directly select the maximum individuals.
        :param opt_settings: the given population(optimization settings)
        :param performances: the performances of the given optimization settings
        :return: the selected population and its performance
        """
        if len(opt_settings) == self.population:
            return opt_settings, performances
        opt_performance = sorted(
            [[performances[_], opt_settings[_]] for _ in range(len(performances))],
            reverse=True,
        )[: self.population]
        return [_[1] for _ in opt_performance], [_[0] for _ in opt_performance]

    def crossover(self, opt_setting1, opt_setting2) -> list[list[int]]:
        """
        Conduct crossover between given optimization settings.
        :param opt_setting1:
        :param opt_setting2:
        :return:
        """
        assert len(opt_setting1) == len(opt_setting2)
        crossover_idx = random.sample(range(len(opt_setting1)), self.crossover_bit)
        for idx in crossover_idx:
            opt_setting1[idx], opt_setting2[idx] = opt_setting2[idx], opt_setting1[idx]
        return [opt_setting1, opt_setting2]

    def search(self):
        performances: list[float]
        population: list[list[int]]

        performances = self.init_evaluate()
        population = self.opt_settings[:]
        while True:
            print(
                f"[GASearcher.search] remain evaluation count: {self.evaluation_cnt_limit}"
            )
            start_time = time.time()

            if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                break

            # selection
            population, performances = self.selection(population, performances)

            # crossover
            new_population = []
            for idx in range(len(population) - 1):
                new_population.extend(
                    self.crossover(population[idx], population[idx + 1])
                )

            # mutation
            new_population = [self.mutation(_) for _ in new_population]

            # calculate performances
            new_performances = []
            for new_opt in new_population:
                if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                    break
                performance = self.evaluate(new_opt)
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_setting = new_opt[:]
                new_performances.append(performance)

            # Different from general GA,
            # to make sure the searcher holds at least population valid optimization settings,
            # we add the old ones into the new population.
            new_population.extend(population)
            new_performances.extend(performances)

            population = new_population
            performances = new_performances
            end_time = time.time()
            self.time_limit -= end_time - start_time

        # return self.best_setting, self.best_performance


class PSOSearcher(Searcher):

    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        ai4c_metadata: AI4CMetaData,
        gcc_param_metadata: GCCParamMetadata,
        search_config: SearchConfig,
        constrains_file: str,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        v_rate: float = 0.1,
    ):
        """
        This searcher conducts PSO algorithm.
        Different from general PSO algorithm, we make the following trade-off to fit the 0-1 option manner.
        (1) Each particle holds a set of float number (instead of int) as position.
        (2) The velocity will be normalized to -0.5~0.5, that is enough for the float number convert from 0 to 1.
        Otherwise, the velocity will be too big (e.g., much more than 1).
        (3) The float number will be transformed to 0-1 options before evaluation, but all position,
        including local(global) optima position, will be recorded as float numbers.
        (4) Reverse if new position is not valid.
        :param opt_settings: the given optimization settings
        :param population: the number of particles
        :param evaluation_cnt_limit: the number fo evaluation count limit.
        :param w: weight of keep original velocity unchanged
        :param c1: weight of influence of local optima position to velocity
        :param c2: weight of influence of global optima position to velocity
        """
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            constrains_file=constrains_file,
            time_limit=time_limit,
        )
        self.w: float
        self.c1: float
        self.c2: float
        self.v_rate: float

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_rate = v_rate  # used to reduce the velocity

    def search(self):
        # initialize particle position and velocity
        # evaluate all particles, and update the local optima position (for each particle)

        def init_velocity() -> list[float]:
            ve: list[float] = [float(random_code_bit(_)) for _ in self.param_metadatas]
            return ve

        performances = self.init_evaluate()
        particles = self.opt_settings[:]
        velocities = [init_velocity() for _ in particles]
        local_best_pos = particles[:]
        local_best_performance = performances[:]

        particles = np.array(particles)
        velocities = np.array(velocities)
        local_best_pos = np.array(local_best_pos)
        local_best_performance = np.array(local_best_performance)

        # find the global optima position and performance
        global_best_pos = self.best_setting[:]

        while True:
            start_time = time.time()

            print(
                f"[PSOSearcher.search] remain evaluation count: {self.evaluation_cnt_limit}"
            )
            if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                break
            # update velocity
            r1 = float(random.randint(0, 100)) / 100.0
            r2 = float(random.randint(0, 100)) / 100.0
            velocities = np.array(
                [
                    self.w * velocities[_]
                    + self.c1 * r1 * (local_best_pos[_] - particles[_])
                    + self.c2 * r2 * (global_best_pos - particles[_])
                    for _ in range(len(particles))
                ]
            )
            # min_velocities = np.min(velocities, axis=0)
            # max_velocities = np.max(velocities, axis=0)
            # size_velocities = max_velocities - min_velocities
            # velocities = np.array([(velocity - min_velocities) / size_velocities for velocity in velocities])

            # update position
            new_particles = [
                particles[_] + velocities[_] for _ in range(len(particles))
            ]
            # new_particles = [[__ if 0.0 <= __ <= 1.0 else (0.0 if __ < 0.0 else 1.0) for __ in _] for _ in new_particles]

            # formalize
            velocities = np.array(
                [
                    [
                        float(formalize_code_bit(_[idx], self.param_metadatas[idx]))
                        * self.v_rate
                        for idx in range(len(_))
                    ]
                    for _ in velocities
                ]
            )
            new_particles = [
                [
                    float(formalize_code_bit(_[idx], self.param_metadatas[idx]))
                    for idx in range(len(_))
                ]
                for _ in new_particles
            ]

            # evaluate and reverse if invalid particles are generated
            # integer_particles = [[int(__ + 0.5) for __ in _] for _ in new_particles]

            for particle_idx in range(len(new_particles)):
                if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                    break
                performance = self.evaluate(new_particles[particle_idx])

                # revert movement if the optimization setting is invalid
                # here, we keep the velocities, because it still points to a better position.
                if performance <= 0:
                    new_particles[particle_idx] = particles[particle_idx]
                    continue

                # update local optima
                if performance > local_best_performance[particle_idx]:
                    local_best_performance[particle_idx] = performance
                    local_best_pos[particle_idx] = np.array(new_particles[particle_idx])

                # update global optima
                if performance > self.best_performance:
                    self.best_performance = performance
                    global_best_pos = np.array(new_particles[particle_idx])

            particles = np.array(new_particles)
            end_time = time.time()
            self.time_limit -= end_time - start_time
        # return self.best_setting, self.best_performance


class SASearcher(MutateBasedSearcher):

    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        gcc_param_metadata: GCCParamMetadata,
        ai4c_metadata: AI4CMetaData,
        search_config: SearchConfig,
        constrains_file: str,
        temperature: float = 100.0,
        cooling_rate: float = 0.95,
        mutation_bit_rate: float = 0.1,
        gcc_param_bit_rate: float = 1,
    ):
        """
        Searcher that conducts Simulated Annealing.
        We will conduct the search in a polling manner.
        :param opt_settings: the given optimization settings as starting point
        :param population: the number of hold state
        :param evaluation_cnt_limit: the limit of maximum evaluate count.
        :param temperature: initial temperature, used to control the probability of accepting worse optimization setting.
        :param cooling_rate: used to lower the temperature, the less this value is,
        the sooner worse optimization setting is rejected.
        """
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            mutation_bit_rate=mutation_bit_rate,
            constrains_file=constrains_file,
            time_limit=time_limit,
            gcc_param_bit_rate=gcc_param_bit_rate,
        )
        self.temperature: float = temperature
        self.cooling_rate: float = cooling_rate

    def search(self):
        current_energy = self.init_evaluate()
        current_state = [[__ for __ in _] for _ in self.opt_settings]
        best_state = [[__ for __ in _] for _ in self.opt_settings]
        best_energy = current_energy[:]

        while True:
            start_time = time.time()

            if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                break
            for idx in range(len(current_state)):
                print(
                    f"[SASearcher.search] remain evaluation count: {self.evaluation_cnt_limit}"
                )
                if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                    break
                # randomly generate a new optimization setting

                print(f"[SASearcher.search] start mutation")

                new_state = self.mutation(current_state[idx])

                # calculate its energy (performance)
                old_energy = current_energy[idx]

                print(f"[SASearcher.search] start evaluate")

                new_energy = self.evaluate(new_state)
                delta_energy = new_energy - old_energy

                # whether we accept this result
                # note that, any invalid optimization setting is rejected by this searcher.

                print(f"[SASearcher.search] start update")

                if (
                    delta_energy > 0
                    or random.random() < math.exp(delta_energy / self.temperature)
                ) and new_energy > 0:
                    current_state[idx] = new_state
                    current_energy[idx] = new_energy
                    # It seems meaningless to hold the local optima...
                    # if new_energy > best_energy[idx]:
                    #     best_state[idx] = new_state
                    #     best_energy[idx] = new_energy
                    if new_energy > self.best_performance:
                        self.best_setting = new_state
                        self.best_performance = new_energy
                self.temperature *= self.cooling_rate
            end_time = time.time()
            self.time_limit -= end_time - start_time
        # return self.best_setting, self.best_performance


class DESearcher(Searcher):

    def __init__(
        self,
        opt_settings: list,
        population: int,
        evaluation_cnt_limit: int,
        time_limit: int,
        crossover_bit_rate: float,
        mutation_bit_rate: float,
        evaluator: Evaluator,
        option_metadata: OptionMetadata,
        ai4c_metadata: AI4CMetaData,
        gcc_param_metadata: GCCParamMetadata,
        constrains_file: str,
        search_config: SearchConfig,
    ):
        """
        Searcher that use Genetic Algorithm.
        :param opt_settings: the given optimization settings
        :param population: the number of starting point, as well as the population size of Genetic Algorithm.
        :param evaluation_cnt_limit: the number of evaluation count, the search process will be stopped when evaluation cnt is reached.
        :param crossover_bit_rate: the number of bit to be crossover each time.
        :param mutation_bit_rate: the number of bit to be flipped each time.
        """
        super().__init__(
            ai4c_metadata=ai4c_metadata,
            evaluation_cnt_limit=evaluation_cnt_limit,
            evaluator=evaluator,
            opt_settings=opt_settings,
            option_metadata=option_metadata,
            gcc_param_metadata=gcc_param_metadata,
            population=population,
            search_config=search_config,
            constrains_file=constrains_file,
            time_limit=time_limit,
        )
        self.crossover_bit: int
        self.mutation_bit: int

        self.crossover_bit = int(crossover_bit_rate * len(opt_settings[0]))
        self.mutation_bit = int(mutation_bit_rate * len(opt_settings[0]))

    def selection(
        self, opt_settings, performances
    ) -> tuple[list[list[int]], list[float]]:
        """
        We directly select the maximum individuals.
        :param opt_settings: the given population(optimization settings)
        :param performances: the performances of the given optimization settings
        :return: the selected population and its performance
        """
        if len(opt_settings) == self.population:
            return opt_settings, performances
        opt_performance = sorted(
            [[performances[_], opt_settings[_]] for _ in range(len(performances))],
            reverse=True,
        )[: self.population]
        return [_[1] for _ in opt_performance], [_[0] for _ in opt_performance]

    def mutation(self, opt1: list, opt2: list, opt3: list) -> list[int]:
        """
        Conduct bit-flip mutation.
        :param opt1: given optimization setting
        :param opt2: given optimization setting
        :param opt3: given optimization setting
        :return: new optimization setting with self.mutation_bits flipped
        """
        return [opt1[_] + (opt2[_] - opt3[_]) for _ in range(len(opt1))]

    def search(self):
        performances = self.init_evaluate()
        population = self.opt_settings[:]
        while True:
            start_time = time.time()

            print(
                f"[DESearcher.search] remain evaluation count: {self.evaluation_cnt_limit}"
            )
            if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                break

            # selection
            population, performances = self.selection(population, performances)

            # mutation, crossover
            new_population = []
            for idx in range(len(population)):
                if self.evaluation_cnt_limit <= 0 or self.time_limit <= 0:
                    break
                # mutation
                i1 = idx
                i2, i3 = random.sample(
                    [_ for _ in range(len(population)) if _ != i1], 2
                )
                opt1 = population[i1]
                opt2 = population[i2]
                opt3 = population[i3]
                mutated = self.mutation(opt1, opt2, opt3)

                # cross over
                crossover_idx = random.sample(range(len(opt1)), self.crossover_bit)
                new_pop = [
                    mutated[_] if _ in crossover_idx else opt1[_]
                    for _ in range(len(opt1))
                ]

                # evaluate
                new_performance = self.evaluate(new_pop)
                if new_performance > performances[idx]:
                    performances[idx] = new_performance
                    population[idx] = new_pop
                if new_performance > self.best_performance:
                    self.best_performance = new_performance
                    self.best_setting = new_pop[:]
            end_time = time.time()
            self.time_limit -= end_time - start_time
        # return self.best_setting, self.best_performance


if __name__ == "__main__":
    # global configs: 10 start points
    from metadata_parser import OptionMetaDataParser
    from config_parser import RandomOptionConfigParser

    option_parser = OptionMetaDataParser(file="../config/option.txt")
    option_metadata = option_parser.parse()
    random_option_config_parser = RandomOptionConfigParser(option_metadata)
    random_option_config = random_option_config_parser.parse()

    from metadata_parser import AI4CMetaDataParser
    from config_parser import RandomAI4CConfigParser

    ai4c_parser = AI4CMetaDataParser(
        metadata_file="../config/ai4c.txt", config_file="../config/function.yaml"
    )
    ai4c_metadata = ai4c_parser.parse()
    random_ai4c_config_parser = RandomAI4CConfigParser(ai4c_metadata)
    random_ai4c_config = random_ai4c_config_parser.parse()

    from codec import encode

    # c = encode(ai4c_config=random_ai4c_config,
    #            ai4c_metadata=ai4c_metadata,
    #            option_config=random_option_config,
    #            option_metadata=option_metadata)

    opt_settings = []
    for _ in range(5):
        random_ai4c_config = random_ai4c_config_parser.parse()
        random_option_config = random_option_config_parser.parse()
        c = encode(
            ai4c_config=random_ai4c_config,
            ai4c_metadata=ai4c_metadata,
            option_config=random_option_config,
            option_metadata=option_metadata,
        )
        opt_settings.append(c)

    ai4c_option_file = (
        "/usr/local/lib/python3.9/site-packages/ai4c/autotuner/yaml/coarse_options.yaml"
    )
    ai4c_plagin_lib = "/usr/local/lib/python3.9/site-packages/ai4c/lib/coarse_option_tuning_plugin_gcc12.so"
    o3_project_path = "/home/scy/2025.04.17.doris.exceeds.O3/src/doris.O3"
    opt_project_path = "/home/scy/2025.04.17.doris.exceeds.O3/src/doris.turning"

    search_config: SearchConfig = SearchConfig(
        ai4c_config_dir="../yamls",
        ai4c_option_file=ai4c_option_file,
        ai4c_plagin_lib=ai4c_plagin_lib,
        opt_record_file="opt.recorder.txt",
        perf_record_file="perf.recorder.txt",
    )
    doris_evaluator: Evaluator = DorisEvaluator(
        o3_project_path=o3_project_path,
        opt_project_path=opt_project_path,
        repeat_time=100,
    )

    # assume that we have 5 optimization settings, each optimization setting with 100 options.
    # opt_settings = [[random.randint(0, 1) for _ in range(100)] for _ in range(100)]
    # test random searcher
    random_searcher = RandomSearcher(
        opt_settings=opt_settings,
        population=5,
        evaluation_cnt_limit=50,
        mutation_bit_rate=50,
        ai4c_metadata=ai4c_metadata,
        evaluator=doris_evaluator,
        option_metadata=option_metadata,
        search_config=search_config,
        constrains_file="../config/constrains.txt",
        time_limit=23 * 60 * 60,
    )
    print(random_searcher.search())

    # # test GA searcher
    # GA_searcher = GASearcher(opt_settings=opt_settings, population=5, evaluation_cnt=50, crossover_bit=50,
    #                          mutation_bit=50)
    # print(GA_searcher.search())
    #
    # # test PSO searcher
    # PSO_searcher = PSOSearcher(opt_settings=opt_settings, population=5, evaluation_cnt=50)
    # print(PSO_searcher.search())
    #
    # # test SA searcher
    # SA_searcher = SASearcher(opt_settings=opt_settings, population=5, evaluation_cnt=50)
    # print(SA_searcher.search())
    #
    # # test DE searcher
    # DE_searcher = DESearcher(opt_settings=opt_settings, population=5, evaluation_cnt=50, mutation_bit=50,
    #                          crossover_bit=50)
    # print(DE_searcher.search())
