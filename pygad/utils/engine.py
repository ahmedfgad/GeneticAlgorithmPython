import numpy
import random
import warnings
from pygad.utils.parallel import FitnessEvaluation

class GAEngine(FitnessEvaluation):

    def round_genes(self, solutions):
        """
        Convert and round genes in ``solutions`` using ``self.gene_type``.
        When ``gene_type_single`` is
        True, the same dtype and precision are applied to every gene;
        otherwise the per-gene dtype / precision pair is used.

        Parameters
        ----------
        solutions : numpy.ndarray
            A 2D array of solutions to round.

        Returns
        -------
        solutions : numpy.ndarray
            The converted and rounded array. The original array is
            updated when its dtype can hold the configured gene types.
        """
        converted_solutions = self.change_population_dtype_and_round(solutions)
        if isinstance(solutions, numpy.ndarray) and solutions.dtype == converted_solutions.dtype:
            solutions[:] = converted_solutions
            return solutions
        return converted_solutions

    def initialize_population(self, allow_duplicate_genes, gene_type, gene_constraint):
        """
        Generate and store the initial population using the validated
        initialization settings. Sample the values, apply constraints,
        and repair duplicate genes when they are not allowed. The working
        population and its initial snapshot are independent arrays.

        Parameters
        ----------
        allow_duplicate_genes : bool
            Whether repeated gene values are allowed within a solution.
        gene_type : type or list
            Retained for compatibility. Sampling uses the validated
            gene types and optional precisions stored in self.gene_type.
        gene_constraint : list or None
            Validated per-gene constraint callables.
        """
        # Store the policies passed by the constructor or by a caller
        # explicitly rebuilding the initial population.
        self.allow_duplicate_genes = allow_duplicate_genes
        self.gene_constraint = gene_constraint
        self.pop_size = (self.sol_per_pop, self.num_genes)
        self.population = self.generate_initial_population(self.sol_per_pop)
        self.initial_population = self.population.copy()

    def generate_initial_population(self, num_solutions):
        """
        Return new solutions using the initialization settings. Sampling
        in bulk avoids rebuilding finite candidate sets for every solution
        and calling the sampler for every value. NSGA-III population growth
        uses this same method.
        """
        continuous_space = (self.gene_space is None or
                            (type(self.gene_space) is dict and 'step' not in self.gene_space))
        if self.gene_type_single and self.gene_type[0] in self.supported_float_types and continuous_space:
            if self.gene_space is None:
                lower = numpy.minimum(self.init_range_low, self.init_range_high)
                upper = numpy.maximum(self.init_range_low, self.init_range_high)
            else:
                lower = min(self.gene_space['low'], self.gene_space['high'])
                upper = max(self.gene_space['low'], self.gene_space['high'])
            # A single draw retains the traditional solution-then-gene
            # order for continuous populations while avoiding scalar calls.
            population = numpy.random.uniform(lower, upper, size=(num_solutions, self.num_genes))
            for gene_index in range(self.num_genes):
                if self.gene_space is None:
                    gene_lower, gene_upper = self.get_initial_population_range(gene_index)
                    gene_lower, gene_upper = sorted([gene_lower, gene_upper])
                else:
                    gene_lower, gene_upper = lower, upper
                population[:, gene_index] = self._convert_initial_population_range_values(
                    gene_index, gene_lower, gene_upper, population[:, gene_index])
        else:
            population = numpy.empty((num_solutions, self.num_genes), dtype=object)
            for gene_index in range(self.num_genes):
                population[:, gene_index] = self.sample_initial_population_gene_values(
                    gene_index, num_solutions)
        return self.prepare_initial_population(population)

    def prepare_initial_population(self, population):
        """
        Convert a generated or supplied population, then apply constraints
        and duplicate repair. Existing supplied values need not belong to
        the gene space or initialization range. Any replacement uses the
        initialization settings for its own gene.
        """
        population = self.change_population_dtype_and_round(population)
        population = self.apply_initial_population_gene_constraints(population)
        if not self.allow_duplicate_genes:
            population = self.solve_duplicate_genes_in_population(
                population, build_initial_pop=True)
        return population

    def apply_initial_population_gene_constraints(self, population):
        """
        Replace values rejected by their constraints using converted
        initialization candidates. Constraints see the complete solution
        and are applied in gene-index order. Leave the existing value and
        warn when no candidate satisfies a constraint.
        """
        if self.gene_constraint is None:
            return population
        for solution in population:
            for gene_index, constraint in enumerate(self.gene_constraint):
                if constraint is None:
                    continue
                accepted_values = self.filter_gene_values_by_constraint(
                    [solution[gene_index]], solution, gene_index, warn=False)
                if accepted_values is not None:
                    if len(accepted_values) != 1:
                        raise ValueError("A gene constraint checking a single value must return an empty list or NumPy array, or one containing only that value.")
                    continue
                candidates = self.get_initial_population_gene_candidates(
                    gene_index, self.sample_size, all_integer_values=False)
                accepted_values = self.filter_gene_values_by_constraint(
                    candidates, solution, gene_index, warn=False)
                if accepted_values is None:
                    if not self.suppress_warnings:
                        warnings.warn(f"No value satisfied the constraint for the gene at index {gene_index} with value {solution[gene_index]} while creating the initial population.")
                else:
                    solution[gene_index] = random.choice(accepted_values)
        return population

    def cal_pop_fitness(self):
        """Compute population fitness with the same cache rules in all modes."""
        try:
            if not self.valid_parameters:
                raise Exception("ERROR calling the cal_pop_fitness() method: "
                                "Please check the parameters passed while creating "
                                "an instance of the GA class.")

            if type(self.best_solutions) is numpy.ndarray:
                self.best_solutions = self.best_solutions.tolist()
            saved_solutions = (self.solutions.tolist()
                               if type(self.solutions) is numpy.ndarray
                               else self.solutions)
            parents = (self.last_generation_parents.tolist()
                       if self.last_generation_parents is not None else [])
            elites = (self.last_generation_elitism.tolist()
                      if self.last_generation_elitism is not None else [])
            pop_fitness = [None] * len(self.population)
            missing_indices = []
            for index, solution in enumerate(self.population):
                values = solution.tolist()
                if self.save_solutions and values in saved_solutions:
                    fitness = self.solutions_fitness[saved_solutions.index(values)]
                elif self.save_best_solutions and values in self.best_solutions:
                    fitness = self.best_solutions_fitness[self.best_solutions.index(values)]
                elif self.keep_elitism > 0 and values in elites:
                    previous_index = self.last_generation_elitism_indices[elites.index(values)]
                    fitness = self.previous_generation_fitness[previous_index]
                elif self.keep_parents != 0 and values in parents:
                    previous_index = self.last_generation_parents_indices[parents.index(values)]
                    fitness = self.previous_generation_fitness[previous_index]
                else:
                    missing_indices.append(index)
                    continue
                pop_fitness[index] = fitness

            fitness_values = self._evaluate_fitness(self.population, missing_indices)
            for index, fitness in zip(missing_indices, fitness_values):
                pop_fitness[index] = fitness
            return numpy.array(pop_fitness)
        except Exception as ex:
            self.logger.exception(ex)
            raise

    def run(self):
        """
        Run the genetic algorithm for ``self.num_generations``
        generations. This is the main entry point for users: it sets
        up the bookkeeping lists, evaluates the initial population,
        runs the generational loop (select, crossover, mutate, update
        population, re-evaluate, callbacks, check stop criteria), and
        finalizes the best-solution data after the last generation.

        Calls the optional user callbacks ``on_start``,
        ``on_fitness``, ``on_parents``, ``on_crossover``,
        ``on_mutation``, ``on_generation`` and ``on_stop`` at the
        appropriate points.

        Raises
        ------
        Exception
            If ``self.valid_parameters`` is False, meaning the GA was
            built with invalid parameters.
        TypeError
            If an NSGA-II / NSGA-III parent selection type is used on
            a single-objective problem.
        ValueError
            If the ``stop_criteria`` parameter is malformed for the
            current number of objectives.
        """
        self._fitness_run_active = True
        try:
            if self.valid_parameters == False:
                raise Exception("Error calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

            # Starting from PyGAD 2.18.0, the 4 properties (best_solutions, best_solutions_fitness, solutions, and solutions_fitness) are no longer reset with each call to the run() method. Instead, they are extended.
            # For example, if there are 50 generations and the user set save_best_solutions=True, then the length of the 2 properties best_solutions and best_solutions_fitness will be 50 after the first call to the run() method, then 100 after the second call, 150 after the third, and so on.

            # self.best_solutions: Holds the best solution in each generation.
            if type(self.best_solutions) is numpy.ndarray:
                self.best_solutions = self.best_solutions.tolist()
            # self.best_solutions_fitness: A list holding the fitness value of the best solution for each generation.
            if type(self.best_solutions_fitness) is numpy.ndarray:
                self.best_solutions_fitness = list(self.best_solutions_fitness)
            # self.solutions: Holds the solutions in each generation.
            if type(self.solutions) is numpy.ndarray:
                self.solutions = self.solutions.tolist()
            # self.solutions_fitness: Holds the fitness of the solutions in each generation.
            if type(self.solutions_fitness) is numpy.ndarray:
                self.solutions_fitness = list(self.solutions_fitness)

            if not (self.on_start is None):
                self.on_start(self)

            # Reset the counters used by the "evaluations_<N>" and
            # "time_<seconds>" stop criteria. Each run() call should
            # only count the work it did itself.
            self.num_fitness_evaluations = 0
            import time as _time
            self.run_start_time = _time.monotonic()

            stop_run = False

            # To continue from where we stopped, the first generation index should start from the value of the 'self.generations_completed' parameter.
            if self.generations_completed != 0 and type(self.generations_completed) in self.supported_int_types:
                # If the 'self.generations_completed' parameter is not '0', then this means we continue execution.
                generation_first_idx = self.generations_completed
                generation_last_idx = self.num_generations + self.generations_completed
            else:
                # If the 'self.generations_completed' parameter is '0', then start from scratch.
                generation_first_idx = 0
                generation_last_idx = self.num_generations

            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            # Know whether the problem is SOO or MOO.
            if type(self.last_generation_fitness[0]) in self.supported_int_float_types:
                # Single-objective problem.
                # If the problem is SOO, the parent selection type cannot be nsga2/nsga3 or their tournament variants.
                if self.parent_selection_type in ['nsga2', 'tournament_nsga2', 'nsga3', 'tournament_nsga3']:
                    raise TypeError(f"Incorrect parent selection type. The fitness function returned a single numeric fitness value which means the problem is single-objective. But the parent selection type {self.parent_selection_type} is used which only works for multi-objective optimization problems.")
            elif type(self.last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                # Multi-objective problem.
                if self.parent_selection_type in ('nsga3', 'tournament_nsga3'):
                    # The reference points are created before starting the evolution and after the initial fitness is calculated.
                    # The number of reference points is determined based on:
                    #     1) The number of divisions (passed by the user).
                    #     2) The number of objectives (only known after the fitness is calculated).
                    # In PyGAD, the number of objectives are known only from the length of the returned result of the fitness function.
                    # This is how NSGA-III knows the number of objectives from the calculated fitness.
                    # It is time to build the reference points.
                    self._bootstrap_nsga3_reference_points()

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

            # Appending the best solution in the initial population to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(list(best_solution))

            for generation in range(generation_first_idx, generation_last_idx):

                self.run_loop_head(best_solution_fitness)

                # Call the 'run_select_parents()' method to select the parents.
                # It edits these 2 instance attributes:
                    # 1) last_generation_parents: A NumPy array of the selected parents.
                    # 2) last_generation_parents_indices: A 1D NumPy array of the indices of the selected parents.
                self.run_select_parents()

                # Call the 'run_crossover()' method to select the offspring.
                # It edits these 2 instance attributes:
                    # 1) last_generation_offspring_crossover: A NumPy array of the selected offspring.
                    # 2) last_generation_elitism: A NumPy array of the current generation elitism. Applicable only if the 'keep_elitism' parameter > 0.
                self.run_crossover()

                # Call the 'run_mutation()' method to mutate the selected offspring.
                # It edits this instance attribute:
                    # 1) last_generation_offspring_mutation: A NumPy array of the mutated offspring.
                self.run_mutation()

                # Call the 'run_update_population()' method to update the population after both crossover and mutation operations complete.
                # It edits this instance attribute:
                    # 1) population: A NumPy array of the population of solutions/chromosomes.
                self.run_update_population()

                # The generations_completed attribute holds the number of the last completed generation.
                self.generations_completed = generation + 1

                self.previous_generation_fitness = self.last_generation_fitness.copy()
                # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
                self.last_generation_fitness = self.cal_pop_fitness()

                best_solution, best_solution_fitness, best_match_idx = self.best_solution(
                    pop_fitness=self.last_generation_fitness)

                # Appending the best solution in the current generation to the best_solutions list.
                if self.save_best_solutions:
                    self.best_solutions.append(list(best_solution))

                # Note: Any code that has loop-dependent statements (e.g. continue, break, etc.) must be kept inside the loop of the 'run()' method. It cannot be moved to another method to clean up the run() method.
                # If the on_generation attribute is not None, then call the callback function after the generation.
                if not (self.on_generation is None):
                    r = self.on_generation(self)
                    if type(r) is str and r.lower() == "stop":
                        break

                if not self.stop_criteria is None:
                    for criterion in self.stop_criteria:
                        if criterion[0] == "reach":
                            # Single-objective problem.
                            if type(self.last_generation_fitness[0]) in self.supported_int_float_types:
                                if max(self.last_generation_fitness) >= criterion[1]:
                                    stop_run = True
                                    break
                            # Multi-objective problem.
                            elif type(self.last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                                # Validate the value passed to the criterion.
                                if len(criterion[1:]) == 1:
                                    # There is a single value used across all the objectives.
                                    pass
                                elif len(criterion[1:]) > 1:
                                    # There are multiple values. The number of values must be equal to the number of objectives.
                                    if len(criterion[1:]) == len(self.last_generation_fitness[0]):
                                        pass
                                    else:
                                        self.valid_parameters = False
                                        raise ValueError(f"When the 'reach' keyword is used with the 'stop_criteria' parameter for solving a multi-objective problem, then the number of numeric values following the keyword can be:\n1) A single numeric value to be used across all the objective functions.\n2) A number of numeric values equal to the number of objective functions.\nBut the value {criterion} found with {len(criterion)-1} numeric values which is not equal to the number of objective functions {len(self.last_generation_fitness[0])}.")

                                stop_run = True
                                for obj_idx in range(len(self.last_generation_fitness[0])):
                                    # Use the objective index to return the proper value for the criterion.

                                    if len(criterion[1:]) == len(self.last_generation_fitness[0]):
                                        reach_fitness_value = criterion[obj_idx + 1]
                                    elif len(criterion[1:]) == 1:
                                        reach_fitness_value = criterion[1]
                                    else:
                                        # Unexpected to be reached, but it is safer to handle it.
                                        self.valid_parameters = False
                                        raise ValueError(f"The number of values {len(criterion[1:])} does not equal the number of objectives {len(self.last_generation_fitness[0])}.")

                                    if max(self.last_generation_fitness[:, obj_idx]) >= reach_fitness_value:
                                        pass
                                    else:
                                        stop_run = False
                                        break
                        elif criterion[0] == "saturate":
                            criterion[1] = int(criterion[1])
                            if self.generations_completed >= criterion[1]:
                                # Single-objective problem.
                                if type(self.last_generation_fitness[0]) in self.supported_int_float_types:
                                    if (self.best_solutions_fitness[self.generations_completed - criterion[1]] - self.best_solutions_fitness[self.generations_completed - 1]) == 0:
                                        stop_run = True
                                        break
                                # Multi-objective problem.
                                elif type(self.last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                                    stop_run = True
                                    for obj_idx in range(len(self.last_generation_fitness[0])):
                                        if (self.best_solutions_fitness[self.generations_completed - criterion[1]][obj_idx] - self.best_solutions_fitness[self.generations_completed - 1][obj_idx]) == 0:
                                            pass
                                        else:
                                            stop_run = False
                                            break
                        elif criterion[0] == "time":
                            # Stop when the time spent inside run()
                            # passes the user limit.
                            import time as _time
                            if _time.monotonic() - self.run_start_time >= float(criterion[1]):
                                stop_run = True
                                break
                        elif criterion[0] == "evaluations":
                            # Stop when the number of fitness calls
                            # reaches the user limit.
                            if self.num_fitness_evaluations >= int(criterion[1]):
                                stop_run = True
                                break

                if stop_run:
                    break

            # Save the fitness of the last generation.
            if self.save_solutions:
                # self.solutions.extend(self.population.copy())
                population_as_list = self.population.copy()
                population_as_list = [list(item) for item in population_as_list]
                self.solutions.extend(population_as_list)

                self.solutions_fitness.extend(self.last_generation_fitness)

            # Call the run_select_parents() method to update these 2 attributes according to the 'last_generation_fitness' attribute:
                # 1) last_generation_parents 2) last_generation_parents_indices
            # Set 'call_on_parents=False' to avoid calling the callable 'on_parents' because this step is not part of the cycle.
            self.run_select_parents(call_on_parents=False)

            # Update the elitism according to the 'last_generation_fitness' attribute.
            if self.keep_elitism > 0:
                self.last_generation_elitism, self.last_generation_elitism_indices = self.steady_state_selection(self.last_generation_fitness,
                                                                                                                 num_parents=self.keep_elitism)

            # Save the fitness value of the best solution.
            _, best_solution_fitness, _ = self.best_solution(
                pop_fitness=self.last_generation_fitness)
            self.best_solutions_fitness.append(best_solution_fitness)

            self.best_solution_generation = numpy.where(numpy.array(
                self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
            # After the run() method completes, the run_completed flag is changed from False to True.
            # Set to True only after the run() method completes gracefully.
            self.run_completed = True

            if not (self.on_stop is None):
                self.on_stop(self, self.last_generation_fitness)

            # Converting the 'best_solutions' list into a NumPy array.
            self.best_solutions = numpy.array(self.best_solutions, dtype=self.population.dtype)

            # Update previous_generation_fitness because it is used to get the fitness of the parents.
            self.previous_generation_fitness = self.last_generation_fitness.copy()

            # Converting the 'solutions' list into a NumPy array.
            # self.solutions = numpy.array(self.solutions)
        except Exception as ex:
            self.logger.exception(ex)
            # sys.exit(-1)
            raise ex

        finally:
            self._fitness_run_active = False
            self._shutdown_fitness_executor()

    def run_loop_head(self, best_solution_fitness):
        """
        Run the bookkeeping that takes place at the top of every
        generation: call ``self.on_fitness`` if set (with optional
        validation of the returned values), append the running best
        fitness to ``self.best_solutions_fitness``, and append the
        current population and fitness to ``self.solutions`` /
        ``self.solutions_fitness`` when ``self.save_solutions`` is
        True.

        Internal helper. Not meant to be called by users.

        Parameters
        ----------
        best_solution_fitness : numeric or numpy.ndarray
            Fitness of the best solution in the previous generation.

        Raises
        ------
        ValueError
            If ``on_fitness`` returns an iterable whose shape does not
            match the population fitness, or an unsupported type.
        """
        if not (self.on_fitness is None):
            on_fitness_output = self.on_fitness(self, 
                                                self.last_generation_fitness)

            if on_fitness_output is None:
                pass
            else:
                if type(on_fitness_output) in [tuple, list, numpy.ndarray, range]:
                    on_fitness_output = numpy.array(on_fitness_output)
                    if on_fitness_output.shape == self.last_generation_fitness.shape:
                        self.last_generation_fitness = on_fitness_output
                    else:
                        raise ValueError(f"Size mismatch between the output of on_fitness() {on_fitness_output.shape} and the expected fitness output {self.last_generation_fitness.shape}.")
                else:
                    raise ValueError(f"The output of on_fitness() is expected to be tuple/list/range/numpy.ndarray but {type(on_fitness_output)} found.")

        # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
        self.best_solutions_fitness.append(best_solution_fitness)

        # Appending the solutions in the current generation to the solutions list.
        if self.save_solutions:
            # self.solutions.extend(self.population.copy())
            population_as_list = self.population.copy()
            population_as_list = [list(item) for item in population_as_list]
            self.solutions.extend(population_as_list)

            self.solutions_fitness.extend(self.last_generation_fitness)

    def run_select_parents(self, call_on_parents=True):
        """
        Run the parent-selection step of one generation. Calls
        ``self.select_parents`` (the operator chosen by
        ``parent_selection_type``), validates the shapes of the
        returned parents and indices, and updates these instance
        attributes:

        - ``self.last_generation_parents``: the selected parent
          solutions.
        - ``self.last_generation_parents_indices``: their indices
          inside ``self.population``.

        Optionally calls the user-supplied ``on_parents`` callback,
        which may replace the parents and / or their indices in place.

        Internal helper. Not meant to be called by users (any
        ``run_*`` method is part of the generational loop driven by
        ``run()``).

        Parameters
        ----------
        call_on_parents : bool
            When True, the ``on_parents`` callback is invoked after
            selection. Set to False on the post-run cleanup pass so
            the callback is not fired again.

        Raises
        ------
        TypeError
            If a user-supplied parent selection function returns
            objects that are not ``numpy.ndarray``.
        ValueError
            If the selected parents have the wrong shape or the
            ``on_parents`` callback returns an output that does not
            match the expected layout.
        """

        # Selecting the best parents in the population for mating.
        if callable(self.parent_selection_type):
            self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(self.last_generation_fitness,
                                                                                                     self.num_parents_mating,
                                                                                                     self)
            if not type(self.last_generation_parents) is numpy.ndarray:
                raise TypeError(f"The type of the iterable holding the selected parents is expected to be (numpy.ndarray) but {type(self.last_generation_parents)} found.")
            if not type(self.last_generation_parents_indices) is numpy.ndarray:
                raise TypeError(f"The type of the iterable holding the selected parents' indices is expected to be (numpy.ndarray) but {type(self.last_generation_parents_indices)} found.")
        else:
            self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(self.last_generation_fitness,
                                                                                                     num_parents=self.num_parents_mating)

        # Validate the output of the parent selection step: self.select_parents()
        if self.last_generation_parents.shape != (self.num_parents_mating, self.num_genes):
            if self.last_generation_parents.shape[0] != self.num_parents_mating:
                raise ValueError(f"Size mismatch between the size of the selected parents {self.last_generation_parents.shape} and the expected size {(self.num_parents_mating, self.num_genes)}. It is expected to select ({self.num_parents_mating}) parents but ({self.last_generation_parents.shape[0]}) selected.")
            elif self.last_generation_parents.shape[1] != self.num_genes:
                raise ValueError(f"Size mismatch between the size of the selected parents {self.last_generation_parents.shape} and the expected size {(self.num_parents_mating, self.num_genes)}. Parents are expected to have ({self.num_genes}) genes but ({self.last_generation_parents.shape[1]}) produced.")

        if self.last_generation_parents_indices.ndim != 1:
            raise ValueError(f"The iterable holding the selected parents indices is expected to have 1 dimension but ({len(self.last_generation_parents_indices)}) found.")
        elif len(self.last_generation_parents_indices) != self.num_parents_mating:
            raise ValueError(f"The iterable holding the selected parents indices is expected to have ({self.num_parents_mating}) values but ({len(self.last_generation_parents_indices)}) found.")

        if callable(self.parent_selection_type):
            self.last_generation_parents = self.change_population_dtype_and_round(self.last_generation_parents)

        if call_on_parents:
            if not (self.on_parents is None):
                on_parents_output = self.on_parents(self, 
                                                    self.last_generation_parents)
    
                if on_parents_output is None:
                    pass
                elif type(on_parents_output) in [list, tuple, numpy.ndarray]:
                    if len(on_parents_output) == 2:
                        on_parents_selected_parents, on_parents_selected_parents_indices = on_parents_output
                    else:
                        raise ValueError(f"The output of on_parents() is expected to be tuple/list/numpy.ndarray of length 2 but {type(on_parents_output)} of length {len(on_parents_output)} found.")
    
                    # Validate the parents.
                    if on_parents_selected_parents is None:
                                raise ValueError("The returned outputs of on_parents() cannot be None but the first output is None.")
                    else:
                        if type(on_parents_selected_parents) in [tuple, list, numpy.ndarray]:
                            on_parents_selected_parents = numpy.asarray(on_parents_selected_parents, dtype=object)
                            if on_parents_selected_parents.shape == self.last_generation_parents.shape:
                                self.last_generation_parents = on_parents_selected_parents
                            else:
                                raise ValueError(f"Size mismatch between the parents returned by on_parents() {on_parents_selected_parents.shape} and the expected parents shape {self.last_generation_parents.shape}.")
                        else:
                            raise ValueError(f"The output of on_parents() is expected to be tuple/list/numpy.ndarray but the first output type is {type(on_parents_selected_parents)}.")
    
                    # Validate the parents indices.
                    if on_parents_selected_parents_indices is None:
                        raise ValueError("The returned outputs of on_parents() cannot be None but the second output is None.")
                    else:
                        if type(on_parents_selected_parents_indices) in [tuple, list, numpy.ndarray, range]:
                            on_parents_selected_parents_indices = numpy.array(on_parents_selected_parents_indices)
                            if on_parents_selected_parents_indices.shape == self.last_generation_parents_indices.shape:
                                # Add this new instance attribute.
                                self.last_generation_parents_indices = on_parents_selected_parents_indices
                            else:
                                raise ValueError(f"Size mismatch between the parents indices returned by on_parents() {on_parents_selected_parents_indices.shape} and the expected crossover output {self.last_generation_parents_indices.shape}.")
                        else:
                            raise ValueError(f"The output of on_parents() is expected to be tuple/list/range/numpy.ndarray but the second output type is {type(on_parents_selected_parents_indices)}.")
    
                else:
                    raise TypeError(f"The output of on_parents() is expected to be tuple/list/numpy.ndarray but {type(on_parents_output)} found.")

        if call_on_parents and self.on_parents is not None:
            self.last_generation_parents = self.change_population_dtype_and_round(self.last_generation_parents)

    def run_crossover(self):
        """
        Run the crossover step of one generation. Produces
        ``self.num_offspring`` offspring from the selected parents and
        updates these instance attributes:

        - ``self.last_generation_offspring_crossover``: the offspring
          generated by crossover (or copied from parents when
          ``crossover_type`` is None).
        - ``self.last_generation_elitism``: the top
          ``self.keep_elitism`` solutions in the population, used
          later by ``run_update_population`` to seat the elite in the
          next generation.

        Optionally calls the user-supplied ``on_crossover`` callback,
        which may replace the offspring in place.

        Internal helper. Not meant to be called by users.

        Raises
        ------
        TypeError
            If a user-supplied crossover function returns an object
            that is not a ``numpy.ndarray``.
        ValueError
            If the crossover output has the wrong shape or the
            ``on_crossover`` callback returns an output that does not
            match the expected layout.
        """

        # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        if self.crossover_type is None:
            if self.keep_elitism == 0:
                num_parents_to_keep = self.num_parents_mating if self.keep_parents == - 1 else self.keep_parents
                if self.num_offspring <= num_parents_to_keep:
                    self.last_generation_offspring_crossover = self.last_generation_parents[0:self.num_offspring]
                else:
                    self.last_generation_offspring_crossover = numpy.concatenate(
                        (self.last_generation_parents, self.population[0:(self.num_offspring - self.last_generation_parents.shape[0])]))
            else:
                # The steady_state_selection() function is called to select the best solutions (i.e. elitism). The keep_elitism parameter defines the number of these solutions.
                # The steady_state_selection() function is still called here even if its output may not be used given that the condition of the next if statement is True. The reason is that it will be used later.
                self.last_generation_elitism, _ = self.steady_state_selection(self.last_generation_fitness,
                                                                              num_parents=self.keep_elitism)
                if self.num_offspring <= self.keep_elitism:
                    self.last_generation_offspring_crossover = self.last_generation_parents[0:self.num_offspring]
                else:
                    self.last_generation_offspring_crossover = numpy.concatenate(
                        (self.last_generation_elitism, self.population[0:(self.num_offspring - self.last_generation_elitism.shape[0])]))
        else:
            # Generating offspring using crossover.
            if callable(self.crossover_type):
                self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                          (self.num_offspring, self.num_genes),
                                                                          self)
                if not type(self.last_generation_offspring_crossover) is numpy.ndarray:
                    raise TypeError(f"The output of the crossover step is expected to be of type (numpy.ndarray) but {type(self.last_generation_offspring_crossover)} found.")
            else:
                self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                          offspring_size=(self.num_offspring, self.num_genes))
            if self.last_generation_offspring_crossover.shape != (self.num_offspring, self.num_genes):
                if self.last_generation_offspring_crossover.shape[0] != self.num_offspring:
                    raise ValueError(f"Size mismatch between the crossover output {self.last_generation_offspring_crossover.shape} and the expected crossover output {(self.num_offspring, self.num_genes)}. It is expected to produce ({self.num_offspring}) offspring but ({self.last_generation_offspring_crossover.shape[0]}) produced.")
                elif self.last_generation_offspring_crossover.shape[1] != self.num_genes:
                    raise ValueError(f"Size mismatch between the crossover output {self.last_generation_offspring_crossover.shape} and the expected crossover output {(self.num_offspring, self.num_genes)}. It is expected that the offspring has ({self.num_genes}) genes but ({self.last_generation_offspring_crossover.shape[1]}) produced.")

        if callable(self.crossover_type) and self.on_crossover is not None:
            self.last_generation_offspring_crossover = self.change_population_dtype_and_round(
                self.last_generation_offspring_crossover)

        # PyGAD 2.18.2 // The on_crossover() callback function is called even if crossover_type is None.
        if not (self.on_crossover is None):
            on_crossover_output = self.on_crossover(self, 
                                                    self.last_generation_offspring_crossover)
            if on_crossover_output is None:
                pass
            else:
                if type(on_crossover_output) in [tuple, list, numpy.ndarray]:
                    on_crossover_output = numpy.asarray(on_crossover_output, dtype=object)
                    if on_crossover_output.shape == self.last_generation_offspring_crossover.shape:
                        self.last_generation_offspring_crossover = on_crossover_output
                    else:
                        raise ValueError(f"Size mismatch between the output of on_crossover() {on_crossover_output.shape} and the expected crossover output {self.last_generation_offspring_crossover.shape}.")
                else:
                    raise ValueError(f"The output of on_crossover() is expected to be tuple/list/numpy.ndarray but {type(on_crossover_output)} found.")

        if callable(self.crossover_type) or self.on_crossover is not None:
            self.last_generation_offspring_crossover = self.prepare_operator_output(
                self.last_generation_offspring_crossover,
                build_initial_pop=self.crossover_type == 'sbx')

    def run_mutation(self):
        """
        Run the mutation step of one generation. Mutates the
        post-crossover offspring and updates this instance attribute:

        - ``self.last_generation_offspring_mutation``: the mutated
          offspring (or the unchanged crossover offspring when
          ``mutation_type`` is None).

        Optionally calls the user-supplied ``on_mutation`` callback,
        which may replace the mutated offspring in place.

        Internal helper. Not meant to be called by users.

        Raises
        ------
        TypeError
            If a user-supplied mutation function returns an object
            that is not a ``numpy.ndarray``.
        ValueError
            If the mutation output has the wrong shape or the
            ``on_mutation`` callback returns an output that does not
            match the expected layout.
        """

        # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        if self.mutation_type is None:
            self.last_generation_offspring_mutation = self.last_generation_offspring_crossover
        else:
            # Adding some variations to the offspring using mutation.
            if callable(self.mutation_type):
                self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover,
                                                                        self)
                if not type(self.last_generation_offspring_mutation) is numpy.ndarray:
                    raise TypeError(f"The output of the mutation step is expected to be of type (numpy.ndarray) but {type(self.last_generation_offspring_mutation)} found.")
            else:
                self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)

            if self.last_generation_offspring_mutation.shape != (self.num_offspring, self.num_genes):
                if self.last_generation_offspring_mutation.shape[0] != self.num_offspring:
                    raise ValueError(f"Size mismatch between the mutation output {self.last_generation_offspring_mutation.shape} and the expected mutation output {(self.num_offspring, self.num_genes)}. It is expected to produce ({self.num_offspring}) offspring but ({self.last_generation_offspring_mutation.shape[0]}) produced.")
                elif self.last_generation_offspring_mutation.shape[1] != self.num_genes:
                    raise ValueError(f"Size mismatch between the mutation output {self.last_generation_offspring_mutation.shape} and the expected mutation output {(self.num_offspring, self.num_genes)}. It is expected that the offspring has ({self.num_genes}) genes but ({self.last_generation_offspring_mutation.shape[1]}) produced.")

        if callable(self.mutation_type) and self.on_mutation is not None:
            self.last_generation_offspring_mutation = self.change_population_dtype_and_round(
                self.last_generation_offspring_mutation)

        # PyGAD 2.18.2 // The on_mutation() callback function is called even if mutation_type is None.
        if not (self.on_mutation is None):
            on_mutation_output = self.on_mutation(self, 
                                                  self.last_generation_offspring_mutation)

            if on_mutation_output is None:
                pass
            else:
                if type(on_mutation_output) in [tuple, list, numpy.ndarray]:
                    on_mutation_output = numpy.asarray(on_mutation_output, dtype=object)
                    if on_mutation_output.shape == self.last_generation_offspring_mutation.shape:
                        self.last_generation_offspring_mutation = on_mutation_output
                    else:
                        raise ValueError(f"Size mismatch between the output of on_mutation() {on_mutation_output.shape} and the expected mutation output {self.last_generation_offspring_mutation.shape}.")
                else:
                    raise ValueError(f"The output of on_mutation() is expected to be tuple/list/numpy.ndarray but {type(on_mutation_output)} found.")

        if callable(self.mutation_type) or self.on_mutation is not None:
            self.last_generation_offspring_mutation = self.prepare_operator_output(
                self.last_generation_offspring_mutation,
                build_initial_pop=self.mutation_type == 'polynomial')

    def prepare_operator_output(self, population, build_initial_pop=False):
        """
        Convert an operator's population using the configured gene types
        and precision, then repair duplicates if they are disallowed.

        Parameters
        ----------
        population : numpy.ndarray
            Parents or offspring to prepare without modifying the input.
        build_initial_pop : bool
            Use initialization bounds for duplicate repair, as required
            for SBX crossover and polynomial mutation.

        Returns
        -------
        numpy.ndarray
            The converted population, with duplicate repair applied when
            allow_duplicate_genes is False.
        """
        if self.allow_duplicate_genes:
            return self.change_population_dtype_and_round(population)
        return self.solve_duplicate_genes_in_population(population, build_initial_pop=build_initial_pop)

    def run_update_population(self):
        """
        Build the next generation in ``self.population`` from the
        offspring produced by mutation plus, optionally, the retained
        parents (``keep_parents``) or elite (``keep_elitism``).

        Layout rules:

        - ``keep_elitism > 0``: top ``keep_elitism`` solutions sit at
          the front of the new population; the rest is the mutated
          offspring.
        - ``keep_elitism == 0`` and ``keep_parents == -1``: all
          selected parents sit at the front; the rest is offspring.
        - ``keep_elitism == 0`` and ``keep_parents == 0``: the new
          population is offspring only.
        - ``keep_elitism == 0`` and ``keep_parents > 0``: top
          ``keep_parents`` selected parents sit at the front; the
          rest is offspring.

        Internal helper. Not meant to be called by users.
        """

        # Update the population attribute according to the offspring generated.
        if self.keep_elitism == 0:
            # If the keep_elitism parameter is 0, then the keep_parents parameter will be used to decide if the parents are kept in the next generation.
            if self.keep_parents == 0:
                self.population = self.last_generation_offspring_mutation
            elif self.keep_parents == -1:
                # Creating the new population based on the parents and offspring.
                self.population[0:self.last_generation_parents.shape[0],:] = self.last_generation_parents
                self.population[self.last_generation_parents.shape[0]:, :] = self.last_generation_offspring_mutation
            elif self.keep_parents > 0:
                parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness,
                                                                 num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0],:] = parents_to_keep
                self.population[parents_to_keep.shape[0]:,:] = self.last_generation_offspring_mutation
        else:
            self.last_generation_elitism, self.last_generation_elitism_indices = self.steady_state_selection(self.last_generation_fitness,
                                                                                                             num_parents=self.keep_elitism)
            self.population[0:self.last_generation_elitism.shape[0],:] = self.last_generation_elitism
            self.population[self.last_generation_elitism.shape[0]:, :] = self.last_generation_offspring_mutation

    def best_solution(self, pop_fitness=None):
        """
        Return the best solution found in the latest population. For
        single-objective problems "best" is the solution with the
        maximum fitness. For multi-objective problems, the best
        solution is the top entry of the NSGA-II sort (front 0,
        highest crowding distance).

        Parameters
        ----------
        pop_fitness : list, tuple, numpy.ndarray, or None
            Pre-computed fitness for the current population. When
            None, ``cal_pop_fitness`` is called to compute it. Useful
            to avoid re-evaluating the fitness function.

        Returns
        -------
        best_solution : numpy.ndarray
            The genes of the best solution.
        best_solution_fitness : numeric or numpy.ndarray
            Its fitness value.
        best_match_idx : int
            Its index inside ``self.population``.

        Raises
        ------
        ValueError
            If ``pop_fitness`` is provided but its length does not
            match the population, or its type is not list / tuple /
            numpy.ndarray.
        """

        try:
            if pop_fitness is None:
                # If the 'pop_fitness' parameter is not passed, then we have to call the 'cal_pop_fitness()' method to calculate the fitness of all solutions in the latest population.
                pop_fitness = self.cal_pop_fitness()
            # Verify the type of the 'pop_fitness' parameter.
            elif type(pop_fitness) in [tuple, list, numpy.ndarray]:
                # Verify that the length of the passed population fitness matches the length of the 'self.population' attribute.
                if len(pop_fitness) == len(self.population):
                    # This successfully verifies the 'pop_fitness' parameter.
                    pass
                else:
                    raise ValueError(f"The length of the list/tuple/numpy.ndarray passed to the 'pop_fitness' parameter ({len(pop_fitness)}) must match the length of the 'self.population' attribute ({len(self.population)}).")
            else:
                raise ValueError(f"The type of the 'pop_fitness' parameter is expected to be list, tuple, or numpy.ndarray but ({type(pop_fitness)}) found.")

            # Return the index of the best solution that has the best fitness value.
            # For multi-objective optimization: find the index of the solution with the maximum fitness in the first objective,
            # break ties using the second objective, then third, etc.
            pop_fitness_arr = numpy.array(pop_fitness)
            # Get the indices that would sort by all objectives in descending order
            if pop_fitness_arr.ndim == 1:
                # Single-objective optimization.
                best_match_idx = numpy.where(
                 pop_fitness == numpy.max(pop_fitness))[0][0]
            elif pop_fitness_arr.ndim == 2:
                # Multi-objective optimization.
                # Use NSGA-2 to sort the solutions using the fitness.
                # Set find_best_solution=True to avoid overriding the pareto_fronts instance attribute.
                best_match_list = self.sort_solutions_nsga2(fitness=pop_fitness,
                                                            find_best_solution=True)

                # Get the first index of the best match.
                best_match_idx = best_match_list[0]

            best_solution = self.population[best_match_idx, :].copy()
            best_solution_fitness = pop_fitness[best_match_idx]
        except Exception as ex:
            self.logger.exception(ex)
            # sys.exit(-1)
            raise ex

        return best_solution, best_solution_fitness, best_match_idx

    def _bootstrap_nsga3_reference_points(self):
        """
        Build the reference-point grid once, right after the first
        fitness evaluation. The number of objectives M is read from the
        length of the first fitness vector.

        If ``sol_per_pop`` is smaller than the number of reference
        points, grow the population to match and re-evaluate fitness so
        the GA loop can carry on with a valid population.
        """
        num_objectives = len(self.last_generation_fitness[0])
        self.nsga3_reference_points = self.nsga3_generate_reference_points(
            num_objectives, self.nsga3_num_divisions)
        required_size = len(self.nsga3_reference_points)
        if self.sol_per_pop < required_size:
            self._nsga3_grow_population(required_size, num_objectives)

    def _nsga3_grow_population(self, required_size, num_objectives):
        """
        Append random solutions to ``self.population`` until the size
        equals ``required_size``, then re-evaluate fitness. The new
        rows follow the same gene space, init range, gene type, gene
        constraints, and ``allow_duplicate_genes`` rules used to build
        the initial population, so the grown population is
        indistinguishable from one created with ``sol_per_pop`` set to
        ``required_size`` from the start.
        """
        original_size = self.sol_per_pop
        if not self.suppress_warnings:
            warnings.warn(
                f"sol_per_pop ({original_size}) is smaller than the number of "
                f"NSGA-III reference points ({required_size}) for M={num_objectives} "
                f"objectives and nsga3_num_divisions={self.nsga3_num_divisions}. "
                f"Growing the population to {required_size} random solutions "
                f"and re-evaluating fitness."
            )
        extra = self._nsga3_generate_extra_random_solutions(
            required_size - original_size)
        self.population = numpy.vstack([self.population, extra])
        self.sol_per_pop = required_size
        self.pop_size = (required_size, self.num_genes)
        # Shared helper on the Validation mixin keeps the rule in one place.
        self._refresh_num_offspring()
        self.last_generation_fitness = self.cal_pop_fitness()

    def _nsga3_generate_extra_random_solutions(self, count):
        """Generate NSGA-III growth rows using the initialization settings."""
        return self.generate_initial_population(count)

    def _nsga3_generate_single_random_gene(self, gene_idx, partial_solution):
        """Compatibility helper for sampling one initialization value."""
        return self.sample_initial_population_gene_values(gene_idx, 1)[0]

    def _nsga3_apply_gene_constraints(self, population):
        """Compatibility helper for applying initialization constraints."""
        return self.apply_initial_population_gene_constraints(population)

    def _nsga3_resolve_duplicate_genes(self, population):
        """Repair newly generated rows using initialization rules."""
        return self.solve_duplicate_genes_in_population(population, build_initial_pop=True)
