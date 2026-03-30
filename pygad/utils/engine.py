import numpy
import random
import warnings
import concurrent.futures

class GAEngine:

    def round_genes(self, solutions):
        for gene_idx in range(self.num_genes):
            if self.gene_type_single:
                if not self.gene_type[1] is None:
                    solutions[:, gene_idx] = numpy.round(solutions[:, gene_idx],
                                                         self.gene_type[1])
            else:
                if not self.gene_type[gene_idx][1] is None:
                    solutions[:, gene_idx] = numpy.round(numpy.asarray(solutions[:, gene_idx],
                                                                       dtype=self.gene_type[gene_idx][0]),
                                                         self.gene_type[gene_idx][1])
        return solutions

    def initialize_population(self,
                              allow_duplicate_genes,
                              gene_type,
                              gene_constraint):
        """
        Creates an initial population randomly as a NumPy array. The array is saved in the instance attribute named 'population'.

        It accepts:
            -allow_duplicate_genes: Whether duplicate genes are allowed or not.
            -gene_type: The data type of the genes.
            -gene_constraint: The constraints of the genes.

        This method assigns the values of the following 3 instance attributes:
            1. pop_size: Size of the population.
            2. population: Initially, holds the initial population and later updated after each generation.
            3. init_population: Keeping the initial population.
        """

        # Population size = (number of chromosomes, number of genes per chromosome)
        # The population will have sol_per_pop chromosome where each chromosome has num_genes genes.
        self.pop_size = (self.sol_per_pop, self.num_genes)

        # There are 4 steps to build the initial population:
            # 1) Generate the population.
            # 2) Change the data type and round the values.
            # 3) Check for the constraints.
            # 4) Solve duplicates if not allowed.

        # Create an empty population.
        self.population = numpy.empty(shape=self.pop_size, dtype=object)

        # 1) Create the initial population either randomly or using the gene space.
        if self.gene_space is None:
            # Create the initial population randomly.

            # Set gene_value=None to consider generating values for the initial population instead of generating values for mutation.
            # Loop through the genes, randomly generate the values of a single gene at a time, and insert the values of each gene to the population.
            for sol_idx in range(self.sol_per_pop):
                for gene_idx in range(self.num_genes):
                    range_min, range_max = self.get_initial_population_range(gene_index=gene_idx)
                    self.population[sol_idx, gene_idx] = self.generate_gene_value_randomly(range_min=range_min,
                                                                                           range_max=range_max,
                                                                                           gene_idx=gene_idx,
                                                                                           mutation_by_replacement=True,
                                                                                           gene_value=None,
                                                                                           sample_size=1,
                                                                                           step=1)

        else:
            # Generate the initial population using the gene_space.
            for sol_idx in range(self.sol_per_pop):
                for gene_idx in range(self.num_genes):
                    self.population[sol_idx, gene_idx] = self.generate_gene_value_from_space(gene_idx=gene_idx,
                                                                                             mutation_by_replacement=True,
                                                                                             gene_value=None,
                                                                                             solution=self.population[sol_idx],
                                                                                             sample_size=1)

        # 2) Change the data type and round all genes within the initial population.
        self.population = self.change_population_dtype_and_round(self.population)

        # Note that gene_constraint is not validated yet.
        # We have to set it as a property of the pygad.GA instance to retrieve without passing it as an additional parameter.
        self.gene_constraint = gene_constraint

        # 3) Enforce the gene constraints as much as possible.
        if self.gene_constraint is None:
            pass
        else:
            for sol_idx, solution in enumerate(self.population):
                for gene_idx in range(self.num_genes):
                    # Check that a constraint is available for the gene and that the current value does not satisfy that constraint
                    if self.gene_constraint[gene_idx]:
                        # Remember that the second argument to the gene constraint callable is a list/numpy.ndarray of the values to check if they meet the gene constraint.
                        values = [solution[gene_idx]]
                        filtered_values = self.gene_constraint[gene_idx](solution, values)
                        result = self.validate_gene_constraint_callable_output(selected_values=filtered_values,
                                                                               values=values)
                        if result:
                            pass
                        else:
                            raise Exception("The output from the gene_constraint callable/function must be a list or NumPy array that is subset of the passed values (second argument).")

                        if len(filtered_values) ==1 and filtered_values[0] != solution[gene_idx]:
                            # Error by the user's defined gene constraint callable.
                            raise Exception(f"It is expected to receive a list/numpy.ndarray from the gene_constraint callable with a single value equal to {values[0]}, but the value {filtered_values[0]} found.")

                        # Check if the gene value does not satisfy the gene constraint.
                        # Note that we already passed a list of a single value.
                        # It is expected to receive a list of either a single value or an empty list.
                        if len(filtered_values) < 1:
                            # Search for a value that satisfies the gene constraint.
                            range_min, range_max = self.get_initial_population_range(gene_index=gene_idx)
                            # While initializing the population, we follow a mutation by replacement approach. So, the original gene value is not needed.
                            values_filtered = self.get_valid_gene_constraint_values(range_min=range_min,
                                                                                    range_max=range_max,
                                                                                    gene_value=None,
                                                                                    gene_idx=gene_idx,
                                                                                    mutation_by_replacement=True,
                                                                                    solution=solution,
                                                                                    sample_size=self.sample_size)
                            if values_filtered is None:
                                if not self.suppress_warnings:
                                    warnings.warn(f"No value satisfied the constraint for the gene at index {gene_idx} with value {solution[gene_idx]} while creating the initial population.")
                            else:
                                self.population[sol_idx, gene_idx] = random.choice(values_filtered)
                        elif len(filtered_values) == 1:
                            # The value already satisfied the gene constraint.
                            pass
                        else:
                            # Error by the user's defined gene constraint callable.
                            raise Exception(f"It is expected to receive a list/numpy.ndarray from the gene_constraint callable that is either empty or has a single value equal, but received a list/numpy.ndarray of length {len(filtered_values)}.")

        # 4) Solve duplicate genes.
        if allow_duplicate_genes == False:
            for solution_idx in range(self.population.shape[0]):
                if self.gene_space is None:
                    self.population[solution_idx], _, _ = self.solve_duplicate_genes_randomly(solution=self.population[solution_idx],
                                                                                              min_val=self.init_range_low,
                                                                                              max_val=self.init_range_high,
                                                                                              gene_type=gene_type,
                                                                                              mutation_by_replacement=True,
                                                                                              sample_size=self.sample_size)
                else:
                    self.population[solution_idx], _, _ = self.solve_duplicate_genes_by_space(solution=self.population[solution_idx].copy(),
                                                                                              gene_type=self.gene_type,
                                                                                              mutation_by_replacement=True,
                                                                                              sample_size=self.sample_size,
                                                                                              build_initial_pop=True)

        # Change the data type and round all genes within the initial population.
        self.population = self.change_population_dtype_and_round(self.population)

        # Keeping the initial population in the initial_population attribute.
        self.initial_population = self.population.copy()

    def cal_pop_fitness(self):
        """
        Calculating the fitness values of batches of solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        try:
            if self.valid_parameters == False:
                raise Exception("ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

            # 'last_generation_parents_as_list' is the list version of 'self.last_generation_parents'
            # It is used to return the parent index using the 'in' membership operator of Python lists. This is much faster than using 'numpy.where()'.
            if self.last_generation_parents is not None:
                last_generation_parents_as_list = self.last_generation_parents.tolist()
            else:
                last_generation_parents_as_list = []

            # 'last_generation_elitism_as_list' is the list version of 'self.last_generation_elitism'
            # It is used to return the elitism index using the 'in' membership operator of Python lists. This is much faster than using 'numpy.where()'.
            if self.last_generation_elitism is not None:
                last_generation_elitism_as_list = self.last_generation_elitism.tolist()
            else:
                last_generation_elitism_as_list = []

            pop_fitness = ["undefined"] * len(self.population)
            if self.parallel_processing is None:
                # Calculating the fitness value of each solution in the current population.
                for sol_idx, sol in enumerate(self.population):
                    # Check if the `save_solutions` parameter is `True` and whether the solution already exists in the `solutions` list. If so, use its fitness rather than calculating it again.
                    # The functions numpy.any()/numpy.all()/numpy.where()/numpy.equal() are very slow.
                    # So, list membership operator 'in' is used to check if the solution exists in the 'self.solutions' list.
                    # Make sure that both the solution and 'self.solutions' are of type 'list' not 'numpy.ndarray'.
                    # if (self.save_solutions) and (len(self.solutions) > 0) and (numpy.any(numpy.all(self.solutions == numpy.array(sol), axis=1)))
                    # if (self.save_solutions) and (len(self.solutions) > 0) and (numpy.any(numpy.all(numpy.equal(self.solutions, numpy.array(sol)), axis=1)))

                    # Make sure self.best_solutions is a list of lists before proceeding.
                    # Because the second condition expects that best_solutions is a list of lists.
                    if type(self.best_solutions) is numpy.ndarray:
                        self.best_solutions = self.best_solutions.tolist()

                    if (self.save_solutions) and (len(self.solutions) > 0) and (list(sol) in self.solutions):
                        solution_idx = self.solutions.index(list(sol))
                        fitness = self.solutions_fitness[solution_idx]
                    elif (self.save_best_solutions) and (len(self.best_solutions) > 0) and (list(sol) in self.best_solutions):
                        solution_idx = self.best_solutions.index(list(sol))
                        fitness = self.best_solutions_fitness[solution_idx]
                    elif (self.keep_elitism > 0) and (self.last_generation_elitism is not None) and (len(self.last_generation_elitism) > 0) and (list(sol) in last_generation_elitism_as_list):
                        # Return the index of the elitism from the elitism array 'self.last_generation_elitism'.
                        # This is not its index within the population. It is just its index in the 'self.last_generation_elitism' array.
                        elitism_idx = last_generation_elitism_as_list.index(list(sol))
                        # Use the returned elitism index to return its index in the last population.
                        elitism_idx = self.last_generation_elitism_indices[elitism_idx]
                        # Use the elitism's index to return its pre-calculated fitness value.
                        fitness = self.previous_generation_fitness[elitism_idx]
                    # If the solutions are not saved (i.e. `save_solutions=False`), check if this solution is a parent from the previous generation and its fitness value is already calculated. If so, use the fitness value instead of calling the fitness function.
                    # We cannot use the `numpy.where()` function directly because it does not support the `axis` parameter. This is why the `numpy.all()` function is used to match the solutions on axis=1.
                    # elif (self.last_generation_parents is not None) and len(numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0] > 0):
                    elif ((self.keep_parents == -1) or (self.keep_parents > 0)) and (self.last_generation_parents is not None) and (len(self.last_generation_parents) > 0) and (list(sol) in last_generation_parents_as_list):
                        # Index of the parent in the 'self.last_generation_parents' array.
                        # This is not its index within the population. It is just its index in the 'self.last_generation_parents' array.
                        # parent_idx = numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0][0]
                        parent_idx = last_generation_parents_as_list.index(list(sol))
                        # Use the returned parent index to return its index in the last population.
                        parent_idx = self.last_generation_parents_indices[parent_idx]
                        # Use the parent's index to return its pre-calculated fitness value.
                        fitness = self.previous_generation_fitness[parent_idx]
                    else:
                        # Check if batch processing is used. If not, then calculate this missing fitness value.
                        if self.fitness_batch_size in [1, None]:
                            fitness = self.fitness_func(self, sol, sol_idx)
                            if type(fitness) in self.supported_int_float_types:
                                # The fitness function returns a single numeric value.
                                # This is a single-objective optimization problem.
                                pass
                            elif type(fitness) in [list, tuple, numpy.ndarray]:
                                # The fitness function returns a list/tuple/numpy.ndarray.
                                # This is a multi-objective optimization problem.
                                pass
                            else:
                                raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value {fitness} of type {type(fitness)} found.")
                        else:
                            # Reaching this point means that batch processing is in effect to calculate the fitness values.
                            # Do not continue the loop as no fitness is calculated. The fitness will be calculated later in batch mode.
                            continue

                    # This is only executed if the fitness value was already calculated.
                    pop_fitness[sol_idx] = fitness

                if self.fitness_batch_size not in [1, None]:
                    # Reaching this block means that batch fitness calculation is used.

                    # Indices of the solutions to calculate their fitness.
                    solutions_indices = [idx for idx, fit in enumerate(pop_fitness) if type(fit) is str and fit == "undefined"]
                    # Number of batches.
                    num_batches = int(numpy.ceil(len(solutions_indices) / self.fitness_batch_size))
                    # For each batch, get its indices and call the fitness function.
                    for batch_idx in range(num_batches):
                        batch_first_index = batch_idx * self.fitness_batch_size
                        batch_last_index = (batch_idx + 1) * self.fitness_batch_size
                        batch_indices = solutions_indices[batch_first_index:batch_last_index]
                        batch_solutions = self.population[batch_indices, :]

                        batch_fitness = self.fitness_func(
                            self, batch_solutions, batch_indices)
                        if type(batch_fitness) not in [list, tuple, numpy.ndarray]:
                            raise TypeError(f"Expected to receive a list, tuple, or numpy.ndarray from the fitness function but the value ({batch_fitness}) of type {type(batch_fitness)}.")
                        elif len(numpy.array(batch_fitness)) != len(batch_indices):
                            raise ValueError(f"There is a mismatch between the number of solutions passed to the fitness function ({len(batch_indices)}) and the number of fitness values returned ({len(batch_fitness)}). They must match.")

                        for index, fitness in zip(batch_indices, batch_fitness):
                            if type(fitness) in self.supported_int_float_types:
                                # The fitness function returns a single numeric value.
                                # This is a single-objective optimization problem.
                                pop_fitness[index] = fitness
                            elif type(fitness) in [list, tuple, numpy.ndarray]:
                                # The fitness function returns a list/tuple/numpy.ndarray.
                                # This is a multi-objective optimization problem.
                                pop_fitness[index] = fitness
                            else:
                                raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value {fitness} of type {type(fitness)} found.")
            else:
                # Calculating the fitness value of each solution in the current population.
                for sol_idx, sol in enumerate(self.population):
                    # Check if the `save_solutions` parameter is `True` and whether the solution already exists in the `solutions` list. If so, use its fitness rather than calculating it again.
                    # The functions numpy.any()/numpy.all()/numpy.where()/numpy.equal() are very slow.
                    # So, list membership operator 'in' is used to check if the solution exists in the 'self.solutions' list.
                    # Make sure that both the solution and 'self.solutions' are of type 'list' not 'numpy.ndarray'.
                    if (self.save_solutions) and (len(self.solutions) > 0) and (list(sol) in self.solutions):
                        solution_idx = self.solutions.index(list(sol))
                        fitness = self.solutions_fitness[solution_idx]
                        pop_fitness[sol_idx] = fitness
                    elif (self.keep_elitism > 0) and (self.last_generation_elitism is not None) and (len(self.last_generation_elitism) > 0) and (list(sol) in last_generation_elitism_as_list):
                        # Return the index of the elitism from the elitism array 'self.last_generation_elitism'.
                        # This is not its index within the population. It is just its index in the 'self.last_generation_elitism' array.
                        elitism_idx = last_generation_elitism_as_list.index(
                            list(sol))
                        # Use the returned elitism index to return its index in the last population.
                        elitism_idx = self.last_generation_elitism_indices[elitism_idx]
                        # Use the elitism's index to return its pre-calculated fitness value.
                        fitness = self.previous_generation_fitness[elitism_idx]

                        pop_fitness[sol_idx] = fitness
                    # If the solutions are not saved (i.e. `save_solutions=False`), check if this solution is a parent from the previous generation and its fitness value is already calculated. If so, use the fitness value instead of calling the fitness function.
                    # We cannot use the `numpy.where()` function directly because it does not support the `axis` parameter. This is why the `numpy.all()` function is used to match the solutions on axis=1.
                    # elif (self.last_generation_parents is not None) and len(numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0] > 0):
                    elif ((self.keep_parents == -1) or (self.keep_parents > 0)) and (self.last_generation_parents is not None) and (len(self.last_generation_parents) > 0) and (list(sol) in last_generation_parents_as_list):
                        # Index of the parent in the 'self.last_generation_parents' array.
                        # This is not its index within the population. It is just its index in the 'self.last_generation_parents' array.
                        # parent_idx = numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0][0]
                        parent_idx = last_generation_parents_as_list.index(
                            list(sol))
                        # Use the returned parent index to return its index in the last population.
                        parent_idx = self.last_generation_parents_indices[parent_idx]
                        # Use the parent's index to return its pre-calculated fitness value.
                        fitness = self.previous_generation_fitness[parent_idx]

                        pop_fitness[sol_idx] = fitness

                # Decide which class to use based on whether the user selected "process" or "thread"
                if self.parallel_processing[0] == "process":
                    ExecutorClass = concurrent.futures.ProcessPoolExecutor
                else:
                    ExecutorClass = concurrent.futures.ThreadPoolExecutor

                # We can use a with statement to ensure threads are cleaned up promptly (https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example)
                with ExecutorClass(max_workers=self.parallel_processing[1]) as executor:
                    solutions_to_submit_indices = []
                    solutions_to_submit = []
                    for sol_idx, sol in enumerate(self.population):
                        # The "undefined" value means that the fitness of this solution must be calculated.
                        if type(pop_fitness[sol_idx]) is str:
                            if pop_fitness[sol_idx] == "undefined":
                                solutions_to_submit.append(sol.copy())
                                solutions_to_submit_indices.append(sol_idx)
                        elif type(pop_fitness[sol_idx]) in [list, tuple, numpy.ndarray]:
                            # This is a multi-objective problem. The fitness is already calculated. Nothing to do.
                            pass

                    # Check if batch processing is used. If not, then calculate the fitness value for individual solutions.
                    if self.fitness_batch_size in [1, None]:
                        for index, fitness in zip(solutions_to_submit_indices, executor.map(self.fitness_func, [self]*len(solutions_to_submit_indices), solutions_to_submit, solutions_to_submit_indices)):
                            if type(fitness) in self.supported_int_float_types:
                                # The fitness function returns a single numeric value.
                                # This is a single-objective optimization problem.
                                pop_fitness[index] = fitness
                            elif type(fitness) in [list, tuple, numpy.ndarray]:
                                # The fitness function returns a list/tuple/numpy.ndarray.
                                # This is a multi-objective optimization problem.
                                pop_fitness[index] = fitness
                            else:
                                raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value {fitness} of type {type(fitness)} found.")
                    else:
                        # Reaching this block means that batch processing is used. The fitness values are calculated in batches.

                        # Number of batches.
                        num_batches = int(numpy.ceil(len(solutions_to_submit_indices) / self.fitness_batch_size))
                        # Each element of the `batches_solutions` list represents the solutions in one batch.
                        batches_solutions = []
                        # Each element of the `batches_indices` list represents the solutions' indices in one batch.
                        batches_indices = []
                        # For each batch, get its indices and call the fitness function.
                        for batch_idx in range(num_batches):
                            batch_first_index = batch_idx * self.fitness_batch_size
                            batch_last_index = (batch_idx + 1) * self.fitness_batch_size
                            batch_indices = solutions_to_submit_indices[batch_first_index:batch_last_index]
                            batch_solutions = self.population[batch_indices, :]

                            batches_solutions.append(batch_solutions)
                            batches_indices.append(batch_indices)

                        for batch_indices, batch_fitness in zip(batches_indices, executor.map(self.fitness_func, [self]*len(solutions_to_submit_indices), batches_solutions, batches_indices)):
                            if type(batch_fitness) not in [list, tuple, numpy.ndarray]:
                                raise TypeError(f"Expected to receive a list, tuple, or numpy.ndarray from the fitness function but the value ({batch_fitness}) of type {type(batch_fitness)}.")
                            elif len(numpy.array(batch_fitness)) != len(batch_indices):
                                raise ValueError(f"There is a mismatch between the number of solutions passed to the fitness function ({len(batch_indices)}) and the number of fitness values returned ({len(batch_fitness)}). They must match.")

                            for index, fitness in zip(batch_indices, batch_fitness):
                                if type(fitness) in self.supported_int_float_types:
                                    # The fitness function returns a single numeric value.
                                    # This is a single-objective optimization problem.
                                    pop_fitness[index] = fitness
                                elif type(fitness) in [list, tuple, numpy.ndarray]:
                                    # The fitness function returns a list/tuple/numpy.ndarray.
                                    # This is a multi-objective optimization problem.
                                    pop_fitness[index] = fitness
                                else:
                                    raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value {fitness} of type {type(fitness)} found.")

            pop_fitness = numpy.array(pop_fitness)
        except Exception as ex:
            self.logger.exception(ex)
            # sys.exit(-1)
            raise ex
        return pop_fitness

    def run(self):
        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
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

            stop_run = False

            # To continue from where we stopped, the first generation index should start from the value of the 'self.generations_completed' parameter.
            if self.generations_completed != 0 and type(self.generations_completed) in self.supported_int_types:
                # If the 'self.generations_completed' parameter is not '0', then this means we continue execution.
                generation_first_idx = self.generations_completed
                generation_last_idx = self.num_generations + self.generations_completed
            else:
                # If the 'self.generations_completed' parameter is '0', then stat from scratch.
                generation_first_idx = 0
                generation_last_idx = self.num_generations

            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            # Know whether the problem is SOO or MOO.
            if type(self.last_generation_fitness[0]) in self.supported_int_float_types:
                # Single-objective problem.
                # If the problem is SOO, the parent selection type cannot be nsga2 or tournament_nsga2.
                if self.parent_selection_type in ['nsga2', 'tournament_nsga2']:
                    raise TypeError(f"Incorrect parent selection type. The fitness function returned a single numeric fitness value which means the problem is single-objective. But the parent selection type {self.parent_selection_type} is used which only works for multi-objective optimization problems.")
            elif type(self.last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                # Multi-objective problem.
                pass                

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

                # Note: Any code that has loop-dependant statements (e.g. continue, break, etc.) must be kept inside the loop of the 'run()' method. It can be moved to another method to clean the run() method.
                # If the on_generation attribute is not None, then cal the callback function after the generation.
                if not (self.on_generation is None):
                    r = self.on_generation(self)
                    if type(r) is str and r.lower() == "stop":
                        # Before aborting the loop, save the fitness value of the best solution.
                        # _, best_solution_fitness, _ = self.best_solution()
                        self.best_solutions_fitness.append(best_solution_fitness)
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
                                        raise ValueError(f"When the the 'reach' keyword is used with the 'stop_criteria' parameter for solving a multi-objective problem, then the number of numeric values following the keyword can be:\n1) A single numeric value to be used across all the objective functions.\n2) A number of numeric values equal to the number of objective functions.\nBut the value {criterion} found with {len(criterion)-1} numeric values which is not equal to the number of objective functions {len(self.last_generation_fitness[0])}.")

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
            self.best_solutions = numpy.array(self.best_solutions)

            # Update previous_generation_fitness because it is used to get the fitness of the parents.
            self.previous_generation_fitness = self.last_generation_fitness.copy()

            # Converting the 'solutions' list into a NumPy array.
            # self.solutions = numpy.array(self.solutions)
        except Exception as ex:
            self.logger.exception(ex)
            # sys.exit(-1)
            raise ex

    def run_loop_head(self, best_solution_fitness):
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
        This method must be only called from inside the run() method. It is not meant for use by the user.
        Generally, any method with a name starting with 'run_' is meant to be only called by PyGAD from inside the 'run()' method.

        The objective of the 'run_select_parents()' method is to select the parents and call the callable on_parents() if defined.
        It does not return any variables. However, it changes these 2 attributes of the pygad.GA class instances:
            1) last_generation_parents: A NumPy array of the selected parents.
            2) last_generation_parents_indices: A 1D NumPy array of the indices of the selected parents.

        Parameters
        ----------
        call_on_parents : bool, optional
            If True, then the callable 'on_parents()' is called. The default is True.
    
        Returns
        -------
        None.
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
                            on_parents_selected_parents = numpy.array(on_parents_selected_parents)
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

    def run_crossover(self):
        """
        This method must be only called from inside the run() method. It is not meant for use by the user.
        Generally, any method with a name starting with 'run_' is meant to be only called by PyGAD from inside the 'run()' method.

        The objective of the 'run_crossover()' method is to apply crossover and call the callable on_crossover() if defined.
        It does not return any variables. However, it changes these 2 attributes of the pygad.GA class instances:
            1) last_generation_offspring_crossover: A NumPy array of the selected offspring.
            2) last_generation_elitism: A NumPy array of the current generation elitism. Applicable only if the 'keep_elitism' parameter > 0.

        Returns
        -------
        None.
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

        # PyGAD 2.18.2 // The on_crossover() callback function is called even if crossover_type is None.
        if not (self.on_crossover is None):
            on_crossover_output = self.on_crossover(self, 
                                                    self.last_generation_offspring_crossover)
            if on_crossover_output is None:
                pass
            else:
                if type(on_crossover_output) in [tuple, list, numpy.ndarray]:
                    on_crossover_output = numpy.array(on_crossover_output)
                    if on_crossover_output.shape == self.last_generation_offspring_crossover.shape:
                        self.last_generation_offspring_crossover = on_crossover_output
                    else:
                        raise ValueError(f"Size mismatch between the output of on_crossover() {on_crossover_output.shape} and the expected crossover output {self.last_generation_offspring_crossover.shape}.")
                else:
                    raise ValueError(f"The output of on_crossover() is expected to be tuple/list/numpy.ndarray but {type(on_crossover_output)} found.")

    def run_mutation(self):
        """
        This method must be only called from inside the run() method. It is not meant for use by the user.
        Generally, any method with a name starting with 'run_' is meant to be only called by PyGAD from inside the 'run()' method.

        The objective of the 'run_mutation()' method is to apply mutation and call the callable on_mutation() if defined.
        It does not return any variables. However, it changes this attribute of the pygad.GA class instances:
            1) last_generation_offspring_mutation: A NumPy array of the mutated offspring.

        Returns
        -------
        None.
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

        # PyGAD 2.18.2 // The on_mutation() callback function is called even if mutation_type is None.
        if not (self.on_mutation is None):
            on_mutation_output = self.on_mutation(self, 
                                                  self.last_generation_offspring_mutation)

            if on_mutation_output is None:
                pass
            else:
                if type(on_mutation_output) in [tuple, list, numpy.ndarray]:
                    on_mutation_output = numpy.array(on_mutation_output)
                    if on_mutation_output.shape == self.last_generation_offspring_mutation.shape:
                        self.last_generation_offspring_mutation = on_mutation_output
                    else:
                        raise ValueError(f"Size mismatch between the output of on_mutation() {on_mutation_output.shape} and the expected mutation output {self.last_generation_offspring_mutation.shape}.")
                else:
                    raise ValueError(f"The output of on_mutation() is expected to be tuple/list/numpy.ndarray but {type(on_mutation_output)} found.")

    def run_update_population(self):
        """
        This method must be only called from inside the run() method. It is not meant for use by the user.
        Generally, any method with a name starting with 'run_' is meant to be only called by PyGAD from inside the 'run()' method.

        The objective of the 'run_update_population()' method is to update the 'population' attribute after completing the processes of crossover and mutation.
        It does not return any variables. However, it changes this attribute of the pygad.GA class instances:
            1) population: A NumPy array of the population of solutions/chromosomes.

        Returns
        -------
        None.
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
        Returns information about the best solution found by the genetic algorithm.
        Accepts the following parameters:
            pop_fitness: An optional parameter holding the fitness values of the solutions in the latest population. If passed, then it save time calculating the fitness. If None, then the 'cal_pop_fitness()' method is called to calculate the fitness of the latest population.
        The following are returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """

        try:
            if pop_fitness is None:
                # If the 'pop_fitness' parameter is not passed, then we have to call the 'cal_pop_fitness()' method to calculate the fitness of all solutions in the lastest population.
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
