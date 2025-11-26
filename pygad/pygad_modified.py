import numpy
import random
import logging
import pygad.utils
import pygad.helper
import pygad.visualize

class GA(pygad.utils.parent_selection.ParentSelection,
         pygad.utils.crossover.Crossover,
         pygad.utils.mutation.Mutation,
         pygad.utils.nsga2.NSGA2,
         pygad.helper.unique.Unique,
         pygad.helper.misc.Helper,
         pygad.visualize.plot.Plot):

    supported_int_types = [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                           numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                           object]
    supported_float_types = [float, numpy.float16, numpy.float32, numpy.float64,
                             object]
    supported_int_float_types = supported_int_types + supported_float_types

    def __init__(self,
                 num_generations,
                 num_parents_mating,
                 fitness_func,
                 fitness_batch_size=None,
                 initial_population=None,
                 sol_per_pop=None,
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 gene_type=float,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 keep_elitism=1,
                 K_tournament=3,
                 crossover_type="single_point",
                 crossover_probability=None,
                 mutation_type="random",
                 mutation_probability=None,
                 mutation_by_replacement=False,
                 mutation_percent_genes='default',
                 mutation_num_genes=None,
                 random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0,
                 gene_space=None,
                 gene_constraint=None,
                 sample_size=100,
                 allow_duplicate_genes=True,
                 on_start=None,
                 on_fitness=None,
                 on_parents=None,
                 on_crossover=None,
                 on_mutation=None,
                 on_generation=None,
                 on_stop=None,
                 save_best_solutions=False,
                 save_solutions=False,
                 suppress_warnings=False,
                 stop_criteria=None,
                 parallel_processing=None,
                 random_seed=None,
                 logger=None):
        """
        The constructor of the GA class accepts all parameters required to create an instance of the GA class. It validates such parameters.

        num_generations: Number of generations.
        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        fitness_func: Accepts a function/method and returns the fitness value of the solution. In PyGAD 2.20.0, a third parameter is passed referring to the 'pygad.GA' instance. If method, then it must accept 4 parameters where the fourth one refers to the method's object.
        fitness_batch_size: Added in PyGAD 2.19.0. Supports calculating the fitness in batches. If the value is 1 or None, then the fitness function is called for each individual solution. If given another value X where X is neither 1 nor None (e.g. X=3), then the fitness function is called once for each X (3) solutions.

        initial_population: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to None which means no initial population is specified by the user. In this case, PyGAD creates an initial population using the 'sol_per_pop' and 'num_genes' parameters. An exception is raised if the 'initial_population' is None while any of the 2 parameters ('sol_per_pop' or 'num_genes') is also None.
        sol_per_pop: Number of solutions in the population. 
        num_genes: Number of parameters in the function.

        init_range_low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        init_range_high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        # It is OK to set the value of the 2 parameters ('init_range_low' and 'init_range_high') to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).

        gene_type: The type of the gene. It is assigned to any of these types (int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, float, numpy.float16, numpy.float32, numpy.float64) and forces all the genes to be of that type.

        parent_selection_type: Type of parent selection.
        keep_parents: If 0, this means no parent in the current population will be used in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. Some parent selection operators such as rank selection, favor population diversity and therefore keeping the parents in the next generation can be beneficial. However, some other parent selection operators, such as roulette wheel selection (RWS), have higher selection pressure and keeping more than one parent in the next generation can seriously harm population diversity. This parameter have an effect only when the keep_elitism parameter is 0. Thanks to Prof. Fernando Jiménez (http://webs.um.es/fernan) for editing this sentence.
        K_tournament: When the value of 'parent_selection_type' is 'tournament', the 'K_tournament' parameter specifies the number of solutions from which a parent is selected randomly.

        keep_elitism: Added in PyGAD 2.18.0. It can take the value 0 or a positive integer that satisfies (0 <= keep_elitism <= sol_per_pop). It defaults to 1 which means only the best solution in the current generation is kept in the next generation. If assigned 0, this means it has no effect. If assigned a positive integer K, then the best K solutions are kept in the next generation. It cannot be assigned a value greater than the value assigned to the sol_per_pop parameter. If this parameter has a value different from 0, then the keep_parents parameter will have no effect.

        crossover_type: Type of the crossover operator. If  crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        crossover_probability: The probability of selecting a solution for the crossover operation. If the solution probability is <= crossover_probability, the solution is selected. The value must be between 0 and 1 inclusive.

        mutation_type: Type of the mutation operator. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        mutation_probability: The probability of selecting a gene for the mutation operation. If the gene probability is <= mutation_probability, the gene is selected. It accepts either a single value for fixed mutation or a list/tuple/numpy.ndarray of 2 values for adaptive mutation. The values must be between 0 and 1 inclusive. If specified, then no need for the 2 parameters mutation_percent_genes and mutation_num_genes.

        mutation_by_replacement: An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, setting mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene.

        mutation_percent_genes: Percentage of genes to mutate which defaults to the string 'default' which means 10%. This parameter has no action if any of the 2 parameters mutation_probability or mutation_num_genes exists.
        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes. This parameter has no action if the mutation_probability parameter exists.
        random_mutation_min_val: The minimum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to -1.0.
        random_mutation_max_val: The maximum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to 1.0.

        gene_space: It accepts a list of all possible values of the gene. This list is used in the mutation step. Should be used only if the gene space is a set of discrete values. No need for the 2 parameters (random_mutation_min_val and random_mutation_max_val) if the parameter gene_space exists. Added in PyGAD 2.5.0. In PyGAD 2.11.0, the gene_space can be assigned a dict.

        gene_constraint: It accepts a list of constraints for the genes. Each constraint is a Python function. Added in PyGAD 3.5.0.
        sample_size: To select a gene value that respects a constraint, this variable defines the size of the sample from which a value is selected randomly. Useful if either allow_duplicate_genes or gene_constraint is used. Added in PyGAD 3.5.0.

        on_start: Accepts a function/method to be called only once before the genetic algorithm starts its evolution. If functioned, then it must accept a single parameter representing the instance of the genetic algorithm. If method, then it must accept 2 parameters where the second one refers to the method's object. Added in PyGAD 2.6.0.
        on_fitness: Accepts a function/method to be called after calculating the fitness values of all solutions in the population. If functioned, then it must accept 2 parameters: 1) a list of all solutions' fitness values 2) the instance of the genetic algorithm. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_parents: Accepts a function/method to be called after selecting the parents that mates. If functioned, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the selected parents. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_crossover: Accepts a function/method to be called each time the crossover operation is applied. If functioned, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring generated using crossover. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_mutation: Accepts a function/method to be called each time the mutation operation is applied. If functioned, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring after applying the mutation. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_generation: Accepts a function/method to be called after each generation. If functioned, then it must accept a single parameter representing the instance of the genetic algorithm. If the function returned "stop", then the run() method stops without completing the other generations. If method, then it must accept 2 parameters where the second one refers to the method's object. Added in PyGAD 2.6.0.
        on_stop: Accepts a function/method to be called only once exactly before the genetic algorithm stops or when it completes all the generations. If functioned, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one is a list of fitness values of the last population's solutions. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.

        save_best_solutions: Added in PyGAD 2.9.0 and its type is bool. If True, then the best solution in each generation is saved into the 'best_solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations or the number of genes is large.
        save_solutions: Added in PyGAD 2.15.0 and its type is bool. If True, then all solutions in each generation are saved into the 'solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.

        suppress_warnings: Added in PyGAD 2.10.0 and its type is bool. If True, then no warning messages will be displayed. It defaults to False.

        allow_duplicate_genes: Added in PyGAD 2.13.0. If True, then a solution/chromosome may have duplicate gene values. If False, then each gene will have a unique value in its solution.

        stop_criteria: Added in PyGAD 2.15.0. It is assigned to some criteria to stop the evolution if at least one criterion holds.

        parallel_processing: Added in PyGAD 2.17.0. Defaults to `None` which means no parallel processing is used. If a positive integer is assigned, it specifies the number of threads to be used. If a list or a tuple of exactly 2 elements is assigned, then: 1) The first element can be either "process" or "thread" to specify whether processes or threads are used, respectively. 2) The second element can be: 1) A positive integer to select the maximum number of processes or threads to be used. 2) 0 to indicate that parallel processing is not used. This is identical to setting 'parallel_processing=None'. 3) None to use the default value as calculated by the concurrent.futures module.

        random_seed: Added in PyGAD 2.18.0. It defines the random seed to be used by the random function generators (we use random functions in the NumPy and random modules). This helps to reproduce the same results by setting the same random seed.

        logger: Added in PyGAD 2.20.0. It accepts a logger object of the 'logging.Logger' class to log the messages. If no logger is passed, then a default logger is created to log/print the messages to the console exactly like using the 'print()' function.
        """
        try:
            # If no logger is passed, then create a logger that logs only the messages to the console.
            if logger is None:
                # Create a logger named with the module name.
                logger = logging.getLogger(__name__)
                # Set the logger log level to 'DEBUG' to log all kinds of messages.
                logger.setLevel(logging.DEBUG)

                # Clear any attached handlers to the logger from the previous runs.
                # If the handlers are not cleared, then the new handler will be appended to the list of handlers.
                # This makes the single log message be repeated according to the length of the list of handlers.
                logger.handlers.clear()

                # Create the handlers.
                stream_handler = logging.StreamHandler()
                # Set the handler log level to 'DEBUG' to log all kinds of messages received from the logger.
                stream_handler.setLevel(logging.DEBUG)

                # Create the formatter that just includes the log message.
                formatter = logging.Formatter('%(message)s')

                # Add the formatter to the handler.
                stream_handler.setFormatter(formatter)

                # Add the handler to the logger.
                logger.addHandler(stream_handler)
            else:
                # Validate that the passed logger is of type 'logging.Logger'.
                if isinstance(logger, logging.Logger):
                    pass
                else:
                    raise TypeError(f"The expected type of the 'logger' parameter is 'logging.Logger' but {type(logger)} found.")

            # Create the 'self.logger' attribute to hold the logger.
            # Instead of using 'print()', use 'self.logger.info()'
            self.logger = logger

            self.random_seed = random_seed
            if random_seed is None:
                pass
            else:
                numpy.random.seed(self.random_seed)
                random.seed(self.random_seed)

            # If suppress_warnings is bool and its value is False, then print warning messages.
            if type(suppress_warnings) is bool:
                self.suppress_warnings = suppress_warnings
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'suppress_warnings' parameter is bool but {type(suppress_warnings)} found.")

            # Validating mutation_by_replacement
            if not (type(mutation_by_replacement) is bool):
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'mutation_by_replacement' parameter is bool but {type(mutation_by_replacement)} found.")

            self.mutation_by_replacement = mutation_by_replacement

            # Validate the sample_size parameter.
            if type(sample_size) in GA.supported_int_types:
                if sample_size > 0:
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The value of the sample_size parameter must be > 0 but the value ({sample_size}) found.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'sample_size' parameter is integer but {type(sample_size)} found.")
            self.sample_size = sample_size

            # Environment state machine implementation
            self.ENV_FAST = "ENV_FAST"
            self.ENV_STABLE = "ENV_STABLE"
            self.ENV_DIVERSE = "ENV_DIVERSE"
            self.environment_cycle = [self.ENV_FAST, self.ENV_STABLE, self.ENV_DIVERSE]
            self.current_environment = self.ENV_STABLE  # Default to stable environment
            
            # Environment-specific parameters
            self.env_params = {
                self.ENV_FAST: {
                    "parent_selection_type": "tournament",
                    "mutation_probability": 0.1,
                    "crossover_probability": 0.9
                },
                self.ENV_STABLE: {
                    "parent_selection_type": "sss",
                    "mutation_probability": 0.05,
                    "crossover_probability": 0.7
                },
                self.ENV_DIVERSE: {
                    "parent_selection_type": "rank",
                    "mutation_probability": 0.15,
                    "crossover_probability": 0.5
                }
            }
            
            # Save original parameters to restore after environment changes
            self.original_parent_selection_type = parent_selection_type
            self.original_mutation_probability = mutation_probability
            self.original_crossover_probability = crossover_probability

            # ... existing code ...
            # Set valid_parameters to True since we've completed all parameter validations
            self.valid_parameters = True
            self.best_solutions = []  # Holds the best solution in each generation.
            self.best_solutions_fitness = []  # Holds the fitness of the best solution in each generation.
            self.solutions = []  # Holds the solutions in each generation.
            self.solutions_fitness = []  # Holds the fitness of the solutions in each generation.
            self.last_generation_fitness = None  # Holds the fitness values of all solutions in the last generation.
            # Callback attributes
            self.on_start = None
            self.on_fitness = None
            self.on_parents = None
            self.on_crossover = None
            self.on_mutation = None
            self.on_generation = None
            self.on_stop = None
            # Generation tracking attributes
            self.generations_completed = 0
            self.run_completed = False
            # Parameters of the genetic algorithm
            self.num_generations = abs(num_generations)
            self.parent_selection_type = parent_selection_type
            self.sol_per_pop = sol_per_pop
            self.num_genes = num_genes
            # Parameters of the mutation operation
            self.mutation_percent_genes = mutation_percent_genes
            self.mutation_num_genes = mutation_num_genes
            # Population initialization
            self.pop_size = (self.sol_per_pop, self.num_genes)
            self.population = numpy.empty(shape=self.pop_size, dtype=object)
            # Fitness function
            self.fitness_func = fitness_func
            # Gene space
            self.gene_space = gene_space
            # Initial population generation
            if self.gene_space is None:
                # Create the initial population randomly
                for sol_idx in range(self.sol_per_pop):
                    for gene_idx in range(self.num_genes):
                        # Default range [-1, 1] for random generation
                        self.population[sol_idx, gene_idx] = numpy.random.uniform(-1, 1)

        except Exception as e:
            self.logger.error(f"Error in the __init__ method: {e}")
            raise

    def get_environment_state(self, generation):
        """
        Returns the current environment state based on the generation number.
        Environment changes every 10 generations.
        """
        env_index = (generation // 10) % len(self.environment_cycle)
        return self.environment_cycle[env_index]

    def update_environment(self, generation):
        """
        Updates the environment parameters based on the current generation.
        """
        new_environment = self.get_environment_state(generation)
        if new_environment != self.current_environment:
            self.current_environment = new_environment
            self.logger.info(f"[Generation {generation}] Environment switched to: {self.current_environment}")
            
            # Update parameters based on the new environment
            env_params = self.env_params[self.current_environment]
            self.parent_selection_type = env_params["parent_selection_type"]
            self.mutation_probability = env_params["mutation_probability"]
            self.crossover_probability = env_params["crossover_probability"]

    def calculate_diversity_score(self, solution, population):
        """
        Calculates the diversity score of a solution as the average distance from all other solutions in the population.
        """
        if len(population) <= 1:
            return 0.0
        
        distances = []
        for other_solution in population:
            if numpy.array_equal(solution, other_solution):
                continue
            distance = numpy.linalg.norm(solution - other_solution)
            distances.append(distance)
        
        return numpy.mean(distances)

    def cal_pop_fitness(self):
        """
        Calculates the fitness of the entire population, including time cost and diversity score.
        """
        import time
        
        fitness = []
        for solution in self.population:
            start_time = time.time()
            fitness_score = self.fitness_func(self, solution)
            time_cost = (time.time() - start_time) * 1000  # Convert to milliseconds
            diversity_score = self.calculate_diversity_score(solution, self.population)
            
            fitness.append([fitness_score, time_cost, diversity_score])
        
        return numpy.array(fitness)

    def run(self):
        """
        Runs the genetic algorithm with environment state machine and multi-objective optimization.
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
            if self.generations_completed != 0 and type(self.generations_completed) in GA.supported_int_types:
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
            if type(self.last_generation_fitness[0]) in GA.supported_int_float_types:
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

                # Update environment every 10 generations
                if generation % 10 == 0:
                    self.update_environment(generation)

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

                # Calculate and print Pareto front
                if type(self.last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                    pareto_fronts, _ = self.non_dominated_sorting(self.last_generation_fitness)
                    pareto_front = pareto_fronts[0]  # Get the first Pareto front
                    
                    # Format the Pareto front data
                    pareto_front_data = []
                    for solution in pareto_front:
                        fitness_score = solution[1][0]
                        time_cost = solution[1][1]
                        diversity_score = solution[1][2]
                        pareto_front_data.append({
                            "fitness": fitness_score,
                            "time": time_cost,
                            "diversity": diversity_score
                        })
                    
                    # Print the Pareto front data
                    self.logger.info(f"Pareto Front for Generation {generation}:")
                    self.logger.info(f"{pareto_front_data}")

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
                            if type(self.last_generation_fitness[0]) in GA.supported_int_float_types:
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
                                        raise ValueError(f"The number of values does not equal the number of objectives.")

                                    if max(self.last_generation_fitness[:, obj_idx]) >= reach_fitness_value:
                                        pass
                                    else:
                                        stop_run = False
                                        break
                        elif criterion[0] == "saturate":
                            criterion[1] = int(criterion[1])
                            if self.generations_completed >= criterion[1]:
                                # Single-objective problem.
                                if type(self.last_generation_fitness[0]) in GA.supported_int_float_types:
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

        except Exception as e:
            self.logger.error(f"Error in the run() method: {e}")
            raise

    # ... existing code ...