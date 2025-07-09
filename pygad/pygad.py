import numpy
import random
import cloudpickle
import warnings
import concurrent.futures
import inspect
import logging
from pygad import utils
from pygad import helper
from pygad import visualize

# Extend all the classes so that they can be referenced by just the `self` object of the `pygad.GA` class.
class GA(utils.parent_selection.ParentSelection,
         utils.crossover.Crossover,
         utils.mutation.Mutation,
         utils.nsga2.NSGA2,
         helper.unique.Unique,
         helper.misc.Helper,
         visualize.plot.Plot):

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

        mutation_percent_genes: Percentage of genes to mutate which defaults to the string 'default' which means 10%. This parameter has no action if any of the 2 parameters mutation_probability or mutation_num_genes exist.
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
                raise TypeError(f"The type of the sample_size parameter must be integer but the value ({sample_size}) of type ({type(sample_size)}) found.")

            self.sample_size = sample_size

            # Validate allow_duplicate_genes
            if not (type(allow_duplicate_genes) is bool):
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'allow_duplicate_genes' parameter is bool but {type(allow_duplicate_genes)} found.")

            self.allow_duplicate_genes = allow_duplicate_genes

            # Validate gene_space
            self.gene_space_nested = False
            if type(gene_space) is type(None):
                pass
            elif type(gene_space) is range:
                if len(gene_space) == 0:
                    self.valid_parameters = False
                    raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
            elif type(gene_space) in [list, numpy.ndarray]:
                if len(gene_space) == 0:
                    self.valid_parameters = False
                    raise ValueError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
                else:
                    for index, el in enumerate(gene_space):
                        if type(el) in [numpy.ndarray, list, tuple, range]:
                            if len(el) == 0:
                                self.valid_parameters = False
                                raise ValueError(f"The element indexed {index} of 'gene_space' with type {type(el)} cannot be empty (i.e. its length must be >= 0).")
                            else:
                                for val in el:
                                    if not (type(val) in [type(None)] + GA.supported_int_float_types):
                                        raise TypeError(f"All values in the sublists inside the 'gene_space' attribute must be numeric of type int/float/None but ({val}) of type {type(val)} found.")
                            self.gene_space_nested = True
                        elif type(el) == type(None):
                            pass
                            # self.gene_space_nested = True
                        elif type(el) is dict:
                            if len(el.items()) == 2:
                                if ('low' in el.keys()) and ('high' in el.keys()):
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {el.keys()}")
                            elif len(el.items()) == 3:
                                if ('low' in el.keys()) and ('high' in el.keys()) and ('step' in el.keys()):
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {el.keys()}")
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items but ({len(el.items())}) items found.")
                            self.gene_space_nested = True
                        elif not (type(el) in GA.supported_int_float_types):
                            self.valid_parameters = False
                            raise TypeError(f"Unexpected type {type(el)} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.")

            elif type(gene_space) is dict:
                if len(gene_space.items()) == 2:
                    if ('low' in gene_space.keys()) and ('high' in gene_space.keys()):
                        pass
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space.keys()}")
                elif len(gene_space.items()) == 3:
                    if ('low' in gene_space.keys()) and ('high' in gene_space.keys()) and ('step' in gene_space.keys()):
                        pass
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space.keys()}")
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When the 'gene_space' parameter is of type dict, then it must have only 2 items but ({len(gene_space.items())}) items found.")

            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of 'gene_space' is list, range, or numpy.ndarray but {type(gene_space)} found.")

            self.gene_space = gene_space

            # Validate init_range_low and init_range_high
            if type(init_range_low) in GA.supported_int_float_types:
                if type(init_range_high) in GA.supported_int_float_types:
                    if init_range_low == init_range_high:
                        if not self.suppress_warnings:
                            warnings.warn("The values of the 2 parameters 'init_range_low' and 'init_range_high' are equal and this might return the same value for some genes in the initial population.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Type mismatch between the 2 parameters 'init_range_low' {type(init_range_low)} and 'init_range_high' {type(init_range_high)}.")
            elif type(init_range_low) in [list, tuple, numpy.ndarray]:
                # Get the number of genes before validating the num_genes parameter.
                if num_genes is None:
                    if initial_population is None:
                        self.valid_parameters = False
                        raise TypeError("When the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.")
                    elif not len(init_range_low) == len(initial_population[0]):
                        self.valid_parameters = False
                        raise ValueError(f"The length of the 'init_range_low' parameter is {len(init_range_low)} which is different from the number of genes {len(initial_population[0])}.")
                elif not len(init_range_low) == num_genes:
                    self.valid_parameters = False
                    raise ValueError(f"The length of the 'init_range_low' parameter is {len(init_range_low)} which is different from the number of genes {num_genes}.")

                if type(init_range_high) in [list, tuple, numpy.ndarray]:
                    if len(init_range_low) == len(init_range_high):
                        pass
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"Size mismatch between the 2 parameters 'init_range_low' {len(init_range_low)} and 'init_range_high' {len(init_range_high)}.")

                    # Validate the values in init_range_low
                    for val in init_range_low:
                        if type(val) in GA.supported_int_float_types:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'init_range_low' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")

                    # Validate the values in init_range_high
                    for val in init_range_high:
                        if type(val) in GA.supported_int_float_types:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'init_range_high' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Type mismatch between the 2 parameters 'init_range_low' {type(init_range_low)} and 'init_range_high' {type(init_range_high)}. Both of them can be either numeric or iterable (list/tuple/numpy.ndarray).")
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'init_range_low' parameter is numeric or list/tuple/numpy.ndarray but {type(init_range_low)} found.")

            self.init_range_low = init_range_low
            self.init_range_high = init_range_high

            # Validate gene_type
            if gene_type in GA.supported_int_float_types:
                self.gene_type = [gene_type, None]
                self.gene_type_single = True
            # A single data type of float with precision.
            elif len(gene_type) == 2 and gene_type[0] in GA.supported_float_types and (type(gene_type[1]) in GA.supported_int_types or gene_type[1] is None):
                self.gene_type = gene_type
                self.gene_type_single = True
            # A single data type of integer with precision None ([int, None]).
            elif len(gene_type) == 2 and gene_type[0] in GA.supported_int_types and gene_type[1] is None:
                self.gene_type = gene_type
                self.gene_type_single = True
            # Raise an exception for a single data type of int with integer precision.
            elif len(gene_type) == 2 and gene_type[0] in GA.supported_int_types and (type(gene_type[1]) in GA.supported_int_types or gene_type[1] is None):
                self.gene_type_single = False
                raise ValueError(f"Integers cannot have precision. Please use the integer data type directly instead of {gene_type}.")
            elif type(gene_type) in [list, tuple, numpy.ndarray]:
                # Get the number of genes before validating the num_genes parameter.
                if num_genes is None:
                    if initial_population is None:
                        self.valid_parameters = False
                        raise TypeError("When the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.")
                    elif not len(gene_type) == len(initial_population[0]):
                        self.valid_parameters = False
                        raise ValueError(f"When the parameter 'gene_type' is nested, then it can be either [float, int<precision>] or with length equal to the number of genes parameter. Instead, value {gene_type} with len(gene_type) ({len(gene_type)}) != number of genes ({len(initial_population[0])}) found.")
                elif not len(gene_type) == num_genes:
                    self.valid_parameters = False
                    raise ValueError(f"When the parameter 'gene_type' is nested, then it can be either [float, int<precision>] or with length equal to the value passed to the 'num_genes' parameter. Instead, value {gene_type} with len(gene_type) ({len(gene_type)}) != len(num_genes) ({num_genes}) found.")
                for gene_type_idx, gene_type_val in enumerate(gene_type):
                    if gene_type_val in GA.supported_int_float_types:
                        # If the gene type is float and no precision is passed or an integer, set its precision to None.
                        gene_type[gene_type_idx] = [gene_type_val, None]
                    elif type(gene_type_val) in [list, tuple, numpy.ndarray]:
                        # A float type is expected in a list/tuple/numpy.ndarray of length 2.
                        if len(gene_type_val) == 2:
                            if gene_type_val[0] in GA.supported_float_types:
                                if type(gene_type_val[1]) in GA.supported_int_types:
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise TypeError(f"In the 'gene_type' parameter, the precision for float gene data types must be an integer but the element {gene_type_val} at index {gene_type_idx} has a precision of {gene_type_val[1]} with type {gene_type_val[0]}.")
                            elif gene_type_val[0] in GA.supported_int_types:
                                if gene_type_val[1] is None:
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise TypeError(f"In the 'gene_type' parameter, either do not set a precision for integer data types or set it to None. But the element {gene_type_val} at index {gene_type_idx} has a precision of {gene_type_val[1]} with type {gene_type_val[0]}.")
                            else:
                                self.valid_parameters = False
                                raise TypeError(
                                    f"In the 'gene_type' parameter, a precision is expected only for float gene data types but the element {gene_type_val} found at index {gene_type_idx}.\nNote that the data type must be at index 0 of the item followed by precision at index 1.")
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"In the 'gene_type' parameter, a precision is specified in a list/tuple/numpy.ndarray of length 2 but value ({gene_type_val}) of type {type(gene_type_val)} with length {len(gene_type_val)} found at index {gene_type_idx}.")
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"When a list/tuple/numpy.ndarray is assigned to the 'gene_type' parameter, then its elements must be of integer, floating-point, list, tuple, or numpy.ndarray data types but the value ({gene_type_val}) of type {type(gene_type_val)} found at index {gene_type_idx}.")
                self.gene_type = gene_type
                self.gene_type_single = False
            else:
                self.valid_parameters = False
                raise ValueError(f"The value passed to the 'gene_type' parameter must be either a single integer, floating-point, list, tuple, or numpy.ndarray but ({gene_type}) of type {type(gene_type)} found.")

            # Call the unpack_gene_space() method in the pygad.helper.unique.Unique class.
            self.gene_space_unpacked = self.unpack_gene_space(range_min=self.init_range_low,
                                                              range_max=self.init_range_high)

            # Build the initial population
            if initial_population is None:
                if (sol_per_pop is None) or (num_genes is None):
                    self.valid_parameters = False
                    raise TypeError("Error creating the initial population:\n\nWhen the parameter 'initial_population' is None, then the 2 parameters 'sol_per_pop' and 'num_genes' cannot be None too.\nThere are 2 options to prepare the initial population:\n1) Assigning the initial population to the 'initial_population' parameter. In this case, the values of the 2 parameters sol_per_pop and num_genes will be deduced.\n2) Assign integer values to the 'sol_per_pop' and 'num_genes' parameters so that PyGAD can create the initial population automatically.")
                elif (type(sol_per_pop) is int) and (type(num_genes) is int):
                    # Validating the number of solutions in the population (sol_per_pop)
                    if sol_per_pop <= 0:
                        self.valid_parameters = False
                        raise ValueError(f"The number of solutions in the population (sol_per_pop) must be > 0 but ({sol_per_pop}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n")
                    # Validating the number of gene.
                    if (num_genes <= 0):
                        self.valid_parameters = False
                        raise ValueError(f"The number of genes cannot be <= 0 but ({num_genes}) found.\n")
                    # When initial_population=None and the 2 parameters sol_per_pop and num_genes have valid integer values, then the initial population is created.
                    # Inside the initialize_population() method, the initial_population attribute is assigned to keep the initial population accessible.
                    self.num_genes = num_genes  # Number of genes in the solution.

                    # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
                    if self.gene_space_nested:
                        if len(gene_space) != self.num_genes:
                            self.valid_parameters = False
                            raise ValueError(f"When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len(gene_space)}) != num_genes ({self.num_genes})")

                    # Number of solutions in the population.
                    self.sol_per_pop = sol_per_pop
                    self.initialize_population(allow_duplicate_genes=allow_duplicate_genes,
                                               gene_type=self.gene_type,
                                               gene_constraint=gene_constraint)
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The expected type of both the sol_per_pop and num_genes parameters is int but {type(sol_per_pop)} and {type(num_genes)} found.")
            elif not type(initial_population) in [list, tuple, numpy.ndarray]:
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the 'initial_population' parameter is expected to by of type list, tuple, or ndarray but {type(initial_population)} found.")
            elif numpy.array(initial_population).ndim != 2:
                self.valid_parameters = False
                raise ValueError(f"A 2D list is expected to the initial_population parameter but a ({numpy.array(initial_population).ndim}-D) list found.")
            else:
                # Validate the type of each value in the 'initial_population' parameter.
                for row_idx in range(len(initial_population)):
                    for col_idx in range(len(initial_population[0])):
                        if type(initial_population[row_idx][col_idx]) in GA.supported_int_float_types:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"The values in the initial population can be integers or floats but the value ({initial_population[row_idx][col_idx]}) of type {type(initial_population[row_idx][col_idx])} found.")

                # Change the data type and round all genes within the initial population.
                self.initial_population = self.change_population_dtype_and_round(initial_population)

                # Check if duplicates are allowed. If not, then solve any existing duplicates in the passed initial population.
                if self.allow_duplicate_genes == False:
                    for initial_solution_idx, initial_solution in enumerate(self.initial_population):
                        if self.gene_space is None:
                            self.initial_population[initial_solution_idx], _, _ = self.solve_duplicate_genes_randomly(solution=initial_solution,
                                                                                                                      min_val=self.init_range_low,
                                                                                                                      max_val=self.init_range_high,
                                                                                                                      mutation_by_replacement=True,
                                                                                                                      gene_type=self.gene_type,
                                                                                                                      sample_size=self.sample_size)
                        else:
                            self.initial_population[initial_solution_idx], _, _ = self.solve_duplicate_genes_by_space(solution=initial_solution,
                                                                                                                      gene_type=self.gene_type,
                                                                                                                      sample_size=self.sample_size,
                                                                                                                      mutation_by_replacement=True,
                                                                                                                      build_initial_pop=True)

                # A NumPy array holding the initial population.
                self.population = self.initial_population.copy()
                # Number of genes in the solution.
                self.num_genes = self.initial_population.shape[1]
                # Number of solutions in the population.
                self.sol_per_pop = self.initial_population.shape[0]
                # The population size.
                self.pop_size = (self.sol_per_pop, self.num_genes)

            # Change the data type and round all genes within the initial population.
            self.initial_population = self.change_population_dtype_and_round(self.initial_population)
            self.population = self.initial_population.copy()

            # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
            if self.gene_space_nested:
                if len(gene_space) != self.num_genes:
                    self.valid_parameters = False
                    raise ValueError(f"When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len(gene_space)}) != num_genes ({self.num_genes})")

            # Validate random_mutation_min_val and random_mutation_max_val
            if type(random_mutation_min_val) in GA.supported_int_float_types:
                if type(random_mutation_max_val) in GA.supported_int_float_types:
                    if random_mutation_min_val == random_mutation_max_val:
                        if not self.suppress_warnings:
                            warnings.warn("The values of the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val' are equal and this might cause a fixed mutation to some genes.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Type mismatch between the 2 parameters 'random_mutation_min_val' {type(random_mutation_min_val)} and 'random_mutation_max_val' {type(random_mutation_max_val)}.")
            elif type(random_mutation_min_val) in [list, tuple, numpy.ndarray]:
                if len(random_mutation_min_val) == self.num_genes:
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The length of the 'random_mutation_min_val' parameter is {len(random_mutation_min_val)} which is different from the number of genes {self.num_genes}.")
                if type(random_mutation_max_val) in [list, tuple, numpy.ndarray]:
                    if len(random_mutation_min_val) == len(random_mutation_max_val):
                        pass
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"Size mismatch between the 2 parameters 'random_mutation_min_val' {len(random_mutation_min_val)} and 'random_mutation_max_val' {len(random_mutation_max_val)}.")

                    # Validate the values in random_mutation_min_val
                    for val in random_mutation_min_val:
                        if type(val) in GA.supported_int_float_types:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'random_mutation_min_val' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")

                    # Validate the values in random_mutation_max_val
                    for val in random_mutation_max_val:
                        if type(val) in GA.supported_int_float_types:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"When an iterable (list/tuple/numpy.ndarray) is assigned to the 'random_mutation_max_val' parameter, its elements must be numeric but the value {val} of type {type(val)} found.")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Type mismatch between the 2 parameters 'random_mutation_min_val' {type(random_mutation_min_val)} and 'random_mutation_max_val' {type(random_mutation_max_val)}.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'random_mutation_min_val' parameter is numeric or list/tuple/numpy.ndarray but {type(random_mutation_min_val)} found.")

            self.random_mutation_min_val = random_mutation_min_val
            self.random_mutation_max_val = random_mutation_max_val

            # Validate that gene_constraint is a list or tuple and every element inside it is either None or callable.
            if gene_constraint:
                if type(gene_constraint) in [list, tuple]:
                    if len(gene_constraint) == self.num_genes:
                        for constraint_idx, item in enumerate(gene_constraint):
                            # Check whether the element is None or a callable.
                            if item is None:
                                pass
                            elif item and callable(item):
                                if item.__code__.co_argcount == 2:
                                    # Every callable is valid if it receives 2 arguments.
                                    # The 2 arguments: 1) solution 2) A list or numpy.ndarray of values to check if they meet the constraint.
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise ValueError(f"Every callable inside the gene_constraint parameter must accept 2 arguments representing 1) The solution/chromosome where the gene exists 2) A list of NumPy array of values to check if they meet the constraint. But the callable at index {constraint_idx} named '{item.__code__.co_name}' accepts {item.__code__.co_argcount} argument(s).")
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"The expected type of an element in the 'gene_constraint' parameter is None or a callable (e.g. function). But {item} at index {constraint_idx} of type {type(item)} found.")
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The number of constrains ({len(gene_constraint)}) in the 'gene_constraint' parameter must be equal to the number of genes ({self.num_genes}).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The expected type of the 'gene_constraint' parameter is either a list or tuple. But the value {gene_constraint} of type {type(gene_constraint)} found.")
            else:
                # gene_constraint is None and not used.
                pass

            self.gene_constraint = gene_constraint

            # Validating the number of parents to be selected for mating (num_parents_mating)
            if num_parents_mating <= 0:
                self.valid_parameters = False
                raise ValueError(f"The number of parents mating (num_parents_mating) parameter must be > 0 but ({num_parents_mating}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n")

            # Validating the number of parents to be selected for mating: num_parents_mating
            if num_parents_mating > self.sol_per_pop:
                self.valid_parameters = False
                raise ValueError(f"The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({self.sol_per_pop}) (i.e., num_parents_mating must always be <= sol_per_pop).\n")

            self.num_parents_mating = num_parents_mating

            # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
            # Validating the crossover type: crossover_type
            if crossover_type is None:
                self.crossover = None
            elif inspect.ismethod(crossover_type):
                # Check if the crossover_type is a method that accepts 4 parameters.
                if crossover_type.__code__.co_argcount == 4:
                    # The crossover method assigned to the crossover_type parameter is validated.
                    self.crossover = crossover_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'crossover_type' is assigned to a method, then this crossover method must accept 4 parameters:\n1) Expected to be the 'self' object.\n2) The selected parents.\n3) The size of the offspring to be produced.\n4) The instance from the pygad.GA class.\n\nThe passed crossover method named '{crossover_type.__code__.co_name}' accepts {crossover_type.__code__.co_argcount} parameter(s).")
            elif callable(crossover_type):
                # Check if the crossover_type is a function that accepts 2 parameters.
                if crossover_type.__code__.co_argcount == 3:
                    # The crossover function assigned to the crossover_type parameter is validated.
                    self.crossover = crossover_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'crossover_type' is assigned to a function, then this crossover function must accept 3 parameters:\n1) The selected parents.\n2) The size of the offspring to be produced.3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed crossover function named '{crossover_type.__code__.co_name}' accepts {crossover_type.__code__.co_argcount} parameter(s).")
            elif not (type(crossover_type) is str):
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'crossover_type' parameter is either callable or str but {type(crossover_type)} found.")
            else:  # type crossover_type is str
                crossover_type = crossover_type.lower()
                if crossover_type == "single_point":
                    self.crossover = self.single_point_crossover
                elif crossover_type == "two_points":
                    self.crossover = self.two_points_crossover
                elif crossover_type == "uniform":
                    self.crossover = self.uniform_crossover
                elif crossover_type == "scattered":
                    self.crossover = self.scattered_crossover
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) parameter does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover)\n-scattered (for scattered crossover).\n")

            self.crossover_type = crossover_type

            # Calculate the value of crossover_probability
            if crossover_probability is None:
                self.crossover_probability = None
            elif type(crossover_probability) in GA.supported_int_float_types:
                if 0 <= crossover_probability <= 1:
                    self.crossover_probability = crossover_probability
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The value assigned to the 'crossover_probability' parameter must be between 0 and 1 inclusive but ({crossover_probability}) found.")
            else:
                self.valid_parameters = False
                raise TypeError(f"Unexpected type for the 'crossover_probability' parameter. Float is expected but ({crossover_probability}) of type {type(crossover_probability)} found.")

            # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
            # Validating the mutation type: mutation_type
            # "adaptive" mutation is supported starting from PyGAD 2.10.0
            if mutation_type is None:
                self.mutation = None
            elif inspect.ismethod(mutation_type):
                # Check if the mutation_type is a method that accepts 3 parameters.
                if (mutation_type.__code__.co_argcount == 3):
                    # The mutation method assigned to the mutation_type parameter is validated.
                    self.mutation = mutation_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'mutation_type' is assigned to a method, then it must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The offspring to be mutated.\n3) The instance from the pygad.GA class.\n\nThe passed mutation method named '{mutation_type.__code__.co_name}' accepts {mutation_type.__code__.co_argcount} parameter(s).")
            elif callable(mutation_type):
                # Check if the mutation_type is a function that accepts 2 parameters.
                if (mutation_type.__code__.co_argcount == 2):
                    # The mutation function assigned to the mutation_type parameter is validated.
                    self.mutation = mutation_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'mutation_type' is assigned to a function, then this mutation function must accept 2 parameters:\n1) The offspring to be mutated.\n2) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed mutation function named '{mutation_type.__code__.co_name}' accepts {mutation_type.__code__.co_argcount} parameter(s).")
            elif not (type(mutation_type) is str):
                self.valid_parameters = False
                raise TypeError(f"The expected type of the 'mutation_type' parameter is either callable or str but {type(mutation_type)} found.")
            else:  # type mutation_type is str
                mutation_type = mutation_type.lower()
                if mutation_type == "random":
                    self.mutation = self.random_mutation
                elif mutation_type == "swap":
                    self.mutation = self.swap_mutation
                elif mutation_type == "scramble":
                    self.mutation = self.scramble_mutation
                elif mutation_type == "inversion":
                    self.mutation = self.inversion_mutation
                elif mutation_type == "adaptive":
                    self.mutation = self.adaptive_mutation
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Undefined mutation type. \nThe assigned string value to the 'mutation_type' parameter ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation)\n-adaptive (for adaptive mutation).\n")

            self.mutation_type = mutation_type

            # Calculate the value of mutation_probability
            if not (self.mutation_type is None):
                if mutation_probability is None:
                    self.mutation_probability = None
                elif mutation_type != "adaptive":
                    # Mutation probability is fixed not adaptive.
                    if type(mutation_probability) in GA.supported_int_float_types:
                        if 0 <= mutation_probability <= 1:
                            self.mutation_probability = mutation_probability
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"The value assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({mutation_probability}) found.")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected type for the 'mutation_probability' parameter. A numeric value is expected but ({mutation_probability}) of type {type(mutation_probability)} found.")
                else:
                    # Mutation probability is adaptive not fixed.
                    if type(mutation_probability) in [list, tuple, numpy.ndarray]:
                        if len(mutation_probability) == 2:
                            for el in mutation_probability:
                                if type(el) in GA.supported_int_float_types:
                                    if 0 <= el <= 1:
                                        pass
                                    else:
                                        self.valid_parameters = False
                                        raise ValueError(f"The values assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({el}) found.")
                                else:
                                    self.valid_parameters = False
                                    raise TypeError(f"Unexpected type for a value assigned to the 'mutation_probability' parameter. A numeric value is expected but ({el}) of type {type(el)} found.")
                            if mutation_probability[0] < mutation_probability[1]:
                                if not self.suppress_warnings:
                                    warnings.warn(f"The first element in the 'mutation_probability' parameter is {mutation_probability[0]} which is smaller than the second element {mutation_probability[1]}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.")
                            self.mutation_probability = mutation_probability
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"When mutation_type='adaptive', then the 'mutation_probability' parameter must have only 2 elements but ({len(mutation_probability)}) element(s) found.")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected type for the 'mutation_probability' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_probability}) of type {type(mutation_probability)} found.")
            else:
                pass

            # Calculate the value of mutation_num_genes
            if not (self.mutation_type is None):
                if mutation_num_genes is None:
                    # The mutation_num_genes parameter does not exist. Checking whether adaptive mutation is used.
                    if mutation_type != "adaptive":
                        # The percent of genes to mutate is fixed not adaptive.
                        if mutation_percent_genes == 'default'.lower():
                            mutation_percent_genes = 10
                            # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                            mutation_num_genes = numpy.uint32(
                                (mutation_percent_genes*self.num_genes)/100)
                            # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                            if mutation_num_genes == 0:
                                if self.mutation_probability is None:
                                    if not self.suppress_warnings:
                                        warnings.warn(
                                            f"The percentage of genes to mutate (mutation_percent_genes={mutation_percent_genes}) resulted in selecting ({mutation_num_genes}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                                mutation_num_genes = 1

                        elif type(mutation_percent_genes) in GA.supported_int_float_types:
                            if mutation_percent_genes <= 0 or mutation_percent_genes > 100:
                                self.valid_parameters = False
                                raise ValueError(f"The percentage of selected genes for mutation (mutation_percent_genes) must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n")
                            else:
                                # If mutation_percent_genes equals the string "default", then it is replaced by the numeric value 10.
                                if mutation_percent_genes == 'default'.lower():
                                    mutation_percent_genes = 10

                                # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                                mutation_num_genes = numpy.uint32(
                                    (mutation_percent_genes*self.num_genes)/100)
                                # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                                if mutation_num_genes == 0:
                                    if self.mutation_probability is None:
                                        if not self.suppress_warnings:
                                            warnings.warn(f"The percentage of genes to mutate (mutation_percent_genes={mutation_percent_genes}) resulted in selecting ({mutation_num_genes}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                                    mutation_num_genes = 1
                        else:
                            self.valid_parameters = False
                            raise TypeError(f"Unexpected value or type of the 'mutation_percent_genes' parameter. It only accepts the string 'default' or a numeric value but ({mutation_percent_genes}) of type {type(mutation_percent_genes)} found.")
                    else:
                        # The percent of genes to mutate is adaptive not fixed.
                        if type(mutation_percent_genes) in [list, tuple, numpy.ndarray]:
                            if len(mutation_percent_genes) == 2:
                                mutation_num_genes = numpy.zeros_like(
                                    mutation_percent_genes, dtype=numpy.uint32)
                                for idx, el in enumerate(mutation_percent_genes):
                                    if type(el) in GA.supported_int_float_types:
                                        if el <= 0 or el > 100:
                                            self.valid_parameters = False
                                            raise ValueError(f"The values assigned to the 'mutation_percent_genes' must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n")
                                    else:
                                        self.valid_parameters = False
                                        raise TypeError(f"Unexpected type for a value assigned to the 'mutation_percent_genes' parameter. An integer value is expected but ({el}) of type {type(el)} found.")
                                    # At this point of the loop, the current value assigned to the parameter 'mutation_percent_genes' is validated.
                                    # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                                    mutation_num_genes[idx] = numpy.uint32(
                                        (mutation_percent_genes[idx]*self.num_genes)/100)
                                    # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                                    if mutation_num_genes[idx] == 0:
                                        if not self.suppress_warnings:
                                            warnings.warn(f"The percentage of genes to mutate ({mutation_percent_genes[idx]}) resulted in selecting ({mutation_num_genes[idx]}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.")
                                        mutation_num_genes[idx] = 1
                                if mutation_percent_genes[0] < mutation_percent_genes[1]:
                                    if not self.suppress_warnings:
                                        warnings.warn(f"The first element in the 'mutation_percent_genes' parameter is ({mutation_percent_genes[0]}) which is smaller than the second element ({mutation_percent_genes[1]}).\nThis means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions.\nPlease make the first element higher than the second element.")
                                # At this point outside the loop, all values of the parameter 'mutation_percent_genes' are validated. Everything is OK.
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"When mutation_type='adaptive', then the 'mutation_percent_genes' parameter must have only 2 elements but ({len(mutation_percent_genes)}) element(s) found.")
                        else:
                            if self.mutation_probability is None:
                                self.valid_parameters = False
                                raise TypeError(f"Unexpected type of the 'mutation_percent_genes' parameter. When mutation_type='adaptive', then the 'mutation_percent_genes' parameter should exist and assigned a list/tuple/numpy.ndarray with 2 values but ({mutation_percent_genes}) found.")
                # The mutation_num_genes parameter exists. Checking whether adaptive mutation is used.
                elif mutation_type != "adaptive":
                    # Number of genes to mutate is fixed not adaptive.
                    if type(mutation_num_genes) in GA.supported_int_types:
                        if mutation_num_genes <= 0:
                            self.valid_parameters = False
                            raise ValueError(f"The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but ({mutation_num_genes}) found. If you do not want to use mutation, please set mutation_type=None\n")
                        elif mutation_num_genes > self.num_genes:
                            self.valid_parameters = False
                            raise ValueError(f"The number of selected genes for mutation (mutation_num_genes), which is ({mutation_num_genes}), cannot be greater than the number of genes ({self.num_genes}).\n")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"The 'mutation_num_genes' parameter is expected to be a positive integer but the value ({mutation_num_genes}) of type {type(mutation_num_genes)} found.\n")
                else:
                    # Number of genes to mutate is adaptive not fixed.
                    if type(mutation_num_genes) in [list, tuple, numpy.ndarray]:
                        if len(mutation_num_genes) == 2:
                            for el in mutation_num_genes:
                                if type(el) in GA.supported_int_types:
                                    if el <= 0:
                                        self.valid_parameters = False
                                        raise ValueError(f"The values assigned to the 'mutation_num_genes' cannot be <= 0 but ({el}) found. If you do not want to use mutation, please set mutation_type=None\n")
                                    elif el > self.num_genes:
                                        self.valid_parameters = False
                                        raise ValueError(f"The values assigned to the 'mutation_num_genes' cannot be greater than the number of genes ({self.num_genes}) but ({el}) found.\n")
                                else:
                                    self.valid_parameters = False
                                    raise TypeError(f"Unexpected type for a value assigned to the 'mutation_num_genes' parameter. An integer value is expected but ({el}) of type {type(el)} found.")
                                # At this point of the loop, the current value assigned to the parameter 'mutation_num_genes' is validated.
                            if mutation_num_genes[0] < mutation_num_genes[1]:
                                if not self.suppress_warnings:
                                    warnings.warn(f"The first element in the 'mutation_num_genes' parameter is {mutation_num_genes[0]} which is smaller than the second element {mutation_num_genes[1]}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high quality solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.")
                            # At this point outside the loop, all values of the parameter 'mutation_num_genes' are validated. Everything is OK.
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"When mutation_type='adaptive', then the 'mutation_num_genes' parameter must have only 2 elements but ({len(mutation_num_genes)}) element(s) found.")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"Unexpected type for the 'mutation_num_genes' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_num_genes}) of type {type(mutation_num_genes)} found.")
            else:
                pass

            # Validating mutation_by_replacement and mutation_type
            if self.mutation_type != "random" and self.mutation_by_replacement:
                if not self.suppress_warnings:
                    warnings.warn(f"The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but ({mutation_type}). Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.")

            # Check if crossover and mutation are both disabled.
            if (self.mutation_type is None) and (self.crossover_type is None):
                if not self.suppress_warnings:
                    warnings.warn("The 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

            # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
            # Validating the selected type of parent selection: parent_selection_type
            if inspect.ismethod(parent_selection_type):
                # Check if the parent_selection_type is a method that accepts 4 parameters.
                if parent_selection_type.__code__.co_argcount == 4:
                    # population: Added in PyGAD 2.16.0. It should use only to support custom parent selection functions. Otherwise, it should be left to None to retrieve the population by self.population.
                    # The parent selection method assigned to the parent_selection_type parameter is validated.
                    self.select_parents = parent_selection_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'parent_selection_type' is assigned to a method, then it must accept 4 parameters:\n1) Expected to be the 'self' object.\n2) The fitness values of the current population.\n3) The number of parents needed.\n4) The instance from the pygad.GA class.\n\nThe passed parent selection method named '{parent_selection_type.__code__.co_name}' accepts {parent_selection_type.__code__.co_argcount} parameter(s).")
            elif callable(parent_selection_type):
                # Check if the parent_selection_type is a function that accepts 3 parameters.
                if parent_selection_type.__code__.co_argcount == 3:
                    # population: Added in PyGAD 2.16.0. It should use only to support custom parent selection functions. Otherwise, it should be left to None to retrieve the population by self.population.
                    # The parent selection function assigned to the parent_selection_type parameter is validated.
                    self.select_parents = parent_selection_type
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When 'parent_selection_type' is assigned to a user-defined function, then this parent selection function must accept 3 parameters:\n1) The fitness values of the current population.\n2) The number of parents needed.\n3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed parent selection function named '{parent_selection_type.__code__.co_name}' accepts {parent_selection_type.__code__.co_argcount} parameter(s).")
            elif not (type(parent_selection_type) is str):
                self.valid_parameters = False
                
                raise TypeError(f"The expected type of the 'parent_selection_type' parameter is either callable or str but {type(parent_selection_type)} found.")
            else:
                parent_selection_type = parent_selection_type.lower()
                if parent_selection_type == "sss":
                    self.select_parents = self.steady_state_selection
                elif parent_selection_type == "rws":
                    self.select_parents = self.roulette_wheel_selection
                elif parent_selection_type == "sus":
                    self.select_parents = self.stochastic_universal_selection
                elif parent_selection_type == "random":
                    self.select_parents = self.random_selection
                elif parent_selection_type == "tournament":
                    self.select_parents = self.tournament_selection
                elif parent_selection_type == "tournament_nsga2": # Supported in PyGAD >= 3.2
                    self.select_parents = self.tournament_selection_nsga2
                elif parent_selection_type == "nsga2": # Supported in PyGAD >= 3.2
                    self.select_parents = self.nsga2_selection
                elif parent_selection_type == "rank":
                    self.select_parents = self.rank_selection
                else:
                    self.valid_parameters = False
                    raise TypeError(f"Undefined parent selection type: {parent_selection_type}. \nThe assigned value to the 'parent_selection_type' parameter does not refer to one of the supported parent selection techniques which are: \n-sss (steady state selection)\n-rws (roulette wheel selection)\n-sus (stochastic universal selection)\n-rank (rank selection)\n-random (random selection)\n-tournament (tournament selection)\n-tournament_nsga2: (Tournament selection for NSGA-II)\n-nsga2: (NSGA-II parent selection).\n")

            # For tournament selection, validate the K value.
            if parent_selection_type == "tournament":
                if type(K_tournament) in GA.supported_int_types:
                    if K_tournament > self.sol_per_pop:
                        K_tournament = self.sol_per_pop
                        if not self.suppress_warnings:
                            warnings.warn(f"K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({self.sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n")
                    elif K_tournament <= 0:
                        self.valid_parameters = False
                        raise ValueError(f"K of the tournament selection cannot be <=0 but ({K_tournament}) found.\n")
                else:
                    self.valid_parameters = False
                    raise ValueError(f"The type of K of the tournament selection must be integer but the value ({K_tournament}) of type ({type(K_tournament)}) found.")

            self.K_tournament = K_tournament

            # Validating the number of parents to keep in the next population: keep_parents
            if not (type(keep_parents) in GA.supported_int_types):
                self.valid_parameters = False
                raise TypeError(f"Incorrect type of the value assigned to the keep_parents parameter. The value ({keep_parents}) of type {type(keep_parents)} found but an integer is expected.")
            elif keep_parents > self.sol_per_pop or keep_parents > self.num_parents_mating or keep_parents < -1:
                self.valid_parameters = False
                raise ValueError(f"Incorrect value to the keep_parents parameter: {keep_parents}. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.")

            self.keep_parents = keep_parents

            if parent_selection_type == "sss" and self.keep_parents == 0:
                if not self.suppress_warnings:
                    warnings.warn("The steady-state parent (sss) selection operator is used despite that no parents are kept in the next generation.")

            # Validating the number of elitism to keep in the next population: keep_elitism
            if not (type(keep_elitism) in GA.supported_int_types):
                self.valid_parameters = False
                raise TypeError(f"Incorrect type of the value assigned to the keep_elitism parameter. The value ({keep_elitism}) of type {type(keep_elitism)} found but an integer is expected.")
            elif keep_elitism > self.sol_per_pop or keep_elitism < 0:
                self.valid_parameters = False
                raise ValueError(f"Incorrect value to the keep_elitism parameter: {keep_elitism}. \nThe assigned value to the keep_elitism parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Greater than or equal to 0.")

            self.keep_elitism = keep_elitism

            # Validate keep_parents.
            if self.keep_elitism == 0:
                # Keep all parents in the next population.
                if self.keep_parents == -1:
                    self.num_offspring = self.sol_per_pop - self.num_parents_mating
                # Keep no parents in the next population.
                elif self.keep_parents == 0:
                    self.num_offspring = self.sol_per_pop
                # Keep the specified number of parents in the next population.
                elif self.keep_parents > 0:
                    self.num_offspring = self.sol_per_pop - self.keep_parents
            else:
                self.num_offspring = self.sol_per_pop - self.keep_elitism

            # Check if the fitness_func is a method.
            # In PyGAD 2.19.0, a method can be passed to the fitness function. If function is passed, then it accepts 2 parameters. If method, then it accepts 3 parameters.
            # In PyGAD 2.20.0, a new parameter is passed referring to the instance of the `pygad.GA` class. So, the function accepts 3 parameters and the method accepts 4 parameters.
            if inspect.ismethod(fitness_func):
                # If the fitness is calculated through a method, not a function, then there is a fourth 'self` parameters.
                if fitness_func.__code__.co_argcount == 4:
                    self.fitness_func = fitness_func
                else:
                    self.valid_parameters = False
                    raise ValueError(f"In PyGAD 2.20.0, if a method is used to calculate the fitness value, then it must accept 4 parameters\n1) Expected to be the 'self' object.\n2) The instance of the 'pygad.GA' class.\n3) A solution to calculate its fitness value.\n4) The solution's index within the population.\n\nThe passed fitness method named '{fitness_func.__code__.co_name}' accepts {fitness_func.__code__.co_argcount} parameter(s).")
            elif callable(fitness_func):
                # Check if the fitness function accepts 2 parameters.
                if fitness_func.__code__.co_argcount == 3:
                    self.fitness_func = fitness_func
                else:
                    self.valid_parameters = False
                    raise ValueError(f"In PyGAD 2.20.0, the fitness function must accept 3 parameters:\n1) The instance of the 'pygad.GA' class.\n2) A solution to calculate its fitness value.\n3) The solution's index within the population.\n\nThe passed fitness function named '{fitness_func.__code__.co_name}' accepts {fitness_func.__code__.co_argcount} parameter(s).")
            else:
                self.valid_parameters = False
                
                raise TypeError(f"The value assigned to the fitness_func parameter is expected to be of type function but {type(fitness_func)} found.")

            if fitness_batch_size is None:
                pass
            elif not (type(fitness_batch_size) in GA.supported_int_types):
                self.valid_parameters = False
                raise TypeError(f"The value assigned to the fitness_batch_size parameter is expected to be integer but the value ({fitness_batch_size}) of type {type(fitness_batch_size)} found.")
            elif fitness_batch_size <= 0 or fitness_batch_size > self.sol_per_pop:
                self.valid_parameters = False
                raise ValueError(f"The value assigned to the fitness_batch_size parameter must be:\n1) Greater than 0.\n2) Less than or equal to sol_per_pop ({self.sol_per_pop}).\nBut the value ({fitness_batch_size}) found.")

            self.fitness_batch_size = fitness_batch_size

            # Check if the on_start exists.
            if not (on_start is None):
                if inspect.ismethod(on_start):
                    # Check if the on_start method accepts 2 parameters.
                    if on_start.__code__.co_argcount == 2:
                        self.on_start = on_start
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_start parameter must accept only 2 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\nThe passed method named '{on_start.__code__.co_name}' accepts {on_start.__code__.co_argcount} parameter(s).")
                # Check if the on_start is a function.
                elif callable(on_start):
                    # Check if the on_start function accepts only a single parameter.
                    if on_start.__code__.co_argcount == 1:
                        self.on_start = on_start
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_start parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{on_start.__code__.co_name}' accepts {on_start.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    
                    raise TypeError(f"The value assigned to the on_start parameter is expected to be of type function but {type(on_start)} found.")
            else:
                self.on_start = None

            # Check if the on_fitness exists.
            if not (on_fitness is None):
                # Check if the on_fitness is a method.
                if inspect.ismethod(on_fitness):
                    # Check if the on_fitness method accepts 3 parameters.
                    if on_fitness.__code__.co_argcount == 3:
                        self.on_fitness = on_fitness
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_fitness parameter must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.3) The fitness values of all solutions.\nThe passed method named '{on_fitness.__code__.co_name}' accepts {on_fitness.__code__.co_argcount} parameter(s).")
                # Check if the on_fitness is a function.
                elif callable(on_fitness):
                    # Check if the on_fitness function accepts 2 parameters.
                    if on_fitness.__code__.co_argcount == 2:
                        self.on_fitness = on_fitness
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_fitness parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{on_fitness.__code__.co_name}' accepts {on_fitness.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the on_fitness parameter is expected to be of type function but {type(on_fitness)} found.")
            else:
                self.on_fitness = None

            # Check if the on_parents exists.
            if not (on_parents is None):
                # Check if the on_parents is a method.
                if inspect.ismethod(on_parents):
                    # Check if the on_parents method accepts 3 parameters.
                    if on_parents.__code__.co_argcount == 3:
                        self.on_parents = on_parents
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_parents parameter must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\n3) The fitness values of all solutions.\nThe passed method named '{on_parents.__code__.co_name}' accepts {on_parents.__code__.co_argcount} parameter(s).")
                # Check if the on_parents is a function.
                elif callable(on_parents):
                    # Check if the on_parents function accepts 2 parameters.
                    if on_parents.__code__.co_argcount == 2:
                        self.on_parents = on_parents
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_parents parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{on_parents.__code__.co_name}' accepts {on_parents.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the on_parents parameter is expected to be of type function but {type(on_parents)} found.")
            else:
                self.on_parents = None

            # Check if the on_crossover exists.
            if not (on_crossover is None):
                # Check if the on_crossover is a method.
                if inspect.ismethod(on_crossover):
                    # Check if the on_crossover method accepts 3 parameters.
                    if on_crossover.__code__.co_argcount == 3:
                        self.on_crossover = on_crossover
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_crossover parameter must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\n2) The offspring generated using crossover.\nThe passed method named '{on_crossover.__code__.co_name}' accepts {on_crossover.__code__.co_argcount} parameter(s).")
                # Check if the on_crossover is a function.
                elif callable(on_crossover):
                    # Check if the on_crossover function accepts 2 parameters.
                    if on_crossover.__code__.co_argcount == 2:
                        self.on_crossover = on_crossover
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_crossover parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring generated using crossover.\nThe passed function named '{on_crossover.__code__.co_name}' accepts {on_crossover.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the on_crossover parameter is expected to be of type function but {type(on_crossover)} found.")
            else:
                self.on_crossover = None

            # Check if the on_mutation exists.
            if not (on_mutation is None):
                # Check if the on_mutation is a method.
                if inspect.ismethod(on_mutation):
                    # Check if the on_mutation method accepts 3 parameters.
                    if on_mutation.__code__.co_argcount == 3:
                        self.on_mutation = on_mutation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_mutation parameter must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\n2) The offspring after applying the mutation operation.\nThe passed method named '{on_mutation.__code__.co_name}' accepts {on_mutation.__code__.co_argcount} parameter(s).")
                # Check if the on_mutation is a function.
                elif callable(on_mutation):
                    # Check if the on_mutation function accepts 2 parameters.
                    if on_mutation.__code__.co_argcount == 2:
                        self.on_mutation = on_mutation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_mutation parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring after applying the mutation operation.\nThe passed function named '{on_mutation.__code__.co_name}' accepts {on_mutation.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the on_mutation parameter is expected to be of type function but {type(on_mutation)} found.")
            else:
                self.on_mutation = None

            # Check if the on_generation exists.
            if not (on_generation is None):
                # Check if the on_generation is a method.
                if inspect.ismethod(on_generation):
                    # Check if the on_generation method accepts 2 parameters.
                    if on_generation.__code__.co_argcount == 2:
                        self.on_generation = on_generation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_generation parameter must accept 2 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\nThe passed method named '{on_generation.__code__.co_name}' accepts {on_generation.__code__.co_argcount} parameter(s).")
                # Check if the on_generation is a function.
                elif callable(on_generation):
                    # Check if the on_generation function accepts only a single parameter.
                    if on_generation.__code__.co_argcount == 1:
                        self.on_generation = on_generation
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{on_generation.__code__.co_name}' accepts {on_generation.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the on_generation parameter is expected to be of type function but {type(on_generation)} found.")
            else:
                self.on_generation = None

            # Check if the on_stop exists.
            if not (on_stop is None):
                # Check if the on_stop is a method.
                if inspect.ismethod(on_stop):
                    # Check if the on_stop method accepts 3 parameters.
                    if on_stop.__code__.co_argcount == 3:
                        self.on_stop = on_stop
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The method assigned to the on_stop parameter must accept 3 parameters:\n1) Expected to be the 'self' object.\n2) The instance of the genetic algorithm.\n2) A list of the fitness values of the solutions in the last population.\nThe passed method named '{on_stop.__code__.co_name}' accepts {on_stop.__code__.co_argcount} parameter(s).")
                # Check if the on_stop is a function.
                elif callable(on_stop):
                    # Check if the on_stop function accepts 2 parameters.
                    if on_stop.__code__.co_argcount == 2:
                        self.on_stop = on_stop
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The function assigned to the on_stop parameter must accept 2 parameters representing the instance of the genetic algorithm and a list of the fitness values of the solutions in the last population.\nThe passed function named '{on_stop.__code__.co_name}' accepts {on_stop.__code__.co_argcount} parameter(s).")
                else:
                    self.valid_parameters = False
                    raise TypeError(f"The value assigned to the 'on_stop' parameter is expected to be of type function but {type(on_stop)} found.")
            else:
                self.on_stop = None

            # Validate save_best_solutions
            if type(save_best_solutions) is bool:
                if save_best_solutions == True:
                    if not self.suppress_warnings:
                        warnings.warn("Use the 'save_best_solutions' parameter with caution as it may cause memory overflow when either the number of generations or number of genes is large.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value passed to the 'save_best_solutions' parameter must be of type bool but {type(save_best_solutions)} found.")

            # Validate save_solutions
            if type(save_solutions) is bool:
                if save_solutions == True:
                    if not self.suppress_warnings:
                        warnings.warn("Use the 'save_solutions' parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The value passed to the 'save_solutions' parameter must be of type bool but {type(save_solutions)} found.")

            def validate_multi_stop_criteria(self, stop_word, number):
                if stop_word == 'reach':
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"Passing multiple numbers following the keyword in the 'stop_criteria' parameter is expected only with the 'reach' keyword but the keyword ({stop_word}) found.")

                for idx, num in enumerate(number):
                    if num.replace(".", "").replace("-", "").isnumeric():
                        number[idx] = float(num)
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The value(s) following the stop word in the 'stop_criteria' parameter must be numeric but the value ({num}) of type {type(num)} found.")
                return number

            self.stop_criteria = []
            self.supported_stop_words = ["reach", "saturate"]
            if stop_criteria is None:
                # None: Stop after passing through all generations.
                self.stop_criteria = None
            elif type(stop_criteria) is str:
                # reach_{target_fitness}: Stop if the target fitness value is reached.
                # saturate_{num_generations}: Stop if the fitness value does not change (saturates) for the given number of generations.
                criterion = stop_criteria.split("_")
                stop_word = criterion[0]
                # criterion[1] might be a single or multiple numbers.
                number = criterion[1:]
                if stop_word in self.supported_stop_words:
                    pass
                else:
                    self.valid_parameters = False
                    raise ValueError(f"In the 'stop_criteria' parameter, the supported stop words are '{self.supported_stop_words}' but '{stop_word}' found.")

                if len(criterion) == 2:
                    # There is only a single number.
                    number = number[0]
                    if number.replace(".", "").replace("-", "").isnumeric():
                        number = float(number)
                    else:
                        self.valid_parameters = False
                        raise ValueError(f"The value following the stop word in the 'stop_criteria' parameter must be a number but the value ({number}) of type {type(number)} found.")

                    self.stop_criteria.append([stop_word, number])
                elif len(criterion) > 2:
                    number = validate_multi_stop_criteria(self, stop_word, number)
                    self.stop_criteria.append([stop_word] + number)
                else:
                    self.valid_parameters = False
                    raise ValueError(f"For format of a single criterion in the 'stop_criteria' parameter is 'word_number' but '{stop_criteria}' found.")

            elif type(stop_criteria) in [list, tuple, numpy.ndarray]:
                # Remove duplicate criteria by converting the list to a set then back to a list.
                stop_criteria = list(set(stop_criteria))
                for idx, val in enumerate(stop_criteria):
                    if type(val) is str:
                        criterion = val.split("_")
                        stop_word = criterion[0]
                        number = criterion[1:]
                        if len(criterion) == 2:
                            # There is only a single number.
                            number = number[0]
                            if stop_word in self.supported_stop_words:
                                pass
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"In the 'stop_criteria' parameter, the supported stop words are {self.supported_stop_words} but '{stop_word}' found.")

                            if number.replace(".", "").replace("-", "").isnumeric():
                                number = float(number)
                            else:
                                self.valid_parameters = False
                                raise ValueError(f"The value following the stop word in the 'stop_criteria' parameter must be a number but the value ({number}) of type {type(number)} found.")

                            self.stop_criteria.append([stop_word, number])
                        elif len(criterion) > 2:
                            number = validate_multi_stop_criteria(self, stop_word, number)
                            self.stop_criteria.append([stop_word] + number)
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"The format of a single criterion in the 'stop_criteria' parameter is 'word_number' but {criterion} found.")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When the 'stop_criteria' parameter is assigned a tuple/list/numpy.ndarray, then its elements must be strings but the value ({val}) of type {type(val)} found at index {idx}.")
            else:
                self.valid_parameters = False
                raise TypeError(f"The expected value of the 'stop_criteria' is a single string or a list/tuple/numpy.ndarray of strings but the value ({stop_criteria}) of type {type(stop_criteria)} found.")

            if parallel_processing is None:
                self.parallel_processing = None
            elif type(parallel_processing) in GA.supported_int_types:
                if parallel_processing > 0:
                    self.parallel_processing = ["thread", parallel_processing]
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When the 'parallel_processing' parameter is assigned an integer, then the integer must be positive but the value ({parallel_processing}) found.")
            elif type(parallel_processing) in [list, tuple]:
                if len(parallel_processing) == 2:
                    if type(parallel_processing[0]) is str:
                        if parallel_processing[0] in ["process", "thread"]:
                            if (type(parallel_processing[1]) in GA.supported_int_types and parallel_processing[1] > 0) or (parallel_processing[1] == 0) or (parallel_processing[1] is None):
                                if parallel_processing[1] == 0:
                                    # If the number of processes/threads is 0, this means no parallel processing is used. It is equivalent to setting parallel_processing=None.
                                    self.parallel_processing = None
                                else:
                                    # Whether the second value is None or a positive integer.
                                    self.parallel_processing = parallel_processing
                            else:
                                self.valid_parameters = False
                                raise TypeError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the second element must be an integer but the value ({parallel_processing[1]}) of type {type(parallel_processing[1])} found.")
                        else:
                            self.valid_parameters = False
                            raise ValueError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the value of the first element must be either 'process' or 'thread' but the value ({parallel_processing[0]}) found.")
                    else:
                        self.valid_parameters = False
                        raise TypeError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then the first element must be of type 'str' but the value ({parallel_processing[0]}) of type {type(parallel_processing[0])} found.")
                else:
                    self.valid_parameters = False
                    raise ValueError(f"When a list or tuple is assigned to the 'parallel_processing' parameter, then it must have 2 elements but ({len(parallel_processing)}) found.")
            else:
                self.valid_parameters = False
                raise ValueError(f"Unexpected value ({parallel_processing}) of type ({type(parallel_processing)}) assigned to the 'parallel_processing' parameter. The accepted values for this parameter are:\n1) None: (Default) It means no parallel processing is used.\n2) A positive integer referring to the number of threads to be used (i.e. threads, not processes, are used.\n3) list/tuple: If a list or a tuple of exactly 2 elements is assigned, then:\n\t*1) The first element can be either 'process' or 'thread' to specify whether processes or threads are used, respectively.\n\t*2) The second element can be:\n\t\t**1) A positive integer to select the maximum number of processes or threads to be used.\n\t\t**2) 0 to indicate that parallel processing is not used. This is identical to setting 'parallel_processing=None'.\n\t\t**3) None to use the default value as calculated by the concurrent.futures module.")

            # Set the `run_completed` property to False. It is set to `True` only after the `run()` method is complete.
            self.run_completed = False

            # The number of completed generations.
            self.generations_completed = 0

            # At this point, all necessary parameters validation is done successfully, and we are sure that the parameters are valid.
            # Set to True when all the parameters passed in the GA class constructor are valid.
            self.valid_parameters = True

            # Parameters of the genetic algorithm.
            self.num_generations = abs(num_generations)
            self.parent_selection_type = parent_selection_type

            # Parameters of the mutation operation.
            self.mutation_percent_genes = mutation_percent_genes
            self.mutation_num_genes = mutation_num_genes

            # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
            # A list holding the fitness value of the best solution for each generation.
            self.best_solutions_fitness = []

            # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.
            self.best_solution_generation = -1

            self.save_best_solutions = save_best_solutions
            self.best_solutions = []  # Holds the best solution in each generation.

            self.save_solutions = save_solutions
            self.solutions = []  # Holds the solutions in each generation.
            # Holds the fitness of the solutions in each generation.
            self.solutions_fitness = []

            # A list holding the fitness values of all solutions in the last generation.
            self.last_generation_fitness = None
            # A list holding the parents of the last generation.
            self.last_generation_parents = None
            # A list holding the offspring after applying crossover in the last generation.
            self.last_generation_offspring_crossover = None
            # A list holding the offspring after applying mutation in the last generation.
            self.last_generation_offspring_mutation = None
            # Holds the fitness values of one generation before the fitness values saved in the last_generation_fitness attribute. Added in PyGAD 2.16.2.
            # They are used inside the cal_pop_fitness() method to fetch the fitness of the parents in one generation before the latest generation.
            # This is to avoid re-calculating the fitness for such parents again.
            self.previous_generation_fitness = None
            # Added in PyGAD 2.18.0. A NumPy array holding the elitism of the current generation according to the value passed in the 'keep_elitism' parameter. It works only if the 'keep_elitism' parameter has a non-zero value.
            self.last_generation_elitism = None
            # Added in PyGAD 2.19.0. A NumPy array holding the indices of the elitism of the current generation. It works only if the 'keep_elitism' parameter has a non-zero value.
            self.last_generation_elitism_indices = None
            # Supported in PyGAD 3.2.0. It holds the pareto fronts when solving a multi-objective problem.
            self.pareto_fronts = None
        except Exception as e:
            self.logger.exception(e)
            # sys.exit(-1)
            raise e

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
                            if type(fitness) in GA.supported_int_float_types:
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
                            if type(fitness) in GA.supported_int_float_types:
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
                            if type(fitness) in GA.supported_int_float_types:
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
                                if type(fitness) in GA.supported_int_float_types:
                                    # The fitness function returns a single numeric value.
                                    # This is a single-objective optimization problem.
                                    pop_fitness[index] = fitness
                                elif type(fitness) in [list, tuple, numpy.ndarray]:
                                    # The fitness function returns a list/tuple/numpy.ndarray.
                                    # This is a multi-objective optimization problem.
                                    pop_fitness[index] = fitness
                                else:
                                    raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value ({fitness}) of type {type(fitness)} found.")

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

    def save(self, filename):
        """
        Saves the genetic algorithm instance:
            -filename: Name of the file to save the instance. No extension is needed.
        """

        cloudpickle_serialized_object = cloudpickle.dumps(self)
        with open(filename + ".pkl", 'wb') as file:
            file.write(cloudpickle_serialized_object)
            cloudpickle.dump(self, file)

    def summary(self,
                line_length=70,
                fill_character=" ",
                line_character="-",
                line_character2="=",
                columns_equal_len=False,
                print_step_parameters=True,
                print_parameters_summary=True):
        """
        The summary() method prints a summary of the PyGAD lifecycle in a Keras style.
        The parameters are:
            line_length: An integer representing the length of the single line in characters.
            fill_character: A character to fill the lines.
            line_character: A character for creating a line separator.
            line_character2: A secondary character to create a line separator.
            columns_equal_len: The table rows are split into equal-sized columns or split subjective to the width needed.
            print_step_parameters: Whether to print extra parameters about each step inside the step. If print_step_parameters=False and print_parameters_summary=True, then the parameters of each step are printed at the end of the table.
            print_parameters_summary: Whether to print parameters summary at the end of the table. If print_step_parameters=False, then the parameters of each step are printed at the end of the table too.
        """

        summary_output = ""

        def fill_message(msg, line_length=line_length, fill_character=fill_character):
            num_spaces = int((line_length - len(msg))/2)
            num_spaces = int(num_spaces / len(fill_character))
            msg = "{spaces}{msg}{spaces}".format(
                msg=msg, spaces=fill_character * num_spaces)
            return msg

        def line_separator(line_length=line_length, line_character=line_character):
            num_characters = int(line_length / len(line_character))
            return line_character * num_characters

        def create_row(columns, line_length=line_length, fill_character=fill_character, split_percentages=None):
            filled_columns = []
            if split_percentages is None:
                split_percentages = [int(100/len(columns))] * 3
            columns_lengths = [int((split_percentages[idx] * line_length) / 100)
                               for idx in range(len(split_percentages))]
            for column_idx, column in enumerate(columns):
                current_column_length = len(column)
                extra_characters = columns_lengths[column_idx] - \
                    current_column_length
                filled_column = column + fill_character * extra_characters
                filled_columns.append(filled_column)

            return "".join(filled_columns)

        def print_parent_selection_params():
            nonlocal summary_output
            m = f"Number of Parents: {self.num_parents_mating}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if self.parent_selection_type == "tournament":
                m = f"K Tournament: {self.K_tournament}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_fitness_params():
            nonlocal summary_output
            if not self.fitness_batch_size is None:
                m = f"Fitness batch size: {self.fitness_batch_size}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_crossover_params():
            nonlocal summary_output
            if not self.crossover_probability is None:
                m = f"Crossover probability: {self.crossover_probability}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_mutation_params():
            nonlocal summary_output
            if not self.mutation_probability is None:
                m = f"Mutation Probability: {self.mutation_probability}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            if self.mutation_percent_genes == "default":
                m = f"Mutation Percentage: {self.mutation_percent_genes}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            # Number of mutation genes is already showed above.
            m = f"Mutation Genes: {self.mutation_num_genes}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Random Mutation Range: ({self.random_mutation_min_val}, {self.random_mutation_max_val})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if not self.gene_space is None:
                m = f"Gene Space: {self.gene_space}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Mutation by Replacement: {self.mutation_by_replacement}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Allow Duplicated Genes: {self.allow_duplicate_genes}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        def print_on_generation_params():
            nonlocal summary_output
            if not self.stop_criteria is None:
                m = f"Stop Criteria: {self.stop_criteria}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_params_summary():
            nonlocal summary_output
            m = f"Population Size: ({self.sol_per_pop}, {self.num_genes})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Number of Generations: {self.num_generations}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Initial Population Range: ({self.init_range_low}, {self.init_range_high})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

            if not print_step_parameters:
                print_fitness_params()

            if not print_step_parameters:
                print_parent_selection_params()

            if self.keep_elitism != 0:
                m = f"Keep Elitism: {self.keep_elitism}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            else:
                m = f"Keep Parents: {self.keep_parents}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Gene DType: {self.gene_type}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

            if not print_step_parameters:
                print_crossover_params()

            if not print_step_parameters:
                print_mutation_params()

            if not print_step_parameters:
                print_on_generation_params()

            if not self.parallel_processing is None:
                m = f"Parallel Processing: {self.parallel_processing}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            if not self.random_seed is None:
                m = f"Random Seed: {self.random_seed}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Save Best Solutions: {self.save_best_solutions}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Save Solutions: {self.save_solutions}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        m = line_separator(line_character=line_character)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = fill_message("PyGAD Lifecycle")
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"

        lifecycle_steps = ["on_start()", "Fitness Function", "On Fitness", "Parent Selection", "On Parents",
                           "Crossover", "On Crossover", "Mutation", "On Mutation", "On Generation", "On Stop"]
        lifecycle_functions = [self.on_start, self.fitness_func, self.on_fitness, self.select_parents, self.on_parents,
                               self.crossover, self.on_crossover, self.mutation, self.on_mutation, self.on_generation, self.on_stop]
        lifecycle_functions = [getattr(
            lifecycle_func, '__name__', "None") for lifecycle_func in lifecycle_functions]
        lifecycle_functions = [lifecycle_func + "()" if lifecycle_func !=
                               "None" else "None" for lifecycle_func in lifecycle_functions]
        lifecycle_output = ["None", "(1)", "None", f"({self.num_parents_mating}, {self.num_genes})", "None",
                            f"({self.num_parents_mating}, {self.num_genes})", "None", f"({self.num_parents_mating}, {self.num_genes})", "None", "None", "None"]
        lifecycle_step_parameters = [None, print_fitness_params, None, print_parent_selection_params, None,
                                     print_crossover_params, None, print_mutation_params, None, print_on_generation_params, None]

        if not columns_equal_len:
            max_lengthes = [max(list(map(len, lifecycle_steps))), max(
                list(map(len, lifecycle_functions))), max(list(map(len, lifecycle_output)))]
            split_percentages = [
                int((column_len / sum(max_lengthes)) * 100) for column_len in max_lengthes]
        else:
            split_percentages = None

        header_columns = ["Step", "Handler", "Output Shape"]
        header_row = create_row(
            header_columns, split_percentages=split_percentages)
        m = header_row
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"

        for lifecycle_idx in range(len(lifecycle_steps)):
            lifecycle_column = [lifecycle_steps[lifecycle_idx],
                                lifecycle_functions[lifecycle_idx], lifecycle_output[lifecycle_idx]]
            if lifecycle_column[1] == "None":
                continue
            lifecycle_row = create_row(
                lifecycle_column, split_percentages=split_percentages)
            m = lifecycle_row
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if print_step_parameters:
                if not lifecycle_step_parameters[lifecycle_idx] is None:
                    lifecycle_step_parameters[lifecycle_idx]()
            m = line_separator(line_character=line_character)
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        if print_parameters_summary:
            print_params_summary()
            m = line_separator(line_character=line_character2)
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
        return summary_output

def load(filename):
    """
    Reads a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. No extension is needed.
    Returns the genetic algorithm instance.
    """

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
    except:
        # raise BaseException("Error loading the file. If the file already exists, please reload all the functions previously used (e.g. fitness function).")
        raise BaseException("Error loading the file.")
    return ga_in
