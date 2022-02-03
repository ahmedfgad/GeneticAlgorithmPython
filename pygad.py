import numpy
import random
import matplotlib.pyplot
import pickle
import time
import warnings

class GA:

    supported_int_types = [int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
    supported_float_types = [float, numpy.float, numpy.float16, numpy.float32, numpy.float64]
    supported_int_float_types = supported_int_types + supported_float_types

    def __init__(self, 
                 num_generations, 
                 num_parents_mating, 
                 fitness_func,
                 initial_population=None,
                 sol_per_pop=None, 
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 gene_type=float,
                 parent_selection_type="sss",
                 keep_parents=-1,
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
                 allow_duplicate_genes=True,
                 on_start=None,
                 on_fitness=None,
                 on_parents=None,
                 on_crossover=None,
                 on_mutation=None,
                 callback_generation=None,
                 on_generation=None,
                 on_stop=None,
                 delay_after_gen=0.0,
                 save_best_solutions=False,
                 save_solutions=False,
                 suppress_warnings=False,
                 stop_criteria=None):

        """
        The constructor of the GA class accepts all parameters required to create an instance of the GA class. It validates such parameters.

        num_generations: Number of generations.
        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        fitness_func: Accepts a function that must accept 2 parameters (a single solution and its index in the population) and return the fitness value of the solution. Available starting from PyGAD 1.0.17 until 1.0.20 with a single parameter representing the solution. Changed in PyGAD 2.0.0 and higher to include the second parameter representing the solution index.

        initial_population: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to None which means no initial population is specified by the user. In this case, PyGAD creates an initial population using the 'sol_per_pop' and 'num_genes' parameters. An exception is raised if the 'initial_population' is None while any of the 2 parameters ('sol_per_pop' or 'num_genes') is also None.
        sol_per_pop: Number of solutions in the population. 
        num_genes: Number of parameters in the function.

        init_range_low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        init_range_high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        # It is OK to set the value of any of the 2 parameters ('init_range_low' and 'init_range_high') to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).

        gene_type: The type of the gene. It is assigned to any of these types (int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64) and forces all the genes to be of that type.

        parent_selection_type: Type of parent selection.
        keep_parents: If 0, this means no parent in the current population will be used in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. For some parent selection operators like rank selection, the parents are of high quality and it is beneficial to keep them in the next generation. In some other parent selection operators like roulette wheel selection (RWS), it is not guranteed that the parents will be of high quality and thus keeping the parents might degarde the quality of the population.
        K_tournament: When the value of 'parent_selection_type' is 'tournament', the 'K_tournament' parameter specifies the number of solutions from which a parent is selected randomly.

        crossover_type: Type of the crossover opreator. If  crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        crossover_probability: The probability of selecting a solution for the crossover operation. If the solution probability is <= crossover_probability, the solution is selected. The value must be between 0 and 1 inclusive.

        mutation_type: Type of the mutation opreator. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        mutation_probability: The probability of selecting a gene for the mutation operation. If the gene probability is <= mutation_probability, the gene is selected. It accepts either a single value for fixed mutation or a list/tuple/numpy.ndarray of 2 values for adaptive mutation. The values must be between 0 and 1 inclusive. If specified, then no need for the 2 parameters mutation_percent_genes and mutation_num_genes.

        mutation_by_replacement: An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, setting mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene.

        mutation_percent_genes: Percentage of genes to mutate which defaults to the string 'default' which means 10%. This parameter has no action if any of the 2 parameters mutation_probability or mutation_num_genes exist.
        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes. This parameter has no action if the mutation_probability parameter exists.
        random_mutation_min_val: The minimum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to -1.0.
        random_mutation_max_val: The maximum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to 1.0.

        gene_space: It accepts a list of all possible values of the gene. This list is used in the mutation step. Should be used only if the gene space is a set of discrete values. No need for the 2 parameters (random_mutation_min_val and random_mutation_max_val) if the parameter gene_space exists. Added in PyGAD 2.5.0. In PyGAD 2.11.0, the gene_space can be assigned a dict.

        on_start: Accepts a function to be called only once before the genetic algorithm starts its evolution. This function must accept a single parameter representing the instance of the genetic algorithm. Added in PyGAD 2.6.0.
        on_fitness: Accepts a function to be called after calculating the fitness values of all solutions in the population. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one is a list of all solutions' fitness values. Added in PyGAD 2.6.0.
        on_parents: Accepts a function to be called after selecting the parents that mates. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the selected parents. Added in PyGAD 2.6.0.
        on_crossover: Accepts a function to be called each time the crossover operation is applied. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring generated using crossover. Added in PyGAD 2.6.0.
        on_mutation: Accepts a function to be called each time the mutation operation is applied. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring after applying the mutation. Added in PyGAD 2.6.0.
        callback_generation: Accepts a function to be called after each generation. This function must accept a single parameter representing the instance of the genetic algorithm. If the function returned "stop", then the run() method stops without completing the other generations. Starting from PyGAD 2.6.0, the callback_generation parameter is deprecated and should be replaced by the on_generation parameter.
        on_generation: Accepts a function to be called after each generation. This function must accept a single parameter representing the instance of the genetic algorithm. If the function returned "stop", then the run() method stops without completing the other generations. Added in PyGAD 2.6.0.
        on_stop: Accepts a function to be called only once exactly before the genetic algorithm stops or when it completes all the generations. This function must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one is a list of fitness values of the last population's solutions. Added in PyGAD 2.6.0. 

        delay_after_gen: Added in PyGAD 2.4.0. It accepts a non-negative number specifying the number of seconds to wait after a generation completes and before going to the next generation. It defaults to 0.0 which means no delay after the generation.

        save_best_solutions: Added in PyGAD 2.9.0 and its type is bool. If True, then the best solution in each generation is saved into the 'best_solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations or the number of genes is large.
        save_solutions: Added in PyGAD 2.15.0 and its type is bool. If True, then all solutions in each generation are saved into the 'solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.

        suppress_warnings: Added in PyGAD 2.10.0 and its type is bool. If True, then no warning messages will be displayed. It defaults to False.

        allow_duplicate_genes: Added in PyGAD 2.13.0. If True, then a solution/chromosome may have duplicate gene values. If False, then each gene will have a unique value in its solution.

        stop_criteria: Added in PyGAD 2.15.0. It is assigned to some criteria to stop the evolution if at least one criterion holds.
        """

        # If suppress_warnings is bool and its valud is False, then print warning messages.
        if type(suppress_warnings) is bool:
            self.suppress_warnings = suppress_warnings
        else:
            self.valid_parameters = False
            raise TypeError("The expected type of the 'suppress_warnings' parameter is bool but {suppress_warnings_type} found.".format(suppress_warnings_type=type(suppress_warnings)))

        # Validating mutation_by_replacement
        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'mutation_by_replacement' parameter is bool but ({mutation_by_replacement_type}) found.".format(mutation_by_replacement_type=type(mutation_by_replacement)))

        self.mutation_by_replacement = mutation_by_replacement

        # Validate gene_space
        self.gene_space_nested = False
        if type(gene_space) is type(None):
            pass
        elif type(gene_space) in [list, tuple, range, numpy.ndarray]:
            if len(gene_space) == 0:
                self.valid_parameters = False
                raise TypeError("'gene_space' cannot be empty (i.e. its length must be >= 0).")
            else:
                for index, el in enumerate(gene_space):
                    if type(el) in [list, tuple, range, numpy.ndarray]:
                        if len(el) == 0:
                            self.valid_parameters = False
                            raise TypeError("The element indexed {index} of 'gene_space' with type {el_type} cannot be empty (i.e. its length must be >= 0).".format(index=index, el_type=type(el)))
                        else:
                            for val in el:
                                if not (type(val) in [type(None)] + GA.supported_int_float_types):
                                    raise TypeError("All values in the sublists inside the 'gene_space' attribute must be numeric of type int/float/None but ({val}) of type {typ} found.".format(val=val, typ=type(val)))
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
                                raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=el.keys()))
                        elif len(el.items()) == 3:
                            if ('low' in el.keys()) and ('high' in el.keys()) and ('step' in el.keys()):
                                pass
                            else:
                                self.valid_parameters = False
                                raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it can have the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=el.keys()))
                        else:
                            self.valid_parameters = False
                            raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(el.items())))
                        self.gene_space_nested = True
                    elif not (type(el) in GA.supported_int_float_types):
                        self.valid_parameters = False
                        raise TypeError("Unexpected type {el_type} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.".format(index=index, el_type=type(el)))

        elif type(gene_space) is dict:
            if len(gene_space.items()) == 2:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise TypeError("When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=gene_space.keys()))
            elif len(gene_space.items()) == 3:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()) and  ('step' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise TypeError("When the 'gene_space' parameter is of type dict, then it can have only the keys 'low', 'high', and 'step' (optional) but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=gene_space.keys()))
            else:
                self.valid_parameters = False
                raise TypeError("When the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(gene_space.items())))

        else:
            self.valid_parameters = False
            raise TypeError("The expected type of 'gene_space' is list, tuple, range, or numpy.ndarray but ({gene_space_type}) found.".format(gene_space_type=type(gene_space)))
            
        self.gene_space = gene_space

        # Validate init_range_low and init_range_high
        if type(init_range_low) in GA.supported_int_float_types:
            if type(init_range_high) in GA.supported_int_float_types:
                self.init_range_low = init_range_low
                self.init_range_high = init_range_high
            else:
                self.valid_parameters = False
                raise ValueError("The value passed to the 'init_range_high' parameter must be either integer or floating-point number but the value ({init_range_high_value}) of type {init_range_high_type} found.".format(init_range_high_value=init_range_high, init_range_high_type=type(init_range_high)))
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'init_range_low' parameter must be either integer or floating-point number but the value ({init_range_low_value}) of type {init_range_low_type} found.".format(init_range_low_value=init_range_low, init_range_low_type=type(init_range_low)))


        # Validate random_mutation_min_val and random_mutation_max_val
        if type(random_mutation_min_val) in GA.supported_int_float_types:
            if type(random_mutation_max_val) in GA.supported_int_float_types:
                if random_mutation_min_val == random_mutation_max_val:
                    if not self.suppress_warnings: warnings.warn("The values of the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val' are equal and this causes a fixed change to all genes.")
            else:
                self.valid_parameters = False
                raise TypeError("The expected type of the 'random_mutation_max_val' parameter is numeric but ({random_mutation_max_val_type}) found.".format(random_mutation_max_val_type=type(random_mutation_max_val)))
        else:
            self.valid_parameters = False
            raise TypeError("The expected type of the 'random_mutation_min_val' parameter is numeric but ({random_mutation_min_val_type}) found.".format(random_mutation_min_val_type=type(random_mutation_min_val)))
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val

        # Validate gene_type
        if gene_type in GA.supported_int_float_types:
            self.gene_type = [gene_type, None]
            self.gene_type_single = True
        # A single data type of float with precision.
        elif len(gene_type) == 2 and gene_type[0] in GA.supported_float_types and (type(gene_type[1]) in GA.supported_int_types or gene_type[1] is None):
            self.gene_type = gene_type
            self.gene_type_single = True
        elif type(gene_type) in [list, tuple, numpy.ndarray]:
            if not len(gene_type) == num_genes:
                self.valid_parameters = False
                raise TypeError("When the parameter 'gene_type' is nested, then it can be either [float, int<precision>] or with length equal to the value passed to the 'num_genes' parameter. Instead, value {gene_type_val} with len(gene_type) ({len_gene_type}) != len(num_genes) ({num_genes}) found.".format(gene_type_val=gene_type, len_gene_type=len(gene_type), num_genes=num_genes))
            for gene_type_idx, gene_type_val in enumerate(gene_type):
                if gene_type_val in GA.supported_float_types:
                    # If the gene type is float and no precision is passed, set it to None.
                    gene_type[gene_type_idx] = [gene_type_val, None]
                elif gene_type_val in GA.supported_int_types:
                    gene_type[gene_type_idx] = [gene_type_val, None]
                elif type(gene_type_val) in [list, tuple, numpy.ndarray]:
                    # A float type is expected in a list/tuple/numpy.ndarray of length 2.
                    if len(gene_type_val) == 2:
                        if gene_type_val[0] in GA.supported_float_types:
                            if type(gene_type_val[1]) in GA.supported_int_types:
                                pass
                            else:
                                self.valid_parameters = False
                                raise ValueError("In the 'gene_type' parameter, the precision for float gene data types must be an integer but the element {gene_type_val} at index {gene_type_idx} has a precision of {gene_type_precision_val} with type {gene_type_type} .".format(gene_type_val=gene_type_val, gene_type_precision_val=gene_type_val[1], gene_type_type=gene_type_val[0], gene_type_idx=gene_type_idx))
                        else:
                            self.valid_parameters = False
                            raise ValueError("In the 'gene_type' parameter, a precision is expected only for float gene data types but the element {gene_type} found at index {gene_type_idx}. Note that the data type must be at index 0 followed by precision at index 1.".format(gene_type=gene_type_val, gene_type_idx=gene_type_idx))
                    else:
                        self.valid_parameters = False
                        raise ValueError("In the 'gene_type' parameter, a precision is specified in a list/tuple/numpy.ndarray of length 2 but value ({gene_type_val}) of type {gene_type_type} with length {gene_type_length} found at index {gene_type_idx}.".format(gene_type_val=gene_type_val, gene_type_type=type(gene_type_val), gene_type_idx=gene_type_idx, gene_type_length=len(gene_type_val)))
                else:
                    self.valid_parameters = False
                    raise ValueError("When a list/tuple/numpy.ndarray is assigned to the 'gene_type' parameter, then its elements must be of integer, floating-point, list, tuple, or numpy.ndarray data types but the value ({gene_type_val}) of type {gene_type_type} found at index {gene_type_idx}.".format(gene_type_val=gene_type_val, gene_type_type=type(gene_type_val), gene_type_idx=gene_type_idx))
            self.gene_type = gene_type
            self.gene_type_single = False
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'gene_type' parameter must be either a single integer, floating-point, list, tuple, or numpy.ndarray but ({gene_type_val}) of type {gene_type_type} found.".format(gene_type_val=gene_type, gene_type_type=type(gene_type)))
        
        # Build the initial population
        if initial_population is None:
            if (sol_per_pop is None) or (num_genes is None):
                self.valid_parameters = False
                raise ValueError("Error creating the initail population\n\nWhen the parameter initial_population is None, then neither of the 2 parameters sol_per_pop and num_genes can be None at the same time.\nThere are 2 options to prepare the initial population:\n1) Create an initial population and assign it to the initial_population parameter. In this case, the values of the 2 parameters sol_per_pop and num_genes will be deduced.\n2) Allow the genetic algorithm to create the initial population automatically by passing valid integer values to the sol_per_pop and num_genes parameters.")
            elif (type(sol_per_pop) is int) and (type(num_genes) is int):
                # Validating the number of solutions in the population (sol_per_pop)
                if sol_per_pop <= 0:
                    self.valid_parameters = False
                    raise ValueError("The number of solutions in the population (sol_per_pop) must be > 0 but ({sol_per_pop}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(sol_per_pop=sol_per_pop))
                # Validating the number of gene.
                if (num_genes <= 0):
                    self.valid_parameters = False
                    raise ValueError("The number of genes cannot be <= 0 but ({num_genes}) found.\n".format(num_genes=num_genes))
                # When initial_population=None and the 2 parameters sol_per_pop and num_genes have valid integer values, then the initial population is created.
                # Inside the initialize_population() method, the initial_population attribute is assigned to keep the initial population accessible.
                self.num_genes = num_genes # Number of genes in the solution.

                # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
                if self.gene_space_nested:
                    if len(gene_space) != self.num_genes:
                        self.valid_parameters = False
                        raise TypeError("When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len_gene_space}) != num_genes ({num_genes})".format(len_gene_space=len(gene_space), num_genes=self.num_genes))

                self.sol_per_pop = sol_per_pop # Number of solutions in the population.
                self.initialize_population(self.init_range_low, self.init_range_high, allow_duplicate_genes, True, self.gene_type)
            else:
                self.valid_parameters = False
                raise TypeError("The expected type of both the sol_per_pop and num_genes parameters is int but ({sol_per_pop_type}) and {num_genes_type} found.".format(sol_per_pop_type=type(sol_per_pop), num_genes_type=type(num_genes)))
        elif numpy.array(initial_population).ndim != 2:
            self.valid_parameters = False
            raise ValueError("A 2D list is expected to the initail_population parameter but a ({initial_population_ndim}-D) list found.".format(initial_population_ndim=numpy.array(initial_population).ndim))
        else:
            # Forcing the initial_population array to have the data type assigned to the gene_type parameter.
            if self.gene_type_single == True:
                if self.gene_type[1] == None:
                    self.initial_population = numpy.array(initial_population, dtype=self.gene_type[0])
                else:
                    self.initial_population = numpy.round(numpy.array(initial_population, dtype=self.gene_type[0]), self.gene_type[1])
            else:
                initial_population = numpy.array(initial_population)
                self.initial_population = numpy.zeros(shape=(initial_population.shape[0], initial_population.shape[1]), dtype=object)
                for gene_idx in range(initial_population.shape[1]):
                    if self.gene_type[gene_idx][1] is None:
                        self.initial_population[:, gene_idx] = numpy.asarray(initial_population[:, gene_idx], 
                                                                             dtype=self.gene_type[gene_idx][0])
                    else:
                        self.initial_population[:, gene_idx] = numpy.round(numpy.asarray(initial_population[:, gene_idx], 
                                                                                         dtype=self.gene_type[gene_idx][0]), 
                                                                           self.gene_type[gene_idx][1])

            self.population = self.initial_population.copy() # A NumPy array holding the initial population.
            self.num_genes = self.initial_population.shape[1] # Number of genes in the solution.
            self.sol_per_pop = self.initial_population.shape[0]  # Number of solutions in the population.
            self.pop_size = (self.sol_per_pop,self.num_genes) # The population size.

        # Round initial_population and population
        self.initial_population = self.round_genes(self.initial_population)
        self.population = self.round_genes(self.population)

        # In case the 'gene_space' parameter is nested, then make sure the number of its elements equals to the number of genes.
        if self.gene_space_nested:
            if len(gene_space) != self.num_genes:
                self.valid_parameters = False
                raise TypeError("When the parameter 'gene_space' is nested, then its length must be equal to the value passed to the 'num_genes' parameter. Instead, length of gene_space ({len_gene_space}) != num_genes ({len_num_genes})".format(len_gene_space=len(gene_space), len_num_genes=self.num_genes))

        # Validating the number of parents to be selected for mating (num_parents_mating)
        if num_parents_mating <= 0:
            self.valid_parameters = False
            raise ValueError("The number of parents mating (num_parents_mating) parameter must be > 0 but ({num_parents_mating}) found. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n".format(num_parents_mating=num_parents_mating))

        # Validating the number of parents to be selected for mating: num_parents_mating
        if (num_parents_mating > self.sol_per_pop):
            self.valid_parameters = False
            raise ValueError("The number of parents to select for mating ({num_parents_mating}) cannot be greater than the number of solutions in the population ({sol_per_pop}) (i.e., num_parents_mating must always be <= sol_per_pop).\n".format(num_parents_mating=num_parents_mating, sol_per_pop=self.sol_per_pop))

        self.num_parents_mating = num_parents_mating

        # crossover: Refers to the method that applies the crossover operator based on the selected type of crossover in the crossover_type property.
        # Validating the crossover type: crossover_type
        if (crossover_type is None):
            self.crossover = None
        elif callable(crossover_type):
            # Check if the crossover_type is a function that accepts 2 paramaters.
            if (crossover_type.__code__.co_argcount == 3):
                # The crossover function assigned to the crossover_type parameter is validated.
                self.crossover = crossover_type
            else:
                self.valid_parameters = False
                raise ValueError("When 'crossover_type' is assigned to a function, then this crossover function must accept 2 parameters:\n1) The selected parents.\n2) The size of the offspring to be produced.3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed crossover function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=crossover_type.__code__.co_name, argcount=crossover_type.__code__.co_argcount))
        elif not (type(crossover_type) is str):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'crossover_type' parameter is either callable or str but ({crossover_type}) found.".format(crossover_type=type(crossover_type)))
        else: # type crossover_type is str
            crossover_type = crossover_type.lower()
            if (crossover_type == "single_point"):
                self.crossover = self.single_point_crossover
            elif (crossover_type == "two_points"):
                self.crossover = self.two_points_crossover
            elif (crossover_type == "uniform"):
                self.crossover = self.uniform_crossover
            elif (crossover_type == "scattered"):
                self.crossover = self.scattered_crossover
            else:
                self.valid_parameters = False
                raise ValueError("Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) parameter does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover)\n-scattered (for scattered crossover).\n".format(crossover_type=crossover_type))

        self.crossover_type = crossover_type

        # Calculate the value of crossover_probability
        if crossover_probability is None:
            self.crossover_probability = None
        elif type(crossover_probability) in GA.supported_int_float_types:
            if crossover_probability >= 0 and crossover_probability <= 1:
                self.crossover_probability = crossover_probability
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'crossover_probability' parameter must be between 0 and 1 inclusive but ({crossover_probability_value}) found.".format(crossover_probability_value=crossover_probability))
        else:
            self.valid_parameters = False
            raise ValueError("Unexpected type for the 'crossover_probability' parameter. Float is expected but ({crossover_probability_value}) of type {crossover_probability_type} found.".format(crossover_probability_value=crossover_probability, crossover_probability_type=type(crossover_probability)))

        # mutation: Refers to the method that applies the mutation operator based on the selected type of mutation in the mutation_type property.
        # Validating the mutation type: mutation_type
        # "adaptive" mutation is supported starting from PyGAD 2.10.0
        if mutation_type is None:
            self.mutation = None
        elif callable(mutation_type):
            # Check if the mutation_type is a function that accepts 1 paramater.
            if (mutation_type.__code__.co_argcount == 2):
                # The mutation function assigned to the mutation_type parameter is validated.
                self.mutation = mutation_type
            else:
                self.valid_parameters = False
                raise ValueError("When 'mutation_type' is assigned to a function, then this mutation function must accept 2 parameters:\n1) The offspring to be mutated.\n2) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed mutation function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=mutation_type.__code__.co_name, argcount=mutation_type.__code__.co_argcount))
        elif not (type(mutation_type) is str):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'mutation_type' parameter is either callable or str but ({mutation_type}) found.".format(mutation_type=type(mutation_type)))
        else: # type mutation_type is str
            mutation_type = mutation_type.lower()
            if (mutation_type == "random"):
                self.mutation = self.random_mutation
            elif (mutation_type == "swap"):
                self.mutation = self.swap_mutation
            elif (mutation_type == "scramble"):
                self.mutation = self.scramble_mutation
            elif (mutation_type == "inversion"):
                self.mutation = self.inversion_mutation
            elif (mutation_type == "adaptive"):
                self.mutation = self.adaptive_mutation
            else:
                self.valid_parameters = False
                raise ValueError("Undefined mutation type. \nThe assigned string value to the 'mutation_type' parameter ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation)\n-adaptive (for adaptive mutation).\n".format(mutation_type=mutation_type))

        self.mutation_type = mutation_type

        # Calculate the value of mutation_probability
        if not (self.mutation_type is None):
            if mutation_probability is None:
                self.mutation_probability = None
            elif (mutation_type != "adaptive"):
                # Mutation probability is fixed not adaptive.
                if type(mutation_probability) in GA.supported_int_float_types:
                    if mutation_probability >= 0 and mutation_probability <= 1:
                        self.mutation_probability = mutation_probability
                    else:
                        self.valid_parameters = False
                        raise ValueError("The value assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({mutation_probability_value}) found.".format(mutation_probability_value=mutation_probability))
                else:
                    self.valid_parameters = False
                    raise ValueError("Unexpected type for the 'mutation_probability' parameter. A numeric value is expected but ({mutation_probability_value}) of type {mutation_probability_type} found.".format(mutation_probability_value=mutation_probability, mutation_probability_type=type(mutation_probability)))
            else:
                # Mutation probability is adaptive not fixed.
                if type(mutation_probability) in [list, tuple, numpy.ndarray]:
                    if len(mutation_probability) == 2:
                        for el in mutation_probability:
                            if type(el) in GA.supported_int_float_types:
                                if el >= 0 and el <= 1:
                                    pass
                                else:
                                    self.valid_parameters = False
                                    raise ValueError("The values assigned to the 'mutation_probability' parameter must be between 0 and 1 inclusive but ({mutation_probability_value}) found.".format(mutation_probability_value=el))
                            else:
                                self.valid_parameters = False
                                raise ValueError("Unexpected type for a value assigned to the 'mutation_probability' parameter. A numeric value is expected but ({mutation_probability_value}) of type {mutation_probability_type} found.".format(mutation_probability_value=el, mutation_probability_type=type(el)))
                        if mutation_probability[0] < mutation_probability[1]:
                            if not self.suppress_warnings: warnings.warn("The first element in the 'mutation_probability' parameter is {first_el} which is smaller than the second element {second_el}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high qualitiy solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.".format(first_el=mutation_probability[0], second_el=mutation_probability[1]))
                        self.mutation_probability = mutation_probability
                    else:
                        self.valid_parameters = False
                        raise ValueError("When mutation_type='adaptive', then the 'mutation_probability' parameter must have only 2 elements but ({mutation_probability_length}) element(s) found.".format(mutation_probability_length=len(mutation_probability)))
                else:
                    self.valid_parameters = False
                    raise ValueError("Unexpected type for the 'mutation_probability' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_probability_value}) of type {mutation_probability_type} found.".format(mutation_probability_value=mutation_probability, mutation_probability_type=type(mutation_probability)))
        else:
            pass

        # Calculate the value of mutation_num_genes
        if not (self.mutation_type is None):
            if mutation_num_genes is None:
                # The mutation_num_genes parameter does not exist. Checking whether adaptive mutation is used.
                if (mutation_type != "adaptive"):
                    # The percent of genes to mutate is fixed not adaptive.
                    if mutation_percent_genes == 'default'.lower():
                        mutation_percent_genes = 10
                        # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                        mutation_num_genes = numpy.uint32((mutation_percent_genes*self.num_genes)/100)
                        # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                        if mutation_num_genes == 0:
                            if self.mutation_probability is None:
                                if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate (mutation_percent_genes={mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes, mutation_num=mutation_num_genes))
                            mutation_num_genes = 1

                    elif type(mutation_percent_genes) in GA.supported_int_float_types:
                        if (mutation_percent_genes <= 0 or mutation_percent_genes > 100):
                            self.valid_parameters = False
                            raise ValueError("The percentage of selected genes for mutation (mutation_percent_genes) must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n".format(mutation_percent_genes=mutation_percent_genes))
                        else:
                            # If mutation_percent_genes equals the string "default", then it is replaced by the numeric value 10.
                            if mutation_percent_genes == 'default'.lower():
                                mutation_percent_genes = 10

                            # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                            mutation_num_genes = numpy.uint32((mutation_percent_genes*self.num_genes)/100)
                            # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                            if mutation_num_genes == 0:
                                if self.mutation_probability is None:
                                    if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate (mutation_percent_genes={mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes, mutation_num=mutation_num_genes))
                                mutation_num_genes = 1
                    else:
                        self.valid_parameters = False
                        raise ValueError("Unexpected value or type of the 'mutation_percent_genes' parameter. It only accepts the string 'default' or a numeric value but ({mutation_percent_genes_value}) of type {mutation_percent_genes_type} found.".format(mutation_percent_genes_value=mutation_percent_genes, mutation_percent_genes_type=type(mutation_percent_genes)))
                else:
                    # The percent of genes to mutate is adaptive not fixed.
                    if type(mutation_percent_genes) in [list, tuple, numpy.ndarray]:
                        if len(mutation_percent_genes) == 2:
                            mutation_num_genes = numpy.zeros_like(mutation_percent_genes, dtype=numpy.uint32)
                            for idx, el in enumerate(mutation_percent_genes):
                                if type(el) in GA.supported_int_float_types:
                                    if (el <= 0 or el > 100):
                                        self.valid_parameters = False
                                        raise ValueError("The values assigned to the 'mutation_percent_genes' must be > 0 and <= 100 but ({mutation_percent_genes}) found.\n".format(mutation_percent_genes=mutation_percent_genes))
                                else:
                                    self.valid_parameters = False
                                    raise ValueError("Unexpected type for a value assigned to the 'mutation_percent_genes' parameter. An integer value is expected but ({mutation_percent_genes_value}) of type {mutation_percent_genes_type} found.".format(mutation_percent_genes_value=el, mutation_percent_genes_type=type(el)))
                                # At this point of the loop, the current value assigned to the parameter 'mutation_percent_genes' is validated.
                                # Based on the mutation percentage in the 'mutation_percent_genes' parameter, the number of genes to mutate is calculated.
                                mutation_num_genes[idx] = numpy.uint32((mutation_percent_genes[idx]*self.num_genes)/100)
                                # Based on the mutation percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
                                if mutation_num_genes[idx] == 0:
                                    if not self.suppress_warnings: warnings.warn("The percentage of genes to mutate ({mutation_percent}) resutled in selecting ({mutation_num}) genes. The number of genes to mutate is set to 1 (mutation_num_genes=1).\nIf you do not want to mutate any gene, please set mutation_type=None.".format(mutation_percent=mutation_percent_genes[idx], mutation_num=mutation_num_genes[idx]))
                                    mutation_num_genes[idx] = 1
                            if mutation_percent_genes[0] < mutation_percent_genes[1]:
                                if not self.suppress_warnings: warnings.warn("The first element in the 'mutation_percent_genes' parameter is ({first_el}) which is smaller than the second element ({second_el}).\nThis means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high qualitiy solutions while making little changes in the low quality solutions.\nPlease make the first element higher than the second element.".format(first_el=mutation_percent_genes[0], second_el=mutation_percent_genes[1]))
                            # At this point outside the loop, all values of the parameter 'mutation_percent_genes' are validated. Eveyrthing is OK.
                        else:
                            self.valid_parameters = False
                            raise ValueError("When mutation_type='adaptive', then the 'mutation_percent_genes' parameter must have only 2 elements but ({mutation_percent_genes_length}) element(s) found.".format(mutation_percent_genes_length=len(mutation_percent_genes)))
                    else:
                        if self.mutation_probability is None:
                            self.valid_parameters = False
                            raise ValueError("Unexpected type for the 'mutation_percent_genes' parameter. When mutation_type='adaptive', then the 'mutation_percent_genes' parameter should exist and assigned a list/tuple/numpy.ndarray with 2 values but ({mutation_percent_genes_value}) found.".format(mutation_percent_genes_value=mutation_percent_genes))
            # The mutation_num_genes parameter exists. Checking whether adaptive mutation is used.
            elif (mutation_type != "adaptive"):
                # Number of genes to mutate is fixed not adaptive.
                if type(mutation_num_genes) in GA.supported_int_types:
                    if (mutation_num_genes <= 0):
                        self.valid_parameters = False
                        raise ValueError("The number of selected genes for mutation (mutation_num_genes) cannot be <= 0 but ({mutation_num_genes}) found. If you do not want to use mutation, please set mutation_type=None\n".format(mutation_num_genes=mutation_num_genes))
                    elif (mutation_num_genes > self.num_genes):
                        self.valid_parameters = False
                        raise ValueError("The number of selected genes for mutation (mutation_num_genes), which is ({mutation_num_genes}), cannot be greater than the number of genes ({num_genes}).\n".format(mutation_num_genes=mutation_num_genes, num_genes=self.num_genes))
                else:
                    self.valid_parameters = False
                    raise ValueError("The 'mutation_num_genes' parameter is expected to be a positive integer but the value ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.\n".format(mutation_num_genes_value=mutation_num_genes, mutation_num_genes_type=type(mutation_num_genes)))
            else:
                # Number of genes to mutate is adaptive not fixed.
                if type(mutation_num_genes) in [list, tuple, numpy.ndarray]:
                    if len(mutation_num_genes) == 2:
                        for el in mutation_num_genes:
                            if type(el) in GA.supported_int_types:
                                if (el <= 0):
                                    self.valid_parameters = False
                                    raise ValueError("The values assigned to the 'mutation_num_genes' cannot be <= 0 but ({mutation_num_genes_value}) found. If you do not want to use mutation, please set mutation_type=None\n".format(mutation_num_genes_value=el))
                                elif (el > self.num_genes):
                                    self.valid_parameters = False
                                    raise ValueError("The values assigned to the 'mutation_num_genes' cannot be greater than the number of genes ({num_genes}) but ({mutation_num_genes_value}) found.\n".format(mutation_num_genes_value=el, num_genes=self.num_genes))
                            else:
                                self.valid_parameters = False
                                raise ValueError("Unexpected type for a value assigned to the 'mutation_num_genes' parameter. An integer value is expected but ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.".format(mutation_num_genes_value=el, mutation_num_genes_type=type(el)))
                            # At this point of the loop, the current value assigned to the parameter 'mutation_num_genes' is validated.
                        if mutation_num_genes[0] < mutation_num_genes[1]:
                            if not self.suppress_warnings: warnings.warn("The first element in the 'mutation_num_genes' parameter is {first_el} which is smaller than the second element {second_el}. This means the mutation rate for the high-quality solutions is higher than the mutation rate of the low-quality ones. This causes high disruption in the high qualitiy solutions while making little changes in the low quality solutions. Please make the first element higher than the second element.".format(first_el=mutation_num_genes[0], second_el=mutation_num_genes[1]))
                        # At this point outside the loop, all values of the parameter 'mutation_num_genes' are validated. Eveyrthing is OK.
                    else:
                        self.valid_parameters = False
                        raise ValueError("When mutation_type='adaptive', then the 'mutation_num_genes' parameter must have only 2 elements but ({mutation_num_genes_length}) element(s) found.".format(mutation_num_genes_length=len(mutation_num_genes)))
                else:
                    self.valid_parameters = False
                    raise ValueError("Unexpected type for the 'mutation_num_genes' parameter. When mutation_type='adaptive', then list/tuple/numpy.ndarray is expected but ({mutation_num_genes_value}) of type {mutation_num_genes_type} found.".format(mutation_num_genes_value=mutation_num_genes, mutation_num_genes_type=type(mutation_num_genes)))
        else:
            pass
        
        # Validating mutation_by_replacement and mutation_type
        if self.mutation_type != "random" and self.mutation_by_replacement:
            if not self.suppress_warnings: warnings.warn("The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but ({mut_type}). Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.".format(mut_type=mutation_type))

        # Check if crossover and mutation are both disabled.
        if (self.mutation_type is None) and (self.crossover_type is None):
            if not self.suppress_warnings: warnings.warn("The 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
        if callable(parent_selection_type):
            # Check if the parent_selection_type is a function that accepts 3 paramaters.
            if (parent_selection_type.__code__.co_argcount == 3):
                # population: Added in PyGAD 2.16.0. It should used only to support custom parent selection functions. Otherwise, it should be left to None to retirve the population by self.population.
                # The parent selection function assigned to the parent_selection_type parameter is validated.
                self.select_parents = parent_selection_type
            else:
                self.valid_parameters = False
                raise ValueError("When 'parent_selection_type' is assigned to a user-defined function, then this parent selection function must accept 3 parameters:\n1) The fitness values of the current population.\n2) The number of parents needed.\n3) The instance from the pygad.GA class to retrieve any property like population, gene data type, gene space, etc.\n\nThe passed parent selection function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=parent_selection_type.__code__.co_name, argcount=parent_selection_type.__code__.co_argcount))
        elif not (type(parent_selection_type) is str):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'parent_selection_type' parameter is either callable or str but ({parent_selection_type}) found.".format(parent_selection_type=type(parent_selection_type)))
        else:
            parent_selection_type = parent_selection_type.lower()
            if (parent_selection_type == "sss"):
                self.select_parents = self.steady_state_selection
            elif (parent_selection_type == "rws"):
                self.select_parents = self.roulette_wheel_selection
            elif (parent_selection_type == "sus"):
                self.select_parents = self.stochastic_universal_selection
            elif (parent_selection_type == "random"):
                self.select_parents = self.random_selection
            elif (parent_selection_type == "tournament"):
                self.select_parents = self.tournament_selection
            elif (parent_selection_type == "rank"):
                self.select_parents = self.rank_selection
            else:
                self.valid_parameters = False
                raise ValueError("Undefined parent selection type: {parent_selection_type}. \nThe assigned value to the 'parent_selection_type' parameter does not refer to one of the supported parent selection techniques which are: \n-sss (for steady state selection)\n-rws (for roulette wheel selection)\n-sus (for stochastic universal selection)\n-rank (for rank selection)\n-random (for random selection)\n-tournament (for tournament selection).\n".format(parent_selection_type=parent_selection_type))

        # For tournament selection, validate the K value.
        if(parent_selection_type == "tournament"):
            if (K_tournament > self.sol_per_pop):
                K_tournament = self.sol_per_pop
                if not self.suppress_warnings: warnings.warn("K of the tournament selection ({K_tournament}) should not be greater than the number of solutions within the population ({sol_per_pop}).\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n".format(K_tournament=K_tournament, sol_per_pop=self.sol_per_pop))
            elif (K_tournament <= 0):
                self.valid_parameters = False
                raise ValueError("K of the tournament selection cannot be <=0 but ({K_tournament}) found.\n".format(K_tournament=K_tournament))

        self.K_tournament = K_tournament

        # Validating the number of parents to keep in the next population: keep_parents
        if (keep_parents > self.sol_per_pop or keep_parents > self.num_parents_mating or keep_parents < -1):
            self.valid_parameters = False
            raise ValueError("Incorrect value to the keep_parents parameter: {keep_parents}. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.".format(keep_parents=keep_parents))

        self.keep_parents = keep_parents
        
        if parent_selection_type == "sss" and self.keep_parents == 0:
            if not self.suppress_warnings: warnings.warn("The steady-state parent (sss) selection operator is used despite that no parents are kept in the next generation.")

        # Validate keep_parents.
        if (self.keep_parents == -1): # Keep all parents in the next population.
            self.num_offspring = self.sol_per_pop - self.num_parents_mating
        elif (self.keep_parents == 0): # Keep no parents in the next population.
            self.num_offspring = self.sol_per_pop
        elif (self.keep_parents > 0): # Keep the specified number of parents in the next population.
            self.num_offspring = self.sol_per_pop - self.keep_parents

        # Check if the fitness_func is a function.
        if callable(fitness_func):
            # Check if the fitness function accepts 2 paramaters.
            if (fitness_func.__code__.co_argcount == 2):
                self.fitness_func = fitness_func
            else:
                self.valid_parameters = False
                raise ValueError("The fitness function must accept 2 parameters:\n1) A solution to calculate its fitness value.\n2) The solution's index within the population.\n\nThe passed fitness function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=fitness_func.__code__.co_name, argcount=fitness_func.__code__.co_argcount))
        else:
            self.valid_parameters = False
            raise ValueError("The value assigned to the fitness_func parameter is expected to be of type function but ({fitness_func_type}) found.".format(fitness_func_type=type(fitness_func)))

        # Check if the on_start exists.
        if not (on_start is None):
            # Check if the on_start is a function.
            if callable(on_start):
                # Check if the on_start function accepts only a single paramater.
                if (on_start.__code__.co_argcount == 1):
                    self.on_start = on_start
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_start parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_start.__code__.co_name, argcount=on_start.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_start parameter is expected to be of type function but ({on_start_type}) found.".format(on_start_type=type(on_start)))
        else:
            self.on_start = None

        # Check if the on_fitness exists.
        if not (on_fitness is None):
            # Check if the on_fitness is a function.
            if callable(on_fitness):
                # Check if the on_fitness function accepts 2 paramaters.
                if (on_fitness.__code__.co_argcount == 2):
                    self.on_fitness = on_fitness
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_fitness parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_fitness.__code__.co_name, argcount=on_fitness.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_fitness parameter is expected to be of type function but ({on_fitness_type}) found.".format(on_fitness_type=type(on_fitness)))
        else:
            self.on_fitness = None

        # Check if the on_parents exists.
        if not (on_parents is None):
            # Check if the on_parents is a function.
            if callable(on_parents):
                # Check if the on_parents function accepts 2 paramaters.
                if (on_parents.__code__.co_argcount == 2):
                    self.on_parents = on_parents
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_parents parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_parents.__code__.co_name, argcount=on_parents.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_parents parameter is expected to be of type function but ({on_parents_type}) found.".format(on_parents_type=type(on_parents)))
        else:
            self.on_parents = None

        # Check if the on_crossover exists.
        if not (on_crossover is None):
            # Check if the on_crossover is a function.
            if callable(on_crossover):
                # Check if the on_crossover function accepts 2 paramaters.
                if (on_crossover.__code__.co_argcount == 2):
                    self.on_crossover = on_crossover
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_crossover parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring generated using crossover.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_crossover.__code__.co_name, argcount=on_crossover.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_crossover parameter is expected to be of type function but ({on_crossover_type}) found.".format(on_crossover_type=type(on_crossover)))
        else:
            self.on_crossover = None

        # Check if the on_mutation exists.
        if not (on_mutation is None):
            # Check if the on_mutation is a function.
            if callable(on_mutation):
                # Check if the on_mutation function accepts 2 paramaters.
                if (on_mutation.__code__.co_argcount == 2):
                    self.on_mutation = on_mutation
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_mutation parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring after applying the mutation operation.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_mutation.__code__.co_name, argcount=on_mutation.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_mutation parameter is expected to be of type function but ({on_mutation_type}) found.".format(on_mutation_type=type(on_mutation)))
        else:
            self.on_mutation = None

        # Check if the callback_generation exists.
        if not (callback_generation is None):
            # Check if the callback_generation is a function.
            if callable(callback_generation):
                # Check if the callback_generation function accepts only a single paramater.
                if (callback_generation.__code__.co_argcount == 1):
                    self.callback_generation = callback_generation
                    on_generation = callback_generation
                    if not self.suppress_warnings: warnings.warn("Starting from PyGAD 2.6.0, the callback_generation parameter is deprecated and will be removed in a later release of PyGAD. Please use the on_generation parameter instead.")
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the callback_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=callback_generation.__code__.co_name, argcount=callback_generation.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the callback_generation parameter is expected to be of type function but ({callback_generation_type}) found.".format(callback_generation_type=type(callback_generation)))
        else:
            self.callback_generation = None

        # Check if the on_generation exists.
        if not (on_generation is None):
            # Check if the on_generation is a function.
            if callable(on_generation):
                # Check if the on_generation function accepts only a single paramater.
                if (on_generation.__code__.co_argcount == 1):
                    self.on_generation = on_generation
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_generation.__code__.co_name, argcount=on_generation.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the on_generation parameter is expected to be of type function but ({on_generation_type}) found.".format(on_generation_type=type(on_generation)))
        else:
            self.on_generation = None

        # Check if the on_stop exists.
        if not (on_stop is None):
            # Check if the on_stop is a function.
            if callable(on_stop):
                # Check if the on_stop function accepts 2 paramaters.
                if (on_stop.__code__.co_argcount == 2):
                    self.on_stop = on_stop
                else:
                    self.valid_parameters = False
                    raise ValueError("The function assigned to the on_stop parameter must accept 2 parameters representing the instance of the genetic algorithm and a list of the fitness values of the solutions in the last population.\nThe passed function named '{funcname}' accepts {argcount} parameter(s).".format(funcname=on_stop.__code__.co_name, argcount=on_stop.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'on_stop' parameter is expected to be of type function but ({on_stop_type}) found.".format(on_stop_type=type(on_stop)))
        else:
            self.on_stop = None

        # Validate delay_after_gen
        if type(delay_after_gen) in GA.supported_int_float_types:
            if delay_after_gen >= 0.0:
                self.delay_after_gen = delay_after_gen
            else:
                self.valid_parameters = False
                raise ValueError("The value passed to the 'delay_after_gen' parameter must be a non-negative number. The value passed is {delay_after_gen} of type {delay_after_gen_type}.".format(delay_after_gen=delay_after_gen, delay_after_gen_type=type(delay_after_gen)))
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'delay_after_gen' parameter must be of type int or float but ({delay_after_gen_type}) found.".format(delay_after_gen_type=type(delay_after_gen)))

        # Validate save_best_solutions
        if type(save_best_solutions) is bool:
            if save_best_solutions == True:
                if not self.suppress_warnings: warnings.warn("Use the 'save_best_solutions' parameter with caution as it may cause memory overflow when either the number of generations or number of genes is large.")
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'save_best_solutions' parameter must be of type bool but ({save_best_solutions_type}) found.".format(save_best_solutions_type=type(save_best_solutions)))

        # Validate save_solutions
        if type(save_solutions) is bool:
            if save_solutions == True:
                if not self.suppress_warnings: warnings.warn("Use the 'save_solutions' parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.")
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'save_solutions' parameter must be of type bool but ({save_solutions_type}) found.".format(save_solutions_type=type(save_solutions)))

        # Validate allow_duplicate_genes
        if not (type(allow_duplicate_genes) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'allow_duplicate_genes' parameter is bool but ({allow_duplicate_genes_type}) found.".format(allow_duplicate_genes_type=type(allow_duplicate_genes)))

        self.allow_duplicate_genes = allow_duplicate_genes

        self.stop_criteria = []
        self.supported_stop_words = ["reach", "saturate"]
        if stop_criteria is None:
            # None: Stop after passing through all generations.
            self.stop_criteria = None
        elif type(stop_criteria) is str:
            # reach_{target_fitness}: Stop if the target fitness value is reached.
            # saturate_{num_generations}: Stop if the fitness value does not change (saturates) for the given number of generations.
            criterion = stop_criteria.split("_")
            if len(criterion) == 2:
                stop_word = criterion[0]
                number = criterion[1]

                if stop_word in self.supported_stop_words:
                    pass
                else:
                    self.valid_parameters = False
                    raise TypeError("In the 'stop_criteria' parameter, the supported stop words are '{supported_stop_words}' but '{stop_word}' found.".format(supported_stop_words=self.supported_stop_words, stop_word=stop_word))

                if number.replace(".", "").isnumeric():
                    number = float(number)
                else:
                    self.valid_parameters = False
                    raise TypeError("The value following the stop word in the 'stop_criteria' parameter must be a number but the value '{stop_val}' of type {stop_val_type} found.".format(stop_val=number, stop_val_type=type(number)))
                
                self.stop_criteria.append([stop_word, number])

            else:
                self.valid_parameters = False
                raise TypeError("For format of a single criterion in the 'stop_criteria' parameter is 'word_number' but '{stop_criteria}' found.".format(stop_criteria=stop_criteria))

        elif type(stop_criteria) in [list, tuple, numpy.ndarray]:
            # Remove duplicate criterira by converting the list to a set then back to a list.
            stop_criteria = list(set(stop_criteria))
            for idx, val in enumerate(stop_criteria):
                if type(val) is str:
                    criterion = val.split("_")
                    if len(criterion) == 2:
                        stop_word = criterion[0]
                        number = criterion[1]

                        if stop_word in self.supported_stop_words:
                            pass
                        else:
                            self.valid_parameters = False
                            raise TypeError("In the 'stop_criteria' parameter, the supported stop words are {supported_stop_words} but '{stop_word}' found.".format(supported_stop_words=self.supported_stop_words, stop_word=stop_word))

                        if number.replace(".", "").isnumeric():
                            number = float(number)
                        else:
                            self.valid_parameters = False
                            raise TypeError("The value following the stop word in the 'stop_criteria' parameter must be a number but the value '{stop_val}' of type {stop_val_type} found.".format(stop_val=number, stop_val_type=type(number)))

                        self.stop_criteria.append([stop_word, number])

                    else:
                        self.valid_parameters = False
                        raise TypeError("For format of a single criterion in the 'stop_criteria' parameter is 'word_number' but {stop_criteria} found.".format(stop_criteria=criterion))
                else:
                    self.valid_parameters = False
                    raise TypeError("When the 'stop_criteria' parameter is assigned a tuple/list/numpy.ndarray, then its elements must be strings but the value '{stop_criteria_val}' of type {stop_criteria_val_type} found at index {stop_criteria_val_idx}.".format(stop_criteria_val=val, stop_criteria_val_type=type(val), stop_criteria_val_idx=idx))
        else:
            self.valid_parameters = False
            raise TypeError("The expected value of the 'stop_criteria' is a single string or a list/tuple/numpy.ndarray of strings but the value {stop_criteria_val} of type {stop_criteria_type} found.".format(stop_criteria_val=stop_criteria, stop_criteria_type=type(stop_criteria)))

        # The number of completed generations.
        self.generations_completed = 0

        # At this point, all necessary parameters validation is done successfully and we are sure that the parameters are valid.
        self.valid_parameters = True # Set to True when all the parameters passed in the GA class constructor are valid.

        # Parameters of the genetic algorithm.
        self.num_generations = abs(num_generations)
        self.parent_selection_type = parent_selection_type

        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.

        self.best_solution_generation = -1 # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.

        self.save_best_solutions = save_best_solutions
        self.best_solutions = [] # Holds the best solution in each generation.

        self.save_solutions = save_solutions
        self.solutions = [] # Holds the solutions in each generation.
        self.solutions_fitness = [] # Holds the fitness of the solutions in each generation.

        self.last_generation_fitness = None # A list holding the fitness values of all solutions in the last generation.
        self.last_generation_parents = None # A list holding the parents of the last generation.
        self.last_generation_offspring_crossover = None # A list holding the offspring after applying crossover in the last generation.
        self.last_generation_offspring_mutation = None # A list holding the offspring after applying mutation in the last generation.
        self.previous_generation_fitness = None # Holds the fitness values of one generation before the fitness values saved in the last_generation_fitness attribute. Added in PyGAD 2.26.2

    def round_genes(self, solutions):
        for gene_idx in range(self.num_genes):
            if self.gene_type_single:
                if not self.gene_type[1] is None:
                    solutions[:, gene_idx] = numpy.round(solutions[:, gene_idx], self.gene_type[1])
            else:
                if not self.gene_type[gene_idx][1] is None:
                    solutions[:, gene_idx] = numpy.round(numpy.asarray(solutions[:, gene_idx], 
                                                                       dtype=self.gene_type[gene_idx][0]), 
                                                         self.gene_type[gene_idx][1])
        return solutions

    def initialize_population(self, low, high, allow_duplicate_genes, mutation_by_replacement, gene_type):

        """
        Creates an initial population randomly as a NumPy array. The array is saved in the instance attribute named 'population'.

        low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.

        This method assigns the values of the following 3 instance attributes:
            1. pop_size: Size of the population.
            2. population: Initially, holds the initial population and later updated after each generation.
            3. init_population: Keeping the initial population.
        """

        # Population size = (number of chromosomes, number of genes per chromosome)
        self.pop_size = (self.sol_per_pop,self.num_genes) # The population will have sol_per_pop chromosome where each chromosome has num_genes genes.

        if self.gene_space is None:
            # Creating the initial population randomly.
            if self.gene_type_single == True:
                self.population = numpy.asarray(numpy.random.uniform(low=low, 
                                                                     high=high, 
                                                                     size=self.pop_size), 
                                                dtype=self.gene_type[0]) # A NumPy array holding the initial population.
            else:
                # Create an empty population of dtype=object to support storing mixed data types within the same array.
                self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                # Loop through the genes, randomly generate the values of a single gene across the entire population, and add the values of each gene to the population.
                for gene_idx in range(self.num_genes):
                    # A vector of all values of this single gene across all solutions in the population.
                    gene_values = numpy.asarray(numpy.random.uniform(low=low, 
                                                                     high=high, 
                                                                     size=self.pop_size[0]), 
                                                dtype=self.gene_type[gene_idx][0])
                    # Adding the current gene values to the population.
                    self.population[:, gene_idx] = gene_values

            if allow_duplicate_genes == False:
                for solution_idx in range(self.population.shape[0]):
                    # print("Before", self.population[solution_idx])
                    self.population[solution_idx], _, _ = self.solve_duplicate_genes_randomly(solution=self.population[solution_idx],
                                                                                              min_val=low, 
                                                                                              max_val=high,
                                                                                              mutation_by_replacement=True,
                                                                                              gene_type=gene_type,
                                                                                              num_trials=10)
                    # print("After", self.population[solution_idx])

        elif self.gene_space_nested:
            if self.gene_type_single == True:
                self.population = numpy.zeros(shape=self.pop_size, dtype=self.gene_type[0])
                for sol_idx in range(self.sol_per_pop):
                    for gene_idx in range(self.num_genes):
                        if type(self.gene_space[gene_idx]) in [list, tuple, range]:
                            # Check if the gene space has None values. If any, then replace it with randomly generated values according to the 3 attributes init_range_low, init_range_high, and gene_type.
                            if type(self.gene_space[gene_idx]) is range:
                                temp = self.gene_space[gene_idx]
                            else:
                                temp = self.gene_space[gene_idx].copy()
                            for idx, val in enumerate(self.gene_space[gene_idx]):
                                if val is None:
                                    self.gene_space[gene_idx][idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                                                                                        high=high, 
                                                                                                        size=1), 
                                                                                   dtype=self.gene_type[0])[0]
                            self.population[sol_idx, gene_idx] = random.choice(self.gene_space[gene_idx])
                            self.population[sol_idx, gene_idx] = self.gene_type[0](self.population[sol_idx, gene_idx])
                            self.gene_space[gene_idx] = temp
                        elif type(self.gene_space[gene_idx]) is dict:
                            if 'step' in self.gene_space[gene_idx].keys():
                                self.population[sol_idx, gene_idx] = numpy.asarray(numpy.random.choice(numpy.arange(start=self.gene_space[gene_idx]['low'],
                                                                                                                    stop=self.gene_space[gene_idx]['high'],
                                                                                                                    step=self.gene_space[gene_idx]['step']),
                                                                                                       size=1),
                                                                                   dtype=self.gene_type[0])[0]
                            else:
                                self.population[sol_idx, gene_idx] = numpy.asarray(numpy.random.uniform(low=self.gene_space[gene_idx]['low'],
                                                                                                        high=self.gene_space[gene_idx]['high'],
                                                                                                        size=1),
                                                                                   dtype=self.gene_type[0])[0]
                        elif type(self.gene_space[gene_idx]) == type(None):

                            # The following commented code replace the None value with a single number that will not change again. 
                            # This means the gene value will be the same across all solutions.
                            # self.gene_space[gene_idx] = numpy.asarray(numpy.random.uniform(low=low,
                            #                high=high, 
                            #                size=1), dtype=self.gene_type[0])[0]
                            # self.population[sol_idx, gene_idx] = self.gene_space[gene_idx].copy()
                            
                            # The above problem is solved by keeping the None value in the gene_space parameter. This forces PyGAD to generate this value for each solution.
                            self.population[sol_idx, gene_idx] = numpy.asarray(numpy.random.uniform(low=low,
                                                                                                    high=high, 
                                                                                                    size=1), 
                                                                               dtype=self.gene_type[0])[0]
                        elif type(self.gene_space[gene_idx]) in GA.supported_int_float_types:
                            self.population[sol_idx, gene_idx] = self.gene_space[gene_idx].copy()
            else:
                self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                for sol_idx in range(self.sol_per_pop):
                    for gene_idx in range(self.num_genes):
                        if type(self.gene_space[gene_idx]) in [list, tuple, range]:
                            # Check if the gene space has None values. If any, then replace it with randomly generated values according to the 3 attributes init_range_low, init_range_high, and gene_type.
                            temp = self.gene_space[gene_idx].copy()
                            for idx, val in enumerate(self.gene_space[gene_idx]):
                                if val is None:
                                    self.gene_space[gene_idx][idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                                                                                        high=high, 
                                                                                                        size=1), 
                                                                                   dtype=self.gene_type[gene_idx][0])[0]
                            self.population[sol_idx, gene_idx] = random.choice(self.gene_space[gene_idx])
                            self.population[sol_idx, gene_idx] = self.gene_type[gene_idx][0](self.population[sol_idx, gene_idx])
                            self.gene_space[gene_idx] = temp.copy()
                        elif type(self.gene_space[gene_idx]) is dict:
                            if 'step' in self.gene_space[gene_idx].keys():
                                self.population[sol_idx, gene_idx] = numpy.asarray(numpy.random.choice(numpy.arange(start=self.gene_space[gene_idx]['low'],
                                                                                                                    stop=self.gene_space[gene_idx]['high'],
                                                                                                                    step=self.gene_space[gene_idx]['step']),
                                                                                                       size=1),
                                                                                   dtype=self.gene_type[gene_idx][0])[0]
                            else:
                                self.population[sol_idx, gene_idx] = numpy.asarray(numpy.random.uniform(low=self.gene_space[gene_idx]['low'],
                                                                                                        high=self.gene_space[gene_idx]['high'],
                                                                                                        size=1), 
                                                                                   dtype=self.gene_type[gene_idx][0])[0]
                        elif type(self.gene_space[gene_idx]) == type(None):
                            # self.gene_space[gene_idx] = numpy.asarray(numpy.random.uniform(low=low,
                            #                                                                high=high, 
                            #                                                                size=1), 
                            #                                           dtype=self.gene_type[gene_idx][0])[0]

                            # self.population[sol_idx, gene_idx] = self.gene_space[gene_idx].copy()

                            temp = numpy.asarray(numpy.random.uniform(low=low,
                                                                      high=high, 
                                                                      size=1), 
                                                 dtype=self.gene_type[gene_idx][0])[0]
                            self.population[sol_idx, gene_idx] = temp
                        elif type(self.gene_space[gene_idx]) in GA.supported_int_float_types:
                            self.population[sol_idx, gene_idx] = self.gene_space[gene_idx]
        else:
            if self.gene_type_single == True:
                # Replace all the None values with random values using the init_range_low, init_range_high, and gene_type attributes.
                for idx, curr_gene_space in enumerate(self.gene_space):
                    if curr_gene_space is None:
                        self.gene_space[idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                                                                  high=high, 
                                                                                  size=1), 
                                                             dtype=self.gene_type[0])[0]
    
                # Creating the initial population by randomly selecting the genes' values from the values inside the 'gene_space' parameter.
                if type(self.gene_space) is dict:
                    if 'step' in self.gene_space.keys():
                        self.population = numpy.asarray(numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                         stop=self.gene_space['high'],
                                                                                         step=self.gene_space['step']),
                                                                            size=self.pop_size),
                                                        dtype=self.gene_type[0])
                    else:
                        self.population = numpy.asarray(numpy.random.uniform(low=self.gene_space['low'],
                                                                             high=self.gene_space['high'],
                                                                             size=self.pop_size),
                                                        dtype=self.gene_type[0]) # A NumPy array holding the initial population.
                else:
                    self.population = numpy.asarray(numpy.random.choice(self.gene_space,
                                                                        size=self.pop_size),
                                                    dtype=self.gene_type[0]) # A NumPy array holding the initial population.
            else:
                # Replace all the None values with random values using the init_range_low, init_range_high, and gene_type attributes.
                for gene_idx, curr_gene_space in enumerate(self.gene_space):
                    if curr_gene_space is None:
                        self.gene_space[gene_idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                                                                  high=high, 
                                                                                  size=1), 
                                                             dtype=self.gene_type[gene_idx][0])[0]
    
                # Creating the initial population by randomly selecting the genes' values from the values inside the 'gene_space' parameter.
                if type(self.gene_space) is dict:
                    # Create an empty population of dtype=object to support storing mixed data types within the same array.
                    self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                    # Loop through the genes, randomly generate the values of a single gene across the entire population, and add the values of each gene to the population.
                    for gene_idx in range(self.num_genes):
                        # A vector of all values of this single gene across all solutions in the population.
                        if 'step' in self.gene_space[gene_idx].keys():
                            gene_values = numpy.asarray(numpy.random.choice(numpy.arange(start=self.gene_space[gene_idx]['low'],
                                                                                         stop=self.gene_space[gene_idx]['high'],
                                                                                         step=self.gene_space[gene_idx]['step']),
                                                                            size=self.pop_size[0]),
                                                        dtype=self.gene_type[gene_idx][0])
                        else:
                            gene_values = numpy.asarray(numpy.random.uniform(low=self.gene_space['low'], 
                                                                             high=self.gene_space['high'], 
                                                                             size=self.pop_size[0]), 
                                                        dtype=self.gene_type[gene_idx][0])
                        # Adding the current gene values to the population.
                        self.population[:, gene_idx] = gene_values
        
                else:
                    # Create an empty population of dtype=object to support storing mixed data types within the same array.
                    self.population = numpy.zeros(shape=self.pop_size, dtype=object)
                    # Loop through the genes, randomly generate the values of a single gene across the entire population, and add the values of each gene to the population.
                    for gene_idx in range(self.num_genes):
                        # A vector of all values of this single gene across all solutions in the population.
                        gene_values = numpy.asarray(numpy.random.choice(self.gene_space, 
                                                                        size=self.pop_size[0]), 
                                                    dtype=self.gene_type[gene_idx][0])
                        # Adding the current gene values to the population.
                        self.population[:, gene_idx] = gene_values

        if not (self.gene_space is None):
            if allow_duplicate_genes == False:
                for sol_idx in range(self.population.shape[0]):
                    self.population[sol_idx], _, _ = self.solve_duplicate_genes_by_space(solution=self.population[sol_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10,
                                                                                         build_initial_pop=True)

        # Keeping the initial population in the initial_population attribute.
        self.initial_population = self.population.copy()

    def cal_pop_fitness(self):

        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
        """

        if self.valid_parameters == False:
            raise ValueError("ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol_idx, sol in enumerate(self.population):

            # Check if this solution is a parent from the previous generation and its fitness value is already calculated. If so, use the fitness value instead of calling the fitness function.
            if (self.last_generation_parents is not None) and len(numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0] > 0):
                # Index of the parent in the parents array (self.last_generation_parents). This is not its index within the population.
                parent_idx = numpy.where(numpy.all(self.last_generation_parents == sol, axis=1))[0][0]
                # Index of the parent in the population.
                parent_idx = self.last_generation_parents_indices[parent_idx]
                # Use the parent's index to return its pre-calculated fitness value.
                fitness = self.previous_generation_fitness[parent_idx]
            else:
                fitness = self.fitness_func(sol, sol_idx)
                if type(fitness) in GA.supported_int_float_types:
                    pass
                else:
                    raise ValueError("The fitness function should return a number but the value {fit_val} of type {fit_type} found.".format(fit_val=fitness, fit_type=type(fitness)))
            pop_fitness.append(fitness)

        pop_fitness = numpy.array(pop_fitness)

        return pop_fitness

    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        if self.valid_parameters == False:
            raise ValueError("Error calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        # Reset the variables that store the solutions and their fitness after each generation. If not reset, then for each call to the run() method the new solutions and their fitness values will be appended to the old variables and their length double. Some errors arise if not reset.
        # If, in the future, new variables are created that get appended after each generation, please consider resetting them here.
        self.best_solutions = [] # Holds the best solution in each generation.
        self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.
        self.solutions = [] # Holds the solutions in each generation.
        self.solutions_fitness = [] # Holds the fitness of the solutions in each generation.

        if not (self.on_start is None):
            self.on_start(self)

        stop_run = False

        # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
        self.last_generation_fitness = self.cal_pop_fitness()

        best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

        # Appending the best solution in the initial population to the best_solutions list.
        if self.save_best_solutions:
            self.best_solutions.append(best_solution)

        # Appending the solutions in the initial population to the solutions list.
        if self.save_solutions:
            self.solutions.extend(self.population.copy())

        for generation in range(self.num_generations):
            if not (self.on_fitness is None):
                self.on_fitness(self, self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)
            
            if self.save_solutions:
                self.solutions_fitness.extend(self.last_generation_fitness)

            # Selecting the best parents in the population for mating.
            if callable(self.parent_selection_type):
                self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(self.last_generation_fitness, self.num_parents_mating, self)
            else:
                self.last_generation_parents, self.last_generation_parents_indices = self.select_parents(self.last_generation_fitness, num_parents=self.num_parents_mating)
            if not (self.on_parents is None):
                self.on_parents(self, self.last_generation_parents)

            # If self.crossover_type=None, then no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
            if self.crossover_type is None:
                if self.num_offspring <= self.keep_parents:
                    self.last_generation_offspring_crossover = self.last_generation_parents[0:self.num_offspring]
                else:
                    self.last_generation_offspring_crossover = numpy.concatenate((self.last_generation_parents, self.population[0:(self.num_offspring - self.last_generation_parents.shape[0])]))
            else:
                # Generating offspring using crossover.
                if callable(self.crossover_type):
                    self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                              (self.num_offspring, self.num_genes),
                                                                              self)
                else:
                    self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                                              offspring_size=(self.num_offspring, self.num_genes))
                if not (self.on_crossover is None):
                    self.on_crossover(self, self.last_generation_offspring_crossover)

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                self.last_generation_offspring_mutation = self.last_generation_offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                if callable(self.mutation_type):
                    self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover, self)
                else:
                    self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)
                if not (self.on_mutation is None):
                    self.on_mutation(self, self.last_generation_offspring_mutation)

            # Update the population attribute according to the offspring generated.
            if (self.keep_parents == 0):
                self.population = self.last_generation_offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:self.last_generation_parents.shape[0], :] = self.last_generation_parents
                self.population[self.last_generation_parents.shape[0]:, :] = self.last_generation_offspring_mutation
            elif (self.keep_parents > 0):
                parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = self.last_generation_offspring_mutation

            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            self.previous_generation_fitness = self.last_generation_fitness.copy()
            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

            # Appending the best solution in the current generation to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            # Appending the solutions in the current generation to the solutions list.
            if self.save_solutions:
                self.solutions.extend(self.population.copy())

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            if not (self.on_generation is None):
                r = self.on_generation(self)
                if type(r) is str and r.lower() == "stop":
                    # Before aborting the loop, save the fitness value of the best solution.
                    _, best_solution_fitness, _ = self.best_solution()
                    self.best_solutions_fitness.append(best_solution_fitness)
                    break

            if not self.stop_criteria is None:
                for criterion in self.stop_criteria:
                    if criterion[0] == "reach":
                        if max(self.last_generation_fitness) >= criterion[1]:
                            stop_run = True
                            break
                    elif criterion[0] == "saturate":
                        criterion[1] = int(criterion[1])
                        if (self.generations_completed >= criterion[1]):
                            if (self.best_solutions_fitness[self.generations_completed - criterion[1]] - self.best_solutions_fitness[self.generations_completed - 1]) == 0:
                                stop_run = True
                                break

            if stop_run:
                break

            time.sleep(self.delay_after_gen)

        # Save the fitness of the last generation.
        if self.save_solutions:
            self.solutions_fitness.extend(self.last_generation_fitness)

        # Save the fitness value of the best solution.
        _, best_solution_fitness, _ = self.best_solution(pop_fitness=self.last_generation_fitness)
        self.best_solutions_fitness.append(best_solution_fitness)

        self.best_solution_generation = numpy.where(numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness)))[0][0]
        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True # Set to True only after the run() method completes gracefully.

        if not (self.on_stop is None):
            self.on_stop(self, self.last_generation_fitness)

        # Converting the 'best_solutions' list into a NumPy array.
        self.best_solutions = numpy.array(self.best_solutions)

        # Converting the 'solutions' list into a NumPy array.
        self.solutions = numpy.array(self.solutions)

    def steady_state_selection(self, fitness, num_parents):

        """
        Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
        
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()

        return parents, fitness_sorted[:num_parents]

    def rank_selection(self, fitness, num_parents):

        """
        Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()

        return parents, fitness_sorted[:num_parents]

    def random_selection(self, fitness, num_parents):

        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :].copy()

        return parents, rand_indices

    def tournament_selection(self, fitness, num_parents):

        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        parents_indices = []

        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents_indices.append(selected_parent_idx)
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :].copy()

        return parents, parents_indices

    def roulette_wheel_selection(self, fitness, num_parents):

        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        
        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    def stochastic_universal_selection(self, fitness, num_parents):

        """
        Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        pointers_distance = 1.0 / self.num_parents_mating # Distance between different pointers.
        first_pointer = numpy.random.uniform(low=0.0, high=pointers_distance, size=1) # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
        
        parents_indices = []

        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num*pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    parents_indices.append(idx)
                    break
        return parents, parents_indices

    def single_point_crossover(self, parents, offspring_size):

        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            # The point at which crossover takes place between two parents. Usually, it is at the center.
            crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]

            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def two_points_crossover(self, parents, offspring_size):

        """
        Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
                crossover_point1 = 0
            else:
                crossover_point1 = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1]/2 + 1), size=1)[0]
    
            crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
            # The genes between the 2 points are copied from the second parent.
            offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
        return offspring

    def uniform_crossover(self, parents, offspring_size):

        """
        Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            genes_source = numpy.random.randint(low=0, high=2, size=offspring_size[1])
            for gene_idx in range(offspring_size[1]):
                if (genes_source[gene_idx] == 0):
                    # The gene will be copied from the first parent if the current gene index is 0.
                    offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
                elif (genes_source[gene_idx] == 1):
                    # The gene will be copied from the second parent if the current gene index is 1.
                    offspring[k, gene_idx] = parents[parent2_idx, gene_idx]
        return offspring

    def scattered_crossover(self, parents, offspring_size):

        """
        Applies the scattered crossover. It randomly selects the gene from one of the 2 parents. 
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(set(indices), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # A 0/1 vector where 0 means the gene is taken from the first parent and 1 means the gene is taken from the second parent.
            gene_sources = numpy.random.randint(0, 2, size=self.num_genes)
            offspring[k, :] = numpy.where(gene_sources == 0, parents[parent1_idx, :], parents[parent2_idx, :])

        return offspring

    def random_mutation(self, offspring):

        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.mutation_by_space(offspring)
            else:
                offspring = self.mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.mutation_probs_by_space(offspring)
            else:
                offspring = self.mutation_probs_randomly(offspring)

        return offspring

    def mutation_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:

                if self.gene_space_nested:
                    # Returning the current gene space from the 'gene_space' attribute.
                    if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                        curr_gene_space = self.gene_space[gene_idx].copy()
                    else:
                        curr_gene_space = self.gene_space[gene_idx]

                    # If the gene space has only a single value, use it as the new gene value.
                    if type(curr_gene_space) in GA.supported_int_float_types:
                        value_from_space = curr_gene_space
                    # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                    elif curr_gene_space is None:
                        rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                        high=self.random_mutation_max_val,
                                                        size=1)
                        if self.mutation_by_replacement:
                            value_from_space = rand_val
                        else:
                            value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                    elif type(curr_gene_space) is dict:
                        # The gene's space of type dict specifies the lower and upper limits of a gene.
                        if 'step' in curr_gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                stop=curr_gene_space['high'],
                                                                                step=curr_gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                    else:
                        # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                        # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                        if len(curr_gene_space) == 1:
                            value_from_space = curr_gene_space[0]
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        else:
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                    else:
                        # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        if len(values_to_select_from) == 0:
                            value_from_space = offspring[offspring_idx, gene_idx]
                        else:
                            value_from_space = random.choice(values_to_select_from)
                    # value_from_space = random.choice(self.gene_space)

                if value_from_space is None:
                    value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                            high=self.random_mutation_max_val, 
                                                            size=1)

                # Assinging the selected value from the space to the gene.
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                         self.gene_type[1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                         self.gene_type[gene_idx][1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring

    def mutation_probs_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space and the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated based on the mutation space.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    if self.gene_space_nested:
                        # Returning the current gene space from the 'gene_space' attribute.
                        if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                            curr_gene_space = self.gene_space[gene_idx].copy()
                        else:
                            curr_gene_space = self.gene_space[gene_idx]
        
                        # If the gene space has only a single value, use it as the new gene value.
                        if type(curr_gene_space) in GA.supported_int_float_types:
                            value_from_space = curr_gene_space
                        # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                        elif curr_gene_space is None:
                            rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                            high=self.random_mutation_max_val,
                                                            size=1)
                            if self.mutation_by_replacement:
                                value_from_space = rand_val
                            else:
                                value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                        elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                            if len(curr_gene_space) == 1:
                                value_from_space = curr_gene_space[0]
                            # If the gene space has more than 1 value, then select a new one that is different from the current value.
                            else:
                                values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                                if len(values_to_select_from) == 0:
                                    value_from_space = offspring[offspring_idx, gene_idx]
                                else:
                                    value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            if 'step' in self.gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                    stop=self.gene_space['high'],
                                                                                    step=self.gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                        high=self.gene_space['high'],
                                                                        size=1)
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)

                    # Assigning the selected value from the space to the gene.
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                             self.gene_type[1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                             self.gene_type[gene_idx][1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def mutation_randomly(self, offspring):

        """
        Applies the random mutation the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:
                # Generating a random value.
                random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
                                                    size=1)
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]
               # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]

                # Round the gene
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        random_value = numpy.round(random_value, self.gene_type[1])
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=self.random_mutation_min_val,
                                                                                         max_val=self.random_mutation_max_val,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)

        return offspring

    def mutation_probs_randomly(self, offspring):

        """
        Applies the random mutation using the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more gene in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                        high=self.random_mutation_max_val, 
                                                        size=1)
                    # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                    if self.mutation_by_replacement:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]

                    # Round the gene
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            random_value = numpy.round(random_value, self.gene_type[1])
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=self.random_mutation_min_val,
                                                                                             max_val=self.random_mutation_max_val,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def swap_mutation(self, offspring):

        """
        Applies the swap mutation which interchanges the values of 2 randomly selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=offspring.shape[1]/2, size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            temp = offspring[idx, mutation_gene1]
            offspring[idx, mutation_gene1] = offspring[idx, mutation_gene2]
            offspring[idx, mutation_gene2] = temp
        return offspring

    def inversion_mutation(self, offspring):

        """
        Applies the inversion mutation which selects a subset of genes and inverts them (in order).
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            genes_to_scramble = numpy.flip(offspring[idx, mutation_gene1:mutation_gene2])
            offspring[idx, mutation_gene1:mutation_gene2] = genes_to_scramble
        return offspring

    def scramble_mutation(self, offspring):

        """
        Applies the scramble mutation which selects a subset of genes and shuffles their order randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)
            genes_range = numpy.arange(start=mutation_gene1, stop=mutation_gene2)
            numpy.random.shuffle(genes_range)
            
            genes_to_scramble = numpy.flip(offspring[idx, genes_range])
            offspring[idx, genes_range] = genes_to_scramble
        return offspring

    def adaptive_mutation_population_fitness(self, offspring):

        """
        A helper method to calculate the average fitness of the solutions before applying the adaptive mutation.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns the average fitness to be used in adaptive mutation.
        """        

        fitness = self.last_generation_fitness.copy()
        temp_population = numpy.zeros_like(self.population)

        if (self.keep_parents == 0):
            parents_to_keep = []
        elif (self.keep_parents == -1):
            parents_to_keep = self.last_generation_parents.copy()
            temp_population[0:len(parents_to_keep), :] = parents_to_keep
        elif (self.keep_parents > 0):
            parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_parents)
            temp_population[0:len(parents_to_keep), :] = parents_to_keep

        temp_population[len(parents_to_keep):, :] = offspring

        fitness[:self.last_generation_parents.shape[0]] = self.last_generation_fitness[self.last_generation_parents_indices]

        for idx in range(len(parents_to_keep), fitness.shape[0]):
            fitness[idx] = self.fitness_func(temp_population[idx], None)
        average_fitness = numpy.mean(fitness)

        return average_fitness, fitness[len(parents_to_keep):]

    def adaptive_mutation(self, offspring):

        """
        Applies the adaptive mutation which changes the values of a number of genes randomly. In adaptive mutation, the number of genes to mutate differs based on the fitness value of the solution.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the attribute 'gene_space' exists (i.e. not None), then the mutation values are selected from the 'gene_space' parameter according to the space of values of each gene. Otherwise, it is selected randomly based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_probs_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_probs_randomly(offspring)

        return offspring

    def adaptive_mutation_by_space(self, offspring):

        """
        Applies the adaptive mutation based on the 2 parameters 'mutation_num_genes' and 'gene_space'. 
        A number of genes equal are selected randomly for mutation. This number depends on the fitness of the solution.
        The random values are selected from the 'gene_space' parameter.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        
        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_num_genes = self.mutation_num_genes[0]
            else:
                adaptive_mutation_num_genes = self.mutation_num_genes[1]
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:

                if self.gene_space_nested:
                    # Returning the current gene space from the 'gene_space' attribute.
                    if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                        curr_gene_space = self.gene_space[gene_idx].copy()
                    else:
                        curr_gene_space = self.gene_space[gene_idx]

                    # If the gene space has only a single value, use it as the new gene value.
                    if type(curr_gene_space) in GA.supported_int_float_types:
                        value_from_space = curr_gene_space
                    # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                    elif curr_gene_space is None:
                        rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                        high=self.random_mutation_max_val,
                                                        size=1)
                        if self.mutation_by_replacement:
                            value_from_space = rand_val
                        else:
                            value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                    elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)
                    else:
                        # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                        # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                        if len(curr_gene_space) == 1:
                            value_from_space = curr_gene_space[0]
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        else:
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                    else:
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        if len(values_to_select_from) == 0:
                            value_from_space = offspring[offspring_idx, gene_idx]
                        else:
                            value_from_space = random.choice(values_to_select_from)


                if value_from_space is None:
                    value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                            high=self.random_mutation_max_val, 
                                                            size=1)

                # Assinging the selected value from the space to the gene.
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                         self.gene_type[1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                         self.gene_type[gene_idx][1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring
        
    def adaptive_mutation_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_num_genes' parameter. 
        A number of genes equal are selected randomly for mutation. This number depends on the fitness of the solution.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_num_genes = self.mutation_num_genes[0]
            else:
                adaptive_mutation_num_genes = self.mutation_num_genes[1]
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:
                # Generating a random value.
                random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
                                                    size=1)
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]
                # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]

                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        random_value = numpy.round(random_value, self.gene_type[1])
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=self.random_mutation_min_val,
                                                                                         max_val=self.random_mutation_max_val,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring

    def adaptive_mutation_probs_by_space(self, offspring):

        """
        Applies the adaptive mutation based on the 2 parameters 'mutation_probability' and 'gene_space'.
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on space of values for each gene.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_probability = self.mutation_probability[0]
            else:
                adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= adaptive_mutation_probability:
                    if self.gene_space_nested:
                        # Returning the current gene space from the 'gene_space' attribute.
                        if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                            curr_gene_space = self.gene_space[gene_idx].copy()
                        else:
                            curr_gene_space = self.gene_space[gene_idx]
        
                        # If the gene space has only a single value, use it as the new gene value.
                        if type(curr_gene_space) in GA.supported_int_float_types:
                            value_from_space = curr_gene_space
                        # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                        elif curr_gene_space is None:
                            rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                            high=self.random_mutation_max_val,
                                                            size=1)
                            if self.mutation_by_replacement:
                                value_from_space = rand_val
                            else:
                                value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                        elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                            if len(curr_gene_space) == 1:
                                value_from_space = curr_gene_space[0]
                            # If the gene space has more than 1 value, then select a new one that is different from the current value.
                            else:
                                values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                                if len(values_to_select_from) == 0:
                                    value_from_space = offspring[offspring_idx, gene_idx]
                                else:
                                    value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            if 'step' in self.gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                    stop=self.gene_space['high'],
                                                                                    step=self.gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                        high=self.gene_space['high'],
                                                                        size=1)
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)

                    if value_from_space is None:
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                                high=self.random_mutation_max_val, 
                                                                size=1)

                    # Assinging the selected value from the space to the gene.
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                             self.gene_type[1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                             self.gene_type[gene_idx][1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring
    
    def adaptive_mutation_probs_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_probability' parameter. 
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_probability = self.mutation_probability[0]
            else:
                adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= adaptive_mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                        high=self.random_mutation_max_val, 
                                                        size=1)
                    # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                    if self.mutation_by_replacement:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]

                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            random_value = numpy.round(random_value, self.gene_type[1])
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=self.random_mutation_min_val,
                                                                                             max_val=self.random_mutation_max_val,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def solve_duplicate_genes_randomly(self, solution, min_val, max_val, mutation_by_replacement, gene_type, num_trials=10):

        """
        Solves the duplicates in a solution by randomly selecting new values for the duplicating genes.
        
        solution: A solution with duplicate values.
        min_val: Minimum value of the range to sample a number randomly.
        max_val: Maximum value of the range to sample a number randomly.
        mutation_by_replacement: Identical to the self.mutation_by_replacement attribute.
        gene_type: Exactly the same as the self.gene_type attribute.
        num_trials: Maximum number of trials to change the gene value to solve the duplicates.

        Returns:
            new_solution: Solution after trying to solve its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
            not_unique_indices: Indices of the genes with duplicate values.
            num_unsolved_duplicates: Number of unsolved duplicates.
        """

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)

        num_unsolved_duplicates = 0
        if len(not_unique_indices) > 0:
            for duplicate_index in not_unique_indices:
                for trial_index in range(num_trials):
                    if self.gene_type_single == True:
                        if gene_type[0] in GA.supported_int_types:
                            temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                       gene_index=duplicate_index, 
                                                                       min_val=min_val, 
                                                                       max_val=max_val, 
                                                                       mutation_by_replacement=mutation_by_replacement, 
                                                                       gene_type=gene_type)
                        else:
                            temp_val = numpy.random.uniform(low=min_val,
                                                            high=max_val,
                                                            size=1)
                            if mutation_by_replacement:
                                pass
                            else:
                                temp_val = new_solution[duplicate_index] + temp_val
                    else:
                        if gene_type[duplicate_index] in GA.supported_int_types:
                            temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                       gene_index=duplicate_index, 
                                                                       min_val=min_val, 
                                                                       max_val=max_val, 
                                                                       mutation_by_replacement=mutation_by_replacement, 
                                                                       gene_type=gene_type)
                        else:
                            temp_val = numpy.random.uniform(low=min_val,
                                                            high=max_val,
                                                            size=1)
                            if mutation_by_replacement:
                                pass
                            else:
                                temp_val = new_solution[duplicate_index] + temp_val
                    
                    if self.gene_type_single == True:
                        if not gene_type[1] is None:
                            temp_val = numpy.round(gene_type[0](temp_val),
                                                   gene_type[1])
                        else:
                            temp_val = gene_type[0](temp_val)
                    else:
                        if not gene_type[duplicate_index][1] is None:
                            temp_val = numpy.round(gene_type[duplicate_index][0](temp_val),
                                                   gene_type[duplicate_index][1])
                        else:
                            temp_val = gene_type[duplicate_index][0](temp_val)

                    if temp_val in new_solution and trial_index == (num_trials - 1):
                        num_unsolved_duplicates = num_unsolved_duplicates + 1
                        if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.".format(gene_idx=duplicate_index))
                    elif temp_val in new_solution:
                        continue
                    else:
                        new_solution[duplicate_index] = temp_val
                        break

                # Update the list of duplicate indices after each iteration.
                _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
                not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
                # print("not_unique_indices INSIDE", not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def solve_duplicate_genes_by_space(self, solution, gene_type, num_trials=10, build_initial_pop=False):

        """
        Solves the duplicates in a solution by selecting values for the duplicating genes from the gene space.
        
        solution: A solution with duplicate values.
        gene_type: Exactly the same as the self.gene_type attribute.
        num_trials: Maximum number of trials to change the gene value to solve the duplicates.

        Returns:
            new_solution: Solution after trying to solve its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
            not_unique_indices: Indices of the genes with duplicate values.
            num_unsolved_duplicates: Number of unsolved duplicates.
        """
        
        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # print("not_unique_indices OUTSIDE", not_unique_indices)

        # First try to solve the duplicates.
        # For a solution like [3 2 0 0], the indices of the 2 duplicating genes are 2 and 3.
        # The next call to the find_unique_value() method tries to change the value of the gene with index 3 to solve the duplicate.
        if len(not_unique_indices) > 0:
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            return new_solution, not_unique_indices, len(not_unique_indices)

        # Do another try if there exist duplicate genes.
        # If there are no possible values for the gene 3 with index 3 to solve the duplicate, try to change the value of the other gene with index 2.
        if len(not_unique_indices) > 0:
            not_unique_indices = set(numpy.where(new_solution == new_solution[list(not_unique_indices)[0]])[0]) - set([list(not_unique_indices)[0]])
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            # If there exist duplicate genes, then changing either of the 2 duplicating genes (with indices 2 and 3) will not solve the problem.
            # This problem can be solved by randomly changing one of the non-duplicating genes that may make a room for a unique value in one the 2 duplicating genes.
            # For example, if gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]] and the solution is [3 2 0 0], then the values of the last 2 genes duplicate.
            # There are no possible changes in the last 2 genes to solve the problem. But it could be solved by changing the second gene from 2 to 4.
            # As a result, any of the last 2 genes can take the value 2 and solve the duplicates.
            return new_solution, not_unique_indices, len(not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def solve_duplicate_genes_by_space_OLD(self, solution, gene_type, num_trials=10):
        # /////////////////////////
        # Just for testing purposes.
        # /////////////////////////

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # print("not_unique_indices OUTSIDE", not_unique_indices)

        num_unsolved_duplicates = 0
        if len(not_unique_indices) > 0:
            for duplicate_index in not_unique_indices:
                for trial_index in range(num_trials):
                    temp_val = self.unique_gene_by_space(solution=solution, 
                                                         gene_idx=duplicate_index, 
                                                         gene_type=gene_type)

                    if temp_val in new_solution and trial_index == (num_trials - 1):
                        # print("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                        num_unsolved_duplicates = num_unsolved_duplicates + 1
                        if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}".format(gene_idx=duplicate_index))
                    elif temp_val in new_solution:
                        continue
                    else:
                        new_solution[duplicate_index] = temp_val
                        # print("SOLVED", duplicate_index)
                        break

                # Update the list of duplicate indices after each iteration.
                _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
                not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
                # print("not_unique_indices INSIDE", not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_int_gene_from_range(self, solution, gene_index, min_val, max_val, mutation_by_replacement, gene_type, step=None):

        """
        Finds a unique integer value for the gene.

        solution: A solution with duplicate values.
        gene_index: Index of the gene to find a unique value.
        min_val: Minimum value of the range to sample a number randomly.
        max_val: Maximum value of the range to sample a number randomly.
        mutation_by_replacement: Identical to the self.mutation_by_replacement attribute.
        gene_type: Exactly the same as the self.gene_type attribute.

        Returns:
            selected_value: The new value of the gene. It may be identical to the original gene value in case there are no possible unique values for the gene.
        """

        if self.gene_type_single == True:
            if step is None:
                all_gene_values = numpy.arange(min_val, max_val, dtype=gene_type[0])
            else:
                # For non-integer steps, the numpy.arange() function returns zeros id the dtype parameter is set to an integer data type. So, this returns zeros if step is non-integer and dtype is set to an int data type: numpy.arange(min_val, max_val, step, dtype=gene_type[0])
                # To solve this issue, the data type casting will not be handled inside numpy.arange(). The range is generated by numpy.arange() and then the data type is converted using the numpy.asarray() function.
                all_gene_values = numpy.asarray(numpy.arange(min_val, max_val, step), dtype=gene_type[0])
        else:
            if step is None:
                all_gene_values = numpy.arange(min_val, max_val, dtype=gene_type[gene_index][0])
            else:
                all_gene_values = numpy.asarray(numpy.arange(min_val, max_val, step), dtype=gene_type[gene_index][0])

        if mutation_by_replacement:
            pass
        else:
            all_gene_values = all_gene_values + solution[gene_index]

        if self.gene_type_single == True:
            if not gene_type[1] is None:
                all_gene_values = numpy.round(gene_type[0](all_gene_values),
                                              gene_type[1])
            else:
                if type(all_gene_values) is numpy.ndarray:
                    all_gene_values = numpy.asarray(all_gene_values, dtype=gene_type[0])
                else:
                    all_gene_values = gene_type[0](all_gene_values)
        else:
            if not gene_type[gene_index][1] is None:
                all_gene_values = numpy.round(gene_type[gene_index][0](all_gene_values),
                                              gene_type[gene_index][1])
            else:
                all_gene_values = gene_type[gene_index][0](all_gene_values)

        values_to_select_from = list(set(all_gene_values) - set(solution))
    
        if len(values_to_select_from) == 0:
            if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but there is no enough values to prevent duplicates.")
            selected_value = solution[gene_index]
        else:
            selected_value = random.choice(values_to_select_from)

        #if self.gene_type_single == True:
        #    selected_value = gene_type[0](selected_value)
        #else:
        #    selected_value = gene_type[gene_index][0](selected_value)

        return selected_value

    def unique_genes_by_space(self, new_solution, gene_type, not_unique_indices, num_trials=10, build_initial_pop=False):

        """
        Loops through all the duplicating genes to find unique values that from their gene spaces to solve the duplicates.
        For each duplicating gene, a call to the unique_gene_by_space() is made.

        new_solution: A solution with duplicate values.
        gene_type: Exactly the same as the self.gene_type attribute.
        not_unique_indices: Indices with duplicating values.
        num_trials: Maximum number of trials to change the gene value to solve the duplicates.

        Returns:
            new_solution: Solution after trying to solve all of its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
            not_unique_indices: Indices of the genes with duplicate values.
            num_unsolved_duplicates: Number of unsolved duplicates.
        """

        num_unsolved_duplicates = 0
        for duplicate_index in not_unique_indices:
            for trial_index in range(num_trials):
                temp_val = self.unique_gene_by_space(solution=new_solution, 
                                                     gene_idx=duplicate_index, 
                                                     gene_type=gene_type,
                                                     build_initial_pop=build_initial_pop)

                if temp_val in new_solution and trial_index == (num_trials - 1):
                    # print("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                    num_unsolved_duplicates = num_unsolved_duplicates + 1
                    if not self.suppress_warnings: warnings.warn("Failed to find a unique value for gene with index {gene_idx}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.".format(gene_idx=duplicate_index))
                elif temp_val in new_solution:
                    continue
                else:
                    new_solution[duplicate_index] = temp_val
                    # print("SOLVED", duplicate_index)
                    break

        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
        not_unique_indices = set(range(len(new_solution))) - set(unique_gene_indices)
        # print("not_unique_indices INSIDE", not_unique_indices)        

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_gene_by_space(self, solution, gene_idx, gene_type, build_initial_pop=False):

        """
        Returns a unique gene value for a single gene based on its value space to solve the duplicates.
        
        solution: A solution with duplicate values.
        gene_idx: The index of the gene that duplicates its value with another gene.
        gene_type: Exactly the same as the self.gene_type attribute.

        Returns:
            A unique value, if exists, for the gene.
        """

        if self.gene_space_nested:
            # Returning the current gene space from the 'gene_space' attribute.
            if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                curr_gene_space = self.gene_space[gene_idx].copy()
            else:
                curr_gene_space = self.gene_space[gene_idx]

            # If the gene space has only a single value, use it as the new gene value.
            if type(curr_gene_space) in GA.supported_int_float_types:
                value_from_space = curr_gene_space
                # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
            elif curr_gene_space is None:
                if self.gene_type_single == True:
                    if gene_type[0] in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val,
                                                                high=self.random_mutation_max_val,
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, 
                                                                               gene_type=gene_type)
                        else:
                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=self.random_mutation_min_val, 
                                                                               max_val=self.random_mutation_max_val, 
                                                                               mutation_by_replacement=True, #self.mutation_by_replacement, 
                                                                               gene_type=gene_type)
                    else:
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val,
                                                                high=self.random_mutation_max_val,
                                                                size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            elif type(curr_gene_space) is dict:
                if self.gene_type_single == True:
                    if gene_type[0] in GA.supported_int_types:
                        if build_initial_pop == True:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=curr_gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                        else:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=curr_gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                    else:
                        if 'step' in curr_gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                stop=curr_gene_space['high'],
                                                                                step=curr_gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=curr_gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                        else:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=curr_gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=curr_gene_space['low'], 
                                                                                   max_val=curr_gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                    else:
                        if 'step' in curr_gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                stop=curr_gene_space['high'],
                                                                                step=curr_gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            else:
                # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                if len(curr_gene_space) == 1:
                    value_from_space = curr_gene_space[0]
                    if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the space of the gene with index {gene_idx} has only a single value. Thus, duplicates are possible.".format(gene_idx=gene_idx))
                # If the gene space has more than 1 value, then select a new one that is different from the current value.
                else:
                    values_to_select_from = list(set(curr_gene_space) - set(solution))
                    if len(values_to_select_from) == 0:
                        if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")
                        value_from_space = solution[gene_idx]
                    else:
                        value_from_space = random.choice(values_to_select_from)
        else:
            # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
            if type(self.gene_space) is dict:
                if self.gene_type_single == True:
                    if gene_type[0] in GA.supported_int_types:
                        if build_initial_pop == True:
                            if 'step' in self.gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=self.gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                        else:
                            if 'step' in self.gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=self.gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                    else:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space
                else:
                    if gene_type[gene_idx] in GA.supported_int_types:
                        if build_initial_pop == True:
                            if 'step' in self.gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=self.gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                        else:
                            if 'step' in self.gene_space.keys():
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=self.gene_space['step'],
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                            else:
                                value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                                   gene_index=gene_idx, 
                                                                                   min_val=self.gene_space['low'], 
                                                                                   max_val=self.gene_space['high'], 
                                                                                   step=None,
                                                                                   mutation_by_replacement=True, 
                                                                                   gene_type=gene_type)
                    else:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                        if self.mutation_by_replacement:
                            pass
                        else:
                            value_from_space = solution[gene_idx] + value_from_space

            else:
                # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                values_to_select_from = list(set(self.gene_space) - set(solution))
                if len(values_to_select_from) == 0:
                    if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")
                    value_from_space = solution[gene_idx]
                else:
                    value_from_space = random.choice(values_to_select_from)

        if value_from_space is None:
            value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
                                                    size=1)

        if self.gene_type_single == True:
            if not gene_type[1] is None:
                value_from_space = numpy.round(gene_type[0](value_from_space),
                                              gene_type[1])
            else:
                value_from_space = gene_type[0](value_from_space)
        else:
            if not gene_type[gene_idx][1] is None:
                value_from_space = numpy.round(gene_type[gene_idx][0](value_from_space),
                                              gene_type[gene_idx][1])
            else:
                value_from_space = gene_type[gene_idx][0](value_from_space)

        return value_from_space

    def best_solution(self, pop_fitness=None):

        """
        Returns information about the best solution found by the genetic algorithm.
        Accepts the following parameters:
            pop_fitness: An optional parameter holding the fitness values of the solutions in the current population. If None, then the cal_pop_fitness() method is called to calculate the fitness of the population.
        The following are returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        if pop_fitness is None:
            pop_fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(pop_fitness == numpy.max(pop_fitness))[0][0]

        best_solution = self.population[best_match_idx, :].copy()
        best_solution_fitness = pop_fitness[best_match_idx]

        return best_solution, best_solution_fitness, best_match_idx

    def plot_result(self, 
                    title="PyGAD - Generation vs. Fitness", 
                    xlabel="Generation", 
                    ylabel="Fitness", 
                    linewidth=3, 
                    font_size=14, 
                    plot_type="plot",
                    color="#3870FF",
                    save_dir=None):

        if not self.suppress_warnings: 
            warnings.warn("Please use the plot_fitness() method instead of plot_result(). The plot_result() method will be removed in the future.")

        return self.plot_fitness(title=title, 
                                 xlabel=xlabel, 
                                 ylabel=ylabel, 
                                 linewidth=linewidth, 
                                 font_size=font_size, 
                                 plot_type=plot_type,
                                 color=color,
                                 save_dir=save_dir)

    def plot_fitness(self, 
                    title="PyGAD - Generation vs. Fitness", 
                    xlabel="Generation", 
                    ylabel="Fitness", 
                    linewidth=3, 
                    font_size=14, 
                    plot_type="plot",
                    color="#3870FF",
                    save_dir=None):

        """
        Creates, shows, and returns a figure that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#3870FF".
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        if self.generations_completed < 1:
            raise RuntimeError("The plot_fitness() (i.e. plot_result()) method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            if not self.suppress_warnings: warnings.warn("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        fig = matplotlib.pyplot.figure()
        if plot_type == "plot":
            matplotlib.pyplot.plot(self.best_solutions_fitness, linewidth=linewidth, color=color)
        elif plot_type == "scatter":
            matplotlib.pyplot.scatter(range(self.generations_completed + 1), self.best_solutions_fitness, linewidth=linewidth, color=color)
        elif plot_type == "bar":
            matplotlib.pyplot.bar(range(self.generations_completed + 1), self.best_solutions_fitness, linewidth=linewidth, color=color)
        matplotlib.pyplot.title(title, fontsize=font_size)
        matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
        matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)
        
        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        matplotlib.pyplot.show()

        return fig

    def plot_new_solution_rate(self,
                               title="PyGAD - Generation vs. New Solution Rate", 
                               xlabel="Generation", 
                               ylabel="New Solution Rate", 
                               linewidth=3, 
                               font_size=14, 
                               plot_type="plot",
                               color="#3870FF",
                               save_dir=None):

        """
        Creates, shows, and returns a figure that summarizes the rate of exploring new solutions. This method works only when save_solutions=True in the constructor of the pygad.GA class.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            color: Color of the plot which defaults to "#3870FF".
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        if self.generations_completed < 1:
            raise RuntimeError("The plot_new_solution_rate() method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=self.generations_completed))

        if self.save_solutions == False:
            raise RuntimeError("The plot_new_solution_rate() method works only when save_solutions=True in the constructor of the pygad.GA class.")

        unique_solutions = set()
        num_unique_solutions_per_generation = []
        for generation_idx in range(self.generations_completed):
            
            len_before = len(unique_solutions)

            start = generation_idx * self.sol_per_pop
            end = start + self.sol_per_pop
        
            for sol in self.solutions[start:end]:
                unique_solutions.add(tuple(sol))
        
            len_after = len(unique_solutions)
        
            generation_num_unique_solutions = len_after - len_before
            num_unique_solutions_per_generation.append(generation_num_unique_solutions)

        fig = matplotlib.pyplot.figure()
        if plot_type == "plot":
            matplotlib.pyplot.plot(num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        elif plot_type == "scatter":
            matplotlib.pyplot.scatter(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        elif plot_type == "bar":
            matplotlib.pyplot.bar(range(self.generations_completed), num_unique_solutions_per_generation, linewidth=linewidth, color=color)
        matplotlib.pyplot.title(title, fontsize=font_size)
        matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
        matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)

        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir, 
                                      bbox_inches='tight')
        matplotlib.pyplot.show()

        return fig

    def plot_genes(self, 
                   title="PyGAD - Gene", 
                   xlabel="Gene", 
                   ylabel="Value", 
                   linewidth=3, 
                   font_size=14,
                   plot_type="plot",
                   graph_type="plot",
                   fill_color="#3870FF",
                   color="black",
                   solutions="all",
                   save_dir=None):

        """
        Creates, shows, and returns a figure with number of subplots equal to the number of genes. Each subplot shows the gene value for each generation. 
        This method works only when save_solutions=True in the constructor of the pygad.GA class. 
        It also works only after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot. Defaults to 3.
            font_size: Font size for the labels and title. Defaults to 14.
            plot_type: Type of the plot which can be either "plot" (default), "scatter", or "bar".
            graph_type: Type of the graph which can be either "plot" (default), "boxplot", or "histogram".
            fill_color: Fill color of the graph which defaults to "#3870FF". This has no effect if graph_type="plot".
            color: Color of the plot which defaults to "black".
            solutions: Defaults to "all" which means use all solutions. If "best" then only the best solutions are used.
            save_dir: Directory to save the figure.

        Returns the figure.
        """

        if self.generations_completed < 1:
            raise RuntimeError("The plot_genes() method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=self.generations_completed))
        
        if type(solutions) is str:
            if solutions == 'all':
                if self.save_solutions:
                    solutions_to_plot = self.solutions
                else:
                    raise RuntimeError("The plot_genes() method with solutions='all' can only be called if 'save_solutions=True' in the pygad.GA class constructor.")
            elif solutions == 'best':
                if self.save_best_solutions:
                    solutions_to_plot = self.best_solutions
                else:
                    raise RuntimeError("The plot_genes() method with solutions='best' can only be called if 'save_best_solutions=True' in the pygad.GA class constructor.")
            else:
                raise RuntimeError("The solutions parameter can be either 'all' or 'best' but {solutions} found.".format(solutions=solutions))
        else:
            raise RuntimeError("The solutions parameter must be a string but {solutions_type} found.".format(solutions_type=type(solutions)))

        if graph_type == "plot":
            # num_rows will be always be >= 1
            # num_cols can only be 0 if num_genes=1
            num_rows = int(numpy.ceil(self.num_genes/5.0))
            num_cols = int(numpy.ceil(self.num_genes/num_rows))
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = matplotlib.pyplot.subplots(num_rows, figsize=figsize)
                if plot_type == "plot":
                    ax.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "scatter":
                    ax.scatter(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                elif plot_type == "bar":
                    ax.bar(range(self.generations_completed + 1), solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                fig, axs = matplotlib.pyplot.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(5 * num_cols)
                    fig.set_figheight(4)
                    axs.plot(solutions_to_plot[:, 0], linewidth=linewidth, color=fill_color)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                elif num_cols == 1 or num_rows == 1:
                    fig.set_figwidth(5 * num_cols)
                    fig.set_figheight(4)
                    for gene_idx in range(len(axs)):
                        if plot_type == "plot":
                            axs[gene_idx].plot(solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        elif plot_type == "scatter":
                            axs[gene_idx].scatter(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        elif plot_type == "bar":
                            axs[gene_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                        axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    gene_idx = 0
                    fig.set_figwidth(25)
                    fig.set_figheight(4*num_rows)
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break
                            if plot_type == "plot":
                                axs[row_idx, col_idx].plot(solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            elif plot_type == "scatter":
                                axs[row_idx, col_idx].scatter(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            elif plot_type == "bar":
                                axs[row_idx, col_idx].bar(range(solutions_to_plot.shape[0]), solutions_to_plot[:, gene_idx], linewidth=linewidth, color=fill_color)
                            axs[row_idx, col_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                            gene_idx += 1
    
            fig.suptitle(title, fontsize=font_size, y=1.001)
            matplotlib.pyplot.tight_layout()

        elif graph_type == "boxplot":
            fig = matplotlib.pyplot.figure(1, figsize=(0.7*self.num_genes, 6))

            # Create an axes instance
            ax = fig.add_subplot(111)
            boxeplots = ax.boxplot(solutions_to_plot, 
                                   labels=range(self.num_genes),
                                   patch_artist=True)
            # adding horizontal grid lines
            ax.yaxis.grid(True)
    
            for box in boxeplots['boxes']:
                # change outline color
                box.set(color='black', linewidth=linewidth)
                # change fill color https://color.adobe.com/create/color-wheel
                box.set_facecolor(fill_color)

            for whisker in boxeplots['whiskers']:
                whisker.set(color=color, linewidth=linewidth)
            for median in boxeplots['medians']:
                median.set(color=color, linewidth=linewidth)
            for cap in boxeplots['caps']:
                cap.set(color=color, linewidth=linewidth)
    
            matplotlib.pyplot.title(title, fontsize=font_size)
            matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
            matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)
            matplotlib.pyplot.tight_layout()

        elif graph_type == "histogram":
            # num_rows will be always be >= 1
            # num_cols can only be 0 if num_genes=1
            num_rows = int(numpy.ceil(self.num_genes/5.0))
            num_cols = int(numpy.ceil(self.num_genes/num_rows))
    
            if num_cols == 0:
                figsize = (10, 8)
                # There is only a single gene
                fig, ax = matplotlib.pyplot.subplots(num_rows, 
                                                     figsize=figsize)
                ax.hist(solutions_to_plot[:, 0], color=fill_color)
                ax.set_xlabel(0, fontsize=font_size)
            else:
                fig, axs = matplotlib.pyplot.subplots(num_rows, num_cols)
    
                if num_cols == 1 and num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    axs.hist(solutions_to_plot[:, 0], 
                             color=fill_color,
                             rwidth=0.95)
                    axs.set_xlabel("Gene " + str(0), fontsize=font_size)
                elif num_cols == 1 or num_rows == 1:
                    fig.set_figwidth(4 * num_cols)
                    fig.set_figheight(3)
                    for gene_idx in range(len(axs)):
                        axs[gene_idx].hist(solutions_to_plot[:, gene_idx], 
                                           color=fill_color,
                                           rwidth=0.95)
                        axs[gene_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                else:
                    gene_idx = 0
                    fig.set_figwidth(20)
                    fig.set_figheight(3*num_rows)
                    for row_idx in range(num_rows):
                        for col_idx in range(num_cols):
                            if gene_idx >= self.num_genes:
                                # axs[row_idx, col_idx].remove()
                                break
                            axs[row_idx, col_idx].hist(solutions_to_plot[:, gene_idx], 
                                                       color=fill_color,
                                                       rwidth=0.95)
                            axs[row_idx, col_idx].set_xlabel("Gene " + str(gene_idx), fontsize=font_size)
                            gene_idx += 1
    
            fig.suptitle(title, fontsize=font_size, y=1.001)
            matplotlib.pyplot.tight_layout()

        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir, 
                                      bbox_inches='tight')

        matplotlib.pyplot.show()

        return fig

    def save(self, filename):

        """
        Saves the genetic algorithm instance:
            -filename: Name of the file to save the instance. No extension is needed.
        """

        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)

def load(filename):

    """
    Reads a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. No extension is needed.
    Returns the genetic algorithm instance.
    """

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Error reading the file {filename}. Please check your inputs.".format(filename=filename))
    except:
        raise BaseException("Error loading the file. If the file already exists, please reload all the functions previously used (e.g. fitness function).")
    return ga_in