import numpy
import random
import matplotlib.pyplot
import pickle
import time
import warnings

class GA:
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
                 suppress_warnings=False):

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

        gene_type: The type of the gene. It works only when the 'gene_space' parameter is None (i.e. the population is created randomly). It is assigned to any of these types (int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64) and forces all the genes to be of that type.

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

        suppress_warnings: Added in PyGAD 2.10.0 and its type is bool. If True, then no warning messages will be displayed. It defaults to False.
        """
        
        if type(suppress_warnings) is bool:
            self.suppress_warnings = suppress_warnings
        else:
            self.valid_parameters = False
            raise TypeError("The expected type of the 'suppress_warnings' parameter is bool but {suppress_warnings_type} found.".format(suppress_warnings_type=type(suppress_warnings)))

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
                                if not (type(val) in [type(None), int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]):
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
                                raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items with keys 'low' and 'high' but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=el.keys()))
                        else:
                            self.valid_parameters = False
                            raise TypeError("When an element in the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(el.items())))
                        self.gene_space_nested = True
                    elif not (type(el) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]):
                        self.valid_parameters = False
                        raise TypeError("Unexpected type {el_type} for the element indexed {index} of 'gene_space'. The accepted types are list/tuple/range/numpy.ndarray of numbers, a single number (int/float), or None.".format(index=index, el_type=type(el)))

        elif type(gene_space) is dict:
            if len(gene_space.items()) == 2:
                if ('low' in gene_space.keys()) and ('high' in gene_space.keys()):
                    pass
                else:
                    self.valid_parameters = False
                    raise TypeError("When the 'gene_space' parameter is of type dict, then it must have only 2 items with keys 'low' and 'high' but the following keys found: {gene_space_dict_keys}".format(gene_space_dict_keys=gene_space.keys()))
            else:
                self.valid_parameters = False
                raise TypeError("When the 'gene_space' parameter is of type dict, then it must have only 2 items but ({num_items}) items found.".format(num_items=len(gene_space.items())))

        else:
            self.valid_parameters = False
            raise TypeError("The expected type of 'gene_space' is list, tuple, range, or numpy.ndarray but ({gene_space_type}) found.".format(gene_space_type=type(gene_space)))

        self.gene_space = gene_space

        self.init_range_low = init_range_low
        self.init_range_high = init_range_high

        if gene_type in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
            self.gene_type = gene_type
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'gene_type' parameter must be either integer or floating-point number but the value ({gene_type_value}) of type {gene_type_type} found.".format(gene_type_value=gene_type, gene_type_type=type(gene_type)))

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
                self.sol_per_pop = sol_per_pop # Number of solutions in the population.
                self.initialize_population(self.init_range_low, self.init_range_high)
            else:
                raise TypeError("The expected type of both the sol_per_pop and num_genes parameters is int but ({sol_per_pop_type}) and {num_genes_type} found.".format(sol_per_pop_type=type(sol_per_pop), num_genes_type=type(num_genes)))
        elif numpy.array(initial_population).ndim != 2:
            raise ValueError("A 2D list is expected to the initail_population parameter but a ({initial_population_ndim}-D) list found.".format(initial_population_ndim=numpy.array(initial_population).ndim))
        else:
            # Forcing the initial_population array to have the data type assigned to the gene_type parameter.
            self.initial_population = numpy.array(initial_population, dtype=self.gene_type)
            self.population = self.initial_population.copy() # A NumPy array holding the initial population.
            self.num_genes = self.initial_population.shape[1] # Number of genes in the solution.
            self.sol_per_pop = self.initial_population.shape[0]  # Number of solutions in the population.
            self.pop_size = (self.sol_per_pop,self.num_genes) # The population size.

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
        if (crossover_type == "single_point"):
            self.crossover = self.single_point_crossover
        elif (crossover_type == "two_points"):
            self.crossover = self.two_points_crossover
        elif (crossover_type == "uniform"):
            self.crossover = self.uniform_crossover
        elif (crossover_type == "scattered"):
            self.crossover = self.scattered_crossover
        elif (crossover_type is None):
            self.crossover = None
        else:
            self.valid_parameters = False
            raise ValueError("Undefined crossover type. \nThe assigned value to the crossover_type ({crossover_type}) argument does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover).\n".format(crossover_type=crossover_type))

        self.crossover_type = crossover_type

        if crossover_probability is None:
            self.crossover_probability = None
        elif type(crossover_probability) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
        elif (mutation_type is None):
            self.mutation = None
        else:
            self.valid_parameters = False
            raise ValueError("Undefined mutation type. \nThe assigned value to the mutation_type argument ({mutation_type}) does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation).\n".format(mutation_type=mutation_type))

        self.mutation_type = mutation_type

        if not (self.mutation_type is None):
            if mutation_probability is None:
                self.mutation_probability = None
            elif (mutation_type != "adaptive"):
                # Mutation probability is fixed not adaptive.
                if type(mutation_probability) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                            if type(el) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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

                    elif type(mutation_percent_genes) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                                if type(el) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                if type(mutation_num_genes) in [int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
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
                            if type(el) in [int, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
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

        if not (type(mutation_by_replacement) is bool):
            self.valid_parameters = False
            raise TypeError("The expected type of the 'mutation_by_replacement' parameter is bool but ({mutation_by_replacement_type}) found.".format(mutation_by_replacement_type=type(mutation_by_replacement)))

        self.mutation_by_replacement = mutation_by_replacement
        
        if self.mutation_type != "random" and self.mutation_by_replacement:
            if not self.suppress_warnings: warnings.warn("The mutation_by_replacement parameter is set to True while the mutation_type parameter is not set to random but ({mut_type}). Note that the mutation_by_replacement parameter has an effect only when mutation_type='random'.".format(mut_type=mutation_type))

        if (self.mutation_type is None) and (self.crossover_type is None):
            if not self.suppress_warnings: warnings.warn("The 2 parameters mutation_type and crossover_type are None. This disables any type of evolution the genetic algorithm can make. As a result, the genetic algorithm cannot find a better solution that the best solution in the initial population.")

        # select_parents: Refers to a method that selects the parents based on the parent selection type specified in the parent_selection_type attribute.
        # Validating the selected type of parent selection: parent_selection_type
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
            raise ValueError("Undefined parent selection type: {parent_selection_type}. \nThe assigned value to the parent_selection_type argument does not refer to one of the supported parent selection techniques which are: \n-sss (for steady state selection)\n-rws (for roulette wheel selection)\n-sus (for stochastic universal selection)\n-rank (for rank selection)\n-random (for random selection)\n-tournament (for tournament selection).\n".format(parent_selection_type))

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
                raise ValueError("The fitness function must accept 2 parameters representing the solution to which the fitness value is calculated and the solution index within the population.\nThe passed fitness function named '{funcname}' accepts {argcount} argument(s).".format(funcname=fitness_func.__code__.co_name, argcount=fitness_func.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_start parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_start.__code__.co_name, argcount=on_start.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_fitness parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_fitness.__code__.co_name, argcount=on_fitness.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_parents parameter must accept 2 parameters representing the instance of the genetic algorithm and the fitness values of all solutions.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_parents.__code__.co_name, argcount=on_parents.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_crossover parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring generated using crossover.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_crossover.__code__.co_name, argcount=on_crossover.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_mutation parameter must accept 2 parameters representing the instance of the genetic algorithm and the offspring after applying the mutation operation.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_mutation.__code__.co_name, argcount=on_mutation.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the callback_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=callback_generation.__code__.co_name, argcount=callback_generation.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_generation parameter must accept only 1 parameter representing the instance of the genetic algorithm.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_generation.__code__.co_name, argcount=on_generation.__code__.co_argcount))
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
                    raise ValueError("The function assigned to the on_stop parameter must accept 2 parameters representing the instance of the genetic algorithm and a list of the fitness values of the solutions in the last population.\nThe passed function named '{funcname}' accepts {argcount} argument(s).".format(funcname=on_stop.__code__.co_name, argcount=on_stop.__code__.co_argcount))
            else:
                self.valid_parameters = False
                raise ValueError("The value assigned to the 'on_stop' parameter is expected to be of type function but ({on_stop_type}) found.".format(on_stop_type=type(on_stop)))
        else:
            self.on_stop = None

        # delay_after_gen
        if type(delay_after_gen) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
            if delay_after_gen >= 0.0:
                self.delay_after_gen = delay_after_gen
            else:
                self.valid_parameters = False
                raise ValueError("The value passed to the 'delay_after_gen' parameter must be a non-negative number. The value passed is {delay_after_gen} of type {delay_after_gen_type}.".format(delay_after_gen=delay_after_gen, delay_after_gen_type=type(delay_after_gen)))
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'delay_after_gen' parameter must be of type int or float but ({delay_after_gen_type}) found.".format(delay_after_gen_type=type(delay_after_gen)))

        # save_best_solutions
        if type(save_best_solutions) is bool:
            if save_best_solutions == True:
                if not self.suppress_warnings: warnings.warn("Use the 'save_best_solutions' parameter with caution as it may cause memory overflow.")
        else:
            self.valid_parameters = False
            raise ValueError("The value passed to the 'save_best_solutions' parameter must be of type bool but ({save_best_solutions_type}) found.".format(save_best_solutions_type=type(save_best_solutions)))

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
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solutions_fitness = [] # A list holding the fitness value of the best solution for each generation.

        self.best_solution_generation = -1 # The generation number at which the best fitness value is reached. It is only assigned the generation number after the `run()` method completes. Otherwise, its value is -1.

        self.save_best_solutions = save_best_solutions
        self.best_solutions = [] # Holds the best solution in each generation.

        self.last_generation_fitness = None # A list holding the fitness values of all solutions in the last generation.
        self.last_generation_parents = None # A list holding the parents of the last generation.
        self.last_generation_offspring_crossover = None # A list holding the offspring after applying crossover in the last generation.
        self.last_generation_offspring_mutation = None # A list holding the offspring after applying mutation in the last generation.

    def initialize_population(self, low, high):

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
            self.population = numpy.asarray(numpy.random.uniform(low=low, 
                                                                 high=high, 
                                                                 size=self.pop_size), dtype=self.gene_type) # A NumPy array holding the initial population.
        elif self.gene_space_nested:
            self.population = numpy.zeros(shape=self.pop_size, dtype=self.gene_type)
            for sol_idx in range(self.sol_per_pop):
                for gene_idx in range(self.num_genes):
                    if type(self.gene_space[gene_idx]) in [list, tuple, range]:
                        # Check if the gene space has None values. If any, then replace it with randomly generated values according to the 3 attributes init_range_low, init_range_high, and gene_type.
                        for idx, val in enumerate(self.gene_space[gene_idx]):
                            if val is None:
                                self.gene_space[gene_idx][idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                               high=high, 
                                               size=1), dtype=self.gene_type)[0]
                        self.population[sol_idx, gene_idx] = random.choice(self.gene_space[gene_idx])
                    elif type(self.gene_space[gene_idx]) is dict:
                        self.population[sol_idx, gene_idx] = numpy.random.uniform(low=self.gene_space[gene_idx]['low'],
                                       high=self.gene_space[gene_idx]['high'],
                                       size=1)
                    elif type(self.gene_space[gene_idx]) == type(None):
                        self.gene_space[gene_idx] = numpy.asarray(numpy.random.uniform(low=low,
                                       high=high, 
                                       size=1), dtype=self.gene_type)[0]
                        self.population[sol_idx, gene_idx] = self.gene_space[gene_idx].copy()
                    elif type(self.gene_space[gene_idx]) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
                        self.population[sol_idx, gene_idx] = self.gene_space[gene_idx]
        else:
            # Replace all the None values with random values using the init_range_low, init_range_high, and gene_type attributes.
            for idx, curr_gene_space in enumerate(self.gene_space):
                if curr_gene_space is None:
                    self.gene_space[idx] = numpy.asarray(numpy.random.uniform(low=low, 
                                   high=high, 
                                   size=1), dtype=self.gene_type)[0]

            # Creating the initial population by randomly selecting the genes' values from the values inside the 'gene_space' parameter.
            if type(self.gene_space) is dict:
                self.population = numpy.asarray(numpy.random.uniform(low=self.gene_space['low'],
                                                                     high=self.gene_space['high'],
                                                                     size=self.pop_size),
                        dtype=self.gene_type) # A NumPy array holding the initial population.
            else:
                self.population = numpy.asarray(numpy.random.choice(self.gene_space,
                                                                    size=self.pop_size),
                                dtype=self.gene_type) # A NumPy array holding the initial population.

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
            fitness = self.fitness_func(sol, sol_idx)
            pop_fitness.append(fitness)

        pop_fitness = numpy.array(pop_fitness)

        return pop_fitness

    def run(self):

        """
        Runs the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """

        if self.valid_parameters == False:
            raise ValueError("ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        if not (self.on_start is None):
            self.on_start(self)

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population. Save the fitness in the last_generation_fitness attribute.
            self.last_generation_fitness = self.cal_pop_fitness()
            if not (self.on_fitness is None):
                self.on_fitness(self, self.last_generation_fitness)

            best_solution, best_solution_fitness, best_match_idx = self.best_solution(pop_fitness=self.last_generation_fitness)

            # Appending the fitness value of the best solution in the current generation to the best_solutions_fitness attribute.
            self.best_solutions_fitness.append(best_solution_fitness)

            # Appending the best solution to the best_solutions list.
            if self.save_best_solutions:
                self.best_solutions.append(best_solution)

            # Selecting the best parents in the population for mating.
            self.last_generation_parents = self.select_parents(self.last_generation_fitness, num_parents=self.num_parents_mating)
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
                self.last_generation_offspring_crossover = self.crossover(self.last_generation_parents,
                                                     offspring_size=(self.num_offspring, self.num_genes))
                if not (self.on_crossover is None):
                    self.on_crossover(self, self.last_generation_offspring_crossover)

            # If self.mutation_type=None, then no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
            if self.mutation_type is None:
                self.last_generation_offspring_mutation = self.last_generation_offspring_crossover
            else:
                # Adding some variations to the offspring using mutation.
                self.last_generation_offspring_mutation = self.mutation(self.last_generation_offspring_crossover)
                if not (self.on_mutation is None):
                    self.on_mutation(self, self.last_generation_offspring_mutation)

            if (self.keep_parents == 0):
                self.population = self.last_generation_offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:self.last_generation_parents.shape[0], :] = self.last_generation_parents
                self.population[self.last_generation_parents.shape[0]:, :] = self.last_generation_offspring_mutation
            elif (self.keep_parents > 0):
                parents_to_keep = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = self.last_generation_offspring_mutation

            self.generations_completed = generation + 1 # The generations_completed attribute holds the number of the last completed generation.

            # If the callback_generation attribute is not None, then cal the callback function after the generation.
            if not (self.on_generation is None):
                r = self.on_generation(self)
                if type(r) is str and r.lower() == "stop":
                    # Before aborting the loop, save the fitness value of the best solution.
                    _, best_solution_fitness, _ = self.best_solution()
                    self.best_solutions_fitness.append(best_solution_fitness)
                    break

            time.sleep(self.delay_after_gen)

        self.last_generation_fitness = self.cal_pop_fitness()
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
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()
        return parents

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
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()
        return parents

    def random_selection(self, fitness, num_parents):

        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :].copy()
        return parents

    def tournament_selection(self, fitness, num_parents):

        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :].copy()
        return parents

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
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    break
        return parents

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
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num*pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    break
        return parents

    def single_point_crossover(self, parents, offspring_size):

        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = numpy.empty(offspring_size)

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

        offspring = numpy.empty(offspring_size)

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

        offspring = numpy.empty(offspring_size)

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

        offspring = numpy.empty(offspring_size)

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
                    if type(curr_gene_space) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                        value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                high=curr_gene_space['high'],
                                                                size=1)
                    else:
                        # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                        values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                high=self.gene_space['high'],
                                                                size=1)
                    else:
                        # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        value_from_space = random.choice(values_to_select_from)
                    # value_from_space = random.choice(self.gene_space)

                # Assinging the selected value from the space to the gene.
                offspring[offspring_idx, gene_idx] = self.gene_type(value_from_space)
                    
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
                        if type(curr_gene_space) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            value_from_space = random.choice(values_to_select_from)

                    # Assigning the selected value from the space to the gene.
                    offspring[offspring_idx, gene_idx] = self.gene_type(value_from_space)

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
                    offspring[offspring_idx, gene_idx] = self.gene_type(random_value)
                # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    offspring[offspring_idx, gene_idx] = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
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
                        offspring[offspring_idx, gene_idx] = self.gene_type(random_value)
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
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
        if self.keep_parents == 0:
            temp_population = offspring
            num_parents = 0
        else:
            if self.keep_parents == -1:
                num_parents = self.num_parents_mating
            else:
                num_parents = self.keep_parents
            parents = self.steady_state_selection(fitness, num_parents=num_parents)
            temp_population[0:parents.shape[0], :] = parents
            temp_population[parents.shape[0]:, :] = offspring

        for idx, sol in enumerate(temp_population):
            fitness[idx] = self.fitness_func(sol, None)
        average_fitness = numpy.mean(fitness)

        return average_fitness, fitness[num_parents:]

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
                    if type(curr_gene_space) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                    else:
                        # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                        values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                high=self.gene_space['high'],
                                                                size=1)
                    else:
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                        value_from_space = random.choice(values_to_select_from)

                # Assinging the selected value from the space to the gene.
                offspring[offspring_idx, gene_idx] = self.gene_type(value_from_space)

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
                    offspring[offspring_idx, gene_idx] = self.gene_type(random_value)
                # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    offspring[offspring_idx, gene_idx] = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
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
                        if type(curr_gene_space) in [int, float, numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float, numpy.float16, numpy.float32, numpy.float64]:
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
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))
                            value_from_space = random.choice(values_to_select_from)

                    # Assinging the selected value from the space to the gene.
                    offspring[offspring_idx, gene_idx] = self.gene_type(value_from_space)

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
                        offspring[offspring_idx, gene_idx] = self.gene_type(random_value)
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type(offspring[offspring_idx, gene_idx] + random_value)
        return offspring

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

#        if self.generations_completed < 1:
#            raise RuntimeError("The best_solution() method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            raise ValueError("Error calling the best_solution() method: \nThe run() method is not yet called and thus the GA did not evolve the solutions. Thus, the best solution is retireved from the initial random population without being evolved.\n")

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        if pop_fitness is None:
            pop_fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(pop_fitness == numpy.max(pop_fitness))[0][0]

        best_solution = self.population[best_match_idx, :].copy()
        best_solution_fitness = pop_fitness[best_match_idx]

        return best_solution, best_solution_fitness, best_match_idx

    def plot_result(self, title="PyGAD - Iteration vs. Fitness", xlabel="Generation", ylabel="Fitness", linewidth=3):

        """
        Creates and shows a plot that summarizes how the fitness value evolved by generation. Can only be called after completing at least 1 generation. If no generation is completed, an exception is raised.

        Accepts the following:
            title: Figure title.
            xlabel: Label on the X-axis.
            ylabel: Label on the Y-axis.
            linewidth: Line width of the plot.

        Returns the figure.
        """

        if self.generations_completed < 1:
            raise RuntimeError("The plot_result() method can only be called after completing at least 1 generation but ({generations_completed}) is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            if not self.suppress_warnings: warnings.warn("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        fig = matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_solutions_fitness, linewidth=linewidth)
        matplotlib.pyplot.title(title)
        matplotlib.pyplot.xlabel(xlabel)
        matplotlib.pyplot.ylabel(ylabel)
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
        raise BaseException("Error loading the file. Please check if the file exists.")
    return ga_in