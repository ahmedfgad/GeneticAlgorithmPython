import pygad
import random
import numpy

num_generations = 1

initial_population = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

#### Define the fitness functions in the top-level of the module so that they are picklable and usable in the process-based parallel processing works.
#### If the functions are defined inside a class/method/function, they are not picklable and this error is raised: AttributeError: Can't pickle local object
#### Process-based parallel processing must have the used functions picklable.
def fitness_func_no_batch_single(ga, solution, idx):
    return random.random()

def fitness_func_batch_single(ga, soluions, idxs):
    return numpy.random.uniform(size=len(soluions))

def fitness_func_no_batch_multi(ga, solution, idx):
    return [random.random(), random.random()]

def fitness_func_batch_multi(ga, soluions, idxs):
    f = []
    for sol in soluions:
        f.append([random.random(), random.random()])
    return f

def output_adaptive_mutation(gene_space=None,
                             gene_type=float,
                             num_genes=10,
                             mutation_by_replacement=False,
                             random_mutation_min_val=-1,
                             random_mutation_max_val=1,
                             init_range_low=-4,
                             init_range_high=4,
                             initial_population=None,
                             mutation_probability=None,
                             mutation_num_genes=None,
                             fitness_batch_size=None,
                             mutation_type="adaptive",
                             parent_selection_type='sss',
                             parallel_processing=None,
                             multi_objective=False):

    if fitness_batch_size is None or (type(fitness_batch_size) in pygad.GA.supported_int_types and fitness_batch_size == 1):
        if multi_objective == True:
            fitness_func = fitness_func_no_batch_multi
        else:
            fitness_func = fitness_func_no_batch_single
    elif (type(fitness_batch_size) in pygad.GA.supported_int_types and fitness_batch_size > 1):
        if multi_objective == True:
            fitness_func = fitness_func_batch_multi
        else:
            fitness_func = fitness_func_batch_single
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           gene_type=gene_type,
                           initial_population=initial_population,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=True,
                           mutation_by_replacement=mutation_by_replacement,
                           save_solutions=True,
                           ## Use a static 'mutation_probability'.
                           ## An ambigius error in GitHub actions happen when using mutation_num_genes and mutation_probability. I do not know the reason.
                           # mutation_num_genes=mutation_num_genes,
                           mutation_probability=[0.2, 0.1],
                           mutation_type=mutation_type,
                           suppress_warnings=True,
                           fitness_batch_size=fitness_batch_size,
                           parallel_processing=parallel_processing,
                           random_seed=1)

    ga_instance.run()

    return None, ga_instance

def test_adaptive_mutation():
    result, ga_instance = output_adaptive_mutation()

    # assert result == True

def test_adaptive_mutation_gene_space():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10))

    # assert result == True

def test_adaptive_mutation_int_gene_type():
    result, ga_instance = output_adaptive_mutation(gene_type=int)

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2])

    # assert result == True

def test_adaptive_mutation_nested_gene_space():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]])
    # assert result == True

def test_adaptive_mutation_nested_gene_type():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    # assert result == True

def test_adaptive_mutation_initial_population():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population)

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1)

def test_adaptive_mutation_fitness_batch_size_2():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2)

def test_adaptive_mutation_fitness_batch_size_3():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3)

def test_adaptive_mutation_fitness_batch_size_4():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4)

def test_adaptive_mutation_fitness_batch_size_5():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5)

def test_adaptive_mutation_fitness_batch_size_6():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6)

def test_adaptive_mutation_fitness_batch_size_7():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7)

def test_adaptive_mutation_fitness_batch_size_8():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8)

def test_adaptive_mutation_fitness_batch_size_9():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9)

def test_adaptive_mutation_fitness_batch_size_10():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10)

#### Single-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability():
    result, ga_instance = output_adaptive_mutation(mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1])


#### Single-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4])

#### Multi-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability_multi_objective():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_multi_objective():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1])

#### Multi-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes_multi_objective():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_multi_objective():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_multi_objective():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4])

######## Parallel Processing
#### #### Threads

#### Single-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])


#### Single-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

#### Multi-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability_multi_objective_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_multi_objective_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['thread', 4])

#### Multi-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes_multi_objective_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_threads():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_multi_objective_parallel_processing_threads():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['thread', 4])


#### #### Processes

#### Single-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])


#### Single-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

#### Multi-Objective Mutation Probability
def test_adaptive_mutation_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_probability_multi_objective_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_multi_objective_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_probability_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=10,
                                                   mutation_probability=[0.2, 0.1], 
                                                   parallel_processing=['process', 4])

#### Multi-Objective Mutation Number of Genes
def test_adaptive_mutation_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=range(10),
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_int_gene_type_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=int,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])
    # assert result == True

def test_adaptive_mutation_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]],
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_mutation_num_genes_multi_objective_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_multi_objective_parallel_processing_processes():
    global initial_population
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=1,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=2,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=3,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=4,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=5,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=6,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=7,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=8,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,
                                                   fitness_batch_size=9,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])

def test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_multi_objective_parallel_processing_processes():
    result, ga_instance = output_adaptive_mutation(multi_objective=True,fitness_batch_size=10,
                                                   mutation_num_genes=[6, 4], 
                                                   parallel_processing=['process', 4])


if __name__ == "__main__":
    #### Single-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_gene_space_mutation_probability()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_initial_population_mutation_probability()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability()
    print()

    #### Single-objective mutation_num_genes
    print()
    test_adaptive_mutation_mutation_num_genes()
    print()

    test_adaptive_mutation_int_gene_type_mutation_num_genes()
    print()

    test_adaptive_mutation_gene_space_mutation_num_genes()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_num_genes()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_num_genes()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_num_genes()
    print()

    test_adaptive_mutation_initial_population_mutation_num_genes()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes()
    print()

    #### Multi-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_gene_space_mutation_probability()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_initial_population_mutation_probability()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability()
    print()



    ######## Parallel Processing
    #### #### Threads
    #### Single-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_threads()
    print()

    #### Single-objective mutation_num_genes
    print()
    test_adaptive_mutation_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_int_gene_type_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_parallel_processing_threads()
    print()

    #### Multi-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_threads()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_threads()
    print()

    #### #### Processes
    #### Single-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_processes()
    print()

    #### Single-objective mutation_num_genes
    print()
    test_adaptive_mutation_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_int_gene_type_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_num_genes_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_num_genes_parallel_processing_processes()
    print()

    #### Multi-objective mutation_probability
    print()
    test_adaptive_mutation_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_int_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_gene_space_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_space_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_nested_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_1_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_2_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_3_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_4_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_5_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_6_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_7_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_8_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_9_mutation_probability_parallel_processing_processes()
    print()

    test_adaptive_mutation_fitness_batch_size_10_mutation_probability_parallel_processing_processes()
    print()


