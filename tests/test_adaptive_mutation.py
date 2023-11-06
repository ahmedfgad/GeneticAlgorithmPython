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

def output_adaptive_mutation(gene_space=None,
                             gene_type=float,
                             num_genes=10,
                             mutation_by_replacement=False,
                             random_mutation_min_val=-1,
                             random_mutation_max_val=1,
                             init_range_low=-4,
                             init_range_high=4,
                             initial_population=None,
                             mutation_probability=[0.2, 0.1],
                             fitness_batch_size=None,
                             mutation_type="adaptive"):

    def fitness_func_single(ga, solution, idx):
        return random.random()

    def fitness_func_batch(ga, soluions, idxs):
        return numpy.random.uniform(size=len(soluions))
    
    if fitness_batch_size in [1, None]:
        fitness_func = fitness_func_single
    else:
        fitness_func = fitness_func_batch

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
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=True,
                           mutation_by_replacement=mutation_by_replacement,
                           save_solutions=True,
                           mutation_probability=mutation_probability,
                           mutation_type=mutation_type,
                           suppress_warnings=True,
                           fitness_batch_size=fitness_batch_size,
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

if __name__ == "__main__":
    print()
    test_adaptive_mutation()
    print()

    test_adaptive_mutation_int_gene_type()
    print()

    test_adaptive_mutation_gene_space()
    print()

    test_adaptive_mutation_gene_space_gene_type()
    print()

    test_adaptive_mutation_nested_gene_space()
    print()

    test_adaptive_mutation_nested_gene_type()
    print()

    test_adaptive_mutation_initial_population()
    print()

    test_adaptive_mutation_initial_population_nested_gene_type()
    print()

    test_adaptive_mutation_fitness_batch_size_1()
    print()

    test_adaptive_mutation_fitness_batch_size_1()
    print()

    test_adaptive_mutation_fitness_batch_size_2()
    print()

    test_adaptive_mutation_fitness_batch_size_3()
    print()

    test_adaptive_mutation_fitness_batch_size_4()
    print()

    test_adaptive_mutation_fitness_batch_size_5()
    print()

    test_adaptive_mutation_fitness_batch_size_6()
    print()

    test_adaptive_mutation_fitness_batch_size_7()
    print()

    test_adaptive_mutation_fitness_batch_size_8()
    print()

    test_adaptive_mutation_fitness_batch_size_9()
    print()

    test_adaptive_mutation_fitness_batch_size_10()
    print()

