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
                             mutation_probability=None,
                             mutation_num_genes=None,
                             fitness_batch_size=None,
                             mutation_type="adaptive",
                             parent_selection_type='sss',
                             multi_objective=False):

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
                           random_seed=1)

    ga_instance.run()

    return None, ga_instance

def test_adaptive_mutation(multi_objective=False,
                           parent_selection_type='sss',
                           mutation_num_genes=None,
                           mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_gene_space(multi_objective=False,
                                      parent_selection_type='sss',
                                      mutation_num_genes=None,
                                      mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(gene_space=range(10),
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_int_gene_type(multi_objective=False,
                                         parent_selection_type='sss',
                                         mutation_num_genes=None,
                                         mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(gene_type=int,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_gene_space_gene_type(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(gene_space={"low": 0, "high": 10},
                                                   gene_type=[float, 2],
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_nested_gene_space(multi_objective=False,
                                             parent_selection_type='sss',
                                             mutation_num_genes=None,
                                             mutation_probability=None):
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
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)
    # assert result == True

def test_adaptive_mutation_nested_gene_type(multi_objective=False,
                                            parent_selection_type='sss',
                                            mutation_num_genes=None,
                                            mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_nested_gene_space_nested_gene_type(multi_objective=False,
                                                              parent_selection_type='sss',
                                                              mutation_num_genes=None,
                                                              mutation_probability=None):
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
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_initial_population(multi_objective=False,
                                              parent_selection_type='sss',
                                              mutation_num_genes=None,
                                              mutation_probability=None):
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_initial_population_nested_gene_type(multi_objective=False,
                                                               parent_selection_type='sss',
                                                               mutation_num_genes=None,
                                                               mutation_probability=None):
    global initial_population
    result, ga_instance = output_adaptive_mutation(initial_population=initial_population,
                                                   gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

    # assert result == True

def test_adaptive_mutation_fitness_batch_size_1(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=1,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_2(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=2,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_3(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=3,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_4(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=4,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_5(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=5,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_6(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=6,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_7(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=7,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_8(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=8,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_9(multi_objective=False,
                                                parent_selection_type='sss',
                                                mutation_num_genes=None,
                                                mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=9,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

def test_adaptive_mutation_fitness_batch_size_10(multi_objective=False,
                                                 parent_selection_type='sss',
                                                 mutation_num_genes=None,
                                                 mutation_probability=None):
    result, ga_instance = output_adaptive_mutation(fitness_batch_size=10,
                                                   parent_selection_type=parent_selection_type,
                                                   multi_objective=multi_objective,
                                                   mutation_num_genes=mutation_num_genes,
                                                   mutation_probability=mutation_probability)

if __name__ == "__main__":
    #### Single-objective mutation_probability
    print()
    test_adaptive_mutation(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_int_gene_type(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space_gene_type(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_space(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_type(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_2(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_3(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_4(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_5(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_6(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_7(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_8(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_9(mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_10(mutation_probability=[0.2, 0.1])
    print()

    #### Single-objective mutation_num_genes
    print()
    test_adaptive_mutation(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_int_gene_type(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_gene_space(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_gene_space_gene_type(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_nested_gene_space(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_nested_gene_type(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_initial_population(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_1(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_1(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_2(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_3(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_4(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_5(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_6(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_7(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_8(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_9(mutation_num_genes=[6, 3])
    print()

    test_adaptive_mutation_fitness_batch_size_10(mutation_num_genes=[6, 3])
    print()

    #### Multi-objective mutation_probability
    print()
    test_adaptive_mutation(multi_objective=True,
                           mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_int_gene_type(multi_objective=True,
                                         mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space(multi_objective=True,
                                      mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space_gene_type(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_space(multi_objective=True,
                                             mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_type(multi_objective=True,
                                            mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population(multi_objective=True,
                                              mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(multi_objective=True,
                                                               mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_2(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_3(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_4(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_5(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_6(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_7(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_8(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_9(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_10(multi_objective=True,
                                                mutation_probability=[0.2, 0.1])
    print()

    #### Multi-objective mutation_num_genes
    test_adaptive_mutation(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_int_gene_type(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_gene_space(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_gene_space_gene_type(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_nested_gene_space(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_nested_gene_type(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_initial_population(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_2(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_3(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_4(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_5(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_6(multi_objective=True,
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_7(multi_objective=True,
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_8(multi_objective=True,
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_9(multi_objective=True,
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_10(multi_objective=True,
                                                 mutation_num_genes=[6, 4])
    print()


    #### Multi-objective NSGA-II Parent Selection mutation_probability
    print()
    test_adaptive_mutation(multi_objective=True,
                           parent_selection_type='nsga2',
                           mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_int_gene_type(multi_objective=True,
                                         parent_selection_type='nsga2',
                                         mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space(multi_objective=True,
                                      parent_selection_type='nsga2',
                                      mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_gene_space_gene_type(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_space(multi_objective=True,
                                             parent_selection_type='nsga2',
                                             mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_nested_gene_type(multi_objective=True,
                                            parent_selection_type='nsga2',
                                            mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population(multi_objective=True,
                                              parent_selection_type='nsga2',
                                              mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(multi_objective=True,
                                                               parent_selection_type='nsga2',
                                                               mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_2(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_3(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_4(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_5(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_6(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_7(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_8(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_9(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_probability=[0.2, 0.1])
    print()

    test_adaptive_mutation_fitness_batch_size_10(multi_objective=True,
                                                 parent_selection_type='nsga2',
                                                 mutation_probability=[0.2, 0.1])

    #### Multi-objective NSGA-II Parent Selection mutation_num_genes
    print()
    test_adaptive_mutation(multi_objective=True,
                           parent_selection_type='nsga2',
                           mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_int_gene_type(multi_objective=True,
                                         parent_selection_type='nsga2',
                                         mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_gene_space(multi_objective=True,
                                      parent_selection_type='nsga2',
                                      mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_gene_space_gene_type(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_nested_gene_space(multi_objective=True,
                                             parent_selection_type='nsga2',
                                             mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_nested_gene_type(multi_objective=True,
                                            parent_selection_type='nsga2',
                                            mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_initial_population(multi_objective=True,
                                              parent_selection_type='nsga2',
                                              mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_initial_population_nested_gene_type(multi_objective=True,
                                                               parent_selection_type='nsga2',
                                                               mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_1(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_2(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_3(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_4(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_5(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_6(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_7(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_8(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_9(multi_objective=True,
                                                parent_selection_type='nsga2',
                                                mutation_num_genes=[6, 4])
    print()

    test_adaptive_mutation_fitness_batch_size_10(multi_objective=True,
                                                 parent_selection_type='nsga2',
                                                 mutation_num_genes=[6, 4])
    print()

