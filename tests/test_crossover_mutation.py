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

def output_crossover_mutation(gene_space=None,
                              gene_type=float,
                              num_genes=10,
                              mutation_by_replacement=False,
                              random_mutation_min_val=-1,
                              random_mutation_max_val=1,
                              init_range_low=-4,
                              init_range_high=4,
                              initial_population=None,
                              crossover_probability=None,
                              mutation_probability=None,
                              crossover_type=None,
                              mutation_type=None,
                              parent_selection_type='sss',
                              multi_objective=False):

    def fitness_func_no_batch_single(ga, solution, idx):
        return random.random()

    def fitness_func_no_batch_multi(ga, solution, idx):
        return [random.random(), random.random()]

    if multi_objective == True:
        fitness_func = fitness_func_no_batch_multi
    else:
        fitness_func = fitness_func_no_batch_single

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           gene_type=gene_type,
                           parent_selection_type=parent_selection_type,
                           initial_population=initial_population,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=True,
                           mutation_by_replacement=mutation_by_replacement,
                           save_solutions=True,
                           crossover_probability=crossover_probability,
                           mutation_probability=mutation_probability,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           suppress_warnings=True,
                           random_seed=1)

    ga_instance.run()

    comparison_result = []
    for solution_idx, solution in enumerate(ga_instance.population):
        if list(solution) in ga_instance.initial_population.tolist():
            comparison_result.append(True)
        else:
            comparison_result.append(False)

    comparison_result = numpy.array(comparison_result)
    result = numpy.all(comparison_result == True)

    print(f"Comparison result is {result}")
    return result, ga_instance

def test_no_crossover_no_mutation():
    result, ga_instance = output_crossover_mutation()

    assert result == True

def test_no_crossover_no_mutation_gene_space():
    result, ga_instance = output_crossover_mutation(gene_space=range(10))

    assert result == True

def test_no_crossover_no_mutation_int_gene_type():
    result, ga_instance = output_crossover_mutation(gene_type=int)

    assert result == True


def test_no_crossover_no_mutation_gene_space_gene_type():
    result, ga_instance = output_crossover_mutation(gene_space={"low": 0, "high": 10},
                                                    gene_type=[float, 2])

    assert result == True


def test_no_crossover_no_mutation_nested_gene_space():
    result, ga_instance = output_crossover_mutation(gene_space=[[0, 1, 2, 3, 4], 
                                                                numpy.arange(5, 10), 
                                                                range(10, 15),
                                                                {"low": 15, "high": 20},
                                                                {"low": 20, "high": 30, "step": 2},
                                                                None,
                                                                numpy.arange(30, 35),
                                                                numpy.arange(35, 40),
                                                                numpy.arange(40, 45),
                                                                [45, 46, 47, 48, 49]])
    assert result == True

def test_no_crossover_no_mutation_nested_gene_type():
    result, ga_instance = output_crossover_mutation(gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert result == True

def test_no_crossover_no_mutation_nested_gene_space_nested_gene_type():
    result, ga_instance = output_crossover_mutation(gene_space=[[0, 1, 2, 3, 4], 
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

    assert result == True

def test_no_crossover_no_mutation_initial_population():
    global initial_population
    result, ga_instance = output_crossover_mutation(initial_population=initial_population)

    assert result == True

def test_no_crossover_no_mutation_initial_population_nested_gene_type():
    global initial_population
    result, ga_instance = output_crossover_mutation(initial_population=initial_population,
                                                    gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert result == True

def test_crossover_no_mutation_zero_crossover_probability():
    global initial_population
    result, ga_instance = output_crossover_mutation(crossover_type="single_point",
                                                    crossover_probability=0.0)

    assert result == True

def test_zero_crossover_probability_zero_mutation_probability():
    global initial_population
    result, ga_instance = output_crossover_mutation(crossover_type="single_point",
                                                    crossover_probability=0.0,
                                                    mutation_type="random",
                                                    mutation_probability=0.0)

    assert result == True

def test_random_mutation_manual_call():
    result, ga_instance = output_crossover_mutation(mutation_type="random",
                                                    random_mutation_min_val=888,
                                                    random_mutation_max_val=999)
    ga_instance.mutation_num_genes = 9

    temp_offspring = numpy.array(initial_population[0:1])
    offspring = ga_instance.random_mutation(offspring=temp_offspring.copy())

    comp = offspring - temp_offspring
    comp_sorted = sorted(comp.copy())
    comp_sorted = numpy.abs(numpy.unique(comp_sorted))

    # The other 1 added to include the last value in the range.
    assert len(comp_sorted) in range(1, 1 + 1 + ga_instance.mutation_num_genes)
    assert comp_sorted[0] == 0

def test_random_mutation_manual_call2():
    result, ga_instance = output_crossover_mutation(mutation_type="random",
                                                    random_mutation_min_val=888,
                                                    random_mutation_max_val=999)
    ga_instance.mutation_num_genes = 10

    temp_offspring = numpy.array(initial_population[0:1])
    offspring = ga_instance.random_mutation(offspring=temp_offspring.copy())

    comp = offspring - temp_offspring
    comp_sorted = sorted(comp.copy())
    comp_sorted = numpy.abs(numpy.unique(comp_sorted))

    # The other 1 added to include the last value in the range.
    assert len(comp_sorted) in range(1, 1 + 1 + ga_instance.mutation_num_genes)
    # assert comp_sorted[0] == 0

def test_random_mutation_manual_call3():
    # Use random_mutation_min_val & random_mutation_max_val as numbers.
    random_mutation_min_val = 888
    random_mutation_max_val = 999
    result, ga_instance = output_crossover_mutation(mutation_type="random",
                                                    random_mutation_min_val=random_mutation_min_val,
                                                    random_mutation_max_val=random_mutation_max_val,
                                                    mutation_by_replacement=True)
    ga_instance.mutation_num_genes = 10

    temp_offspring = numpy.array(initial_population[0:1])
    offspring = ga_instance.random_mutation(offspring=temp_offspring.copy())

    comp = offspring
    comp_sorted = sorted(comp.copy())
    comp_sorted = numpy.abs(numpy.unique(comp))

    value_space = list(range(random_mutation_min_val, random_mutation_max_val))
    for value in comp_sorted:
        assert value in value_space

def test_random_mutation_manual_call4():
    # Use random_mutation_min_val & random_mutation_max_val as lists.
    random_mutation_min_val = [888]*10
    random_mutation_max_val = [999]*10
    result, ga_instance = output_crossover_mutation(mutation_type="random",
                                                    random_mutation_min_val=random_mutation_min_val,
                                                    random_mutation_max_val=random_mutation_max_val,
                                                    mutation_by_replacement=True)
    ga_instance.mutation_num_genes = 10

    temp_offspring = numpy.array(initial_population[0:1])
    offspring = ga_instance.random_mutation(offspring=temp_offspring.copy())

    comp = offspring
    comp_sorted = sorted(comp.copy())
    comp_sorted = numpy.abs(numpy.unique(comp))

    value_space = list(range(random_mutation_min_val[0], random_mutation_max_val[0]))
    for value in comp_sorted:
        assert value in value_space

if __name__ == "__main__":
    #### Single-objective
    print()
    test_no_crossover_no_mutation()
    print()

    test_no_crossover_no_mutation_int_gene_type()
    print()

    test_no_crossover_no_mutation_gene_space()
    print()

    test_no_crossover_no_mutation_gene_space_gene_type()
    print()

    test_no_crossover_no_mutation_nested_gene_space()
    print()

    test_no_crossover_no_mutation_nested_gene_type()
    print()

    test_no_crossover_no_mutation_initial_population()
    print()

    test_no_crossover_no_mutation_initial_population_nested_gene_type()
    print()

    test_crossover_no_mutation_zero_crossover_probability()
    print()

    test_zero_crossover_probability_zero_mutation_probability()
    print()

    test_random_mutation_manual_call()
    print()

    test_random_mutation_manual_call2()
    print()

    test_random_mutation_manual_call3()
    print()

    test_random_mutation_manual_call4()
    print()
