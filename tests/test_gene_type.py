import pygad
import random
import numpy

num_generations = 5

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

def validate_gene_type_and_rounding(gene_space=None,
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
                                    gene_constraint=None,
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
                           gene_constraint=gene_constraint,
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

    for sol_idx in range(len(ga_instance.solutions)):
        for gene_idx in range(ga_instance.num_genes):
            if ga_instance.gene_type_single:
                dtype = ga_instance.gene_type
            else:
                dtype = ga_instance.gene_type[gene_idx]

            if dtype[0] is float:
                # NumPy converts the Python float data type to numpy.float64. Both are identical.
                assert type(ga_instance.solutions[sol_idx][gene_idx]) in [dtype[0], numpy.float64]
            else:
                assert type(ga_instance.solutions[sol_idx][gene_idx]) is dtype[0]

            if dtype[1] is None:
                pass
            else:
                num_decimals = len(str(ga_instance.solutions[sol_idx][gene_idx]).split('.')[1])
                # The numbers may not have the exact precision.
                # For example, the float number might be 5.7, and we set the precision to 2.
                # Because there is no enough digits after the decimal point, we cannot meet the precision of 2.
                # We only care about not exceeding the user-defined precision.
                assert num_decimals <= dtype[1]

    return ga_instance

def test_nested_gene_type():
    ga_instance = validate_gene_type_and_rounding(gene_type=[numpy.int32,
                                                             numpy.float16,
                                                             numpy.float32,
                                                             [numpy.float16, 3],
                                                             [numpy.float32, 4],
                                                             numpy.int16,
                                                             [numpy.float32, 1],
                                                             numpy.int32,
                                                             numpy.float16,
                                                             numpy.float64])

def test_single_gene_type_float16():
    ga_instance = validate_gene_type_and_rounding(gene_type=[numpy.float16, 2])

def test_single_gene_type_int32():
    ga_instance = validate_gene_type_and_rounding(gene_type=numpy.int32)

def test_single_gene_space_single_gene_type():
    ga_instance = validate_gene_type_and_rounding(gene_space={"low": 0, "high": 10},
                                                  gene_type=[float, 2])

def test_nested_gene_space_single_gene_type():
    ga_instance = validate_gene_type_and_rounding(gene_space=[[0, 1, 2, 3, 4],
                                                              numpy.arange(5, 10),
                                                              range(10, 15),
                                                              {"low": 15, "high": 20},
                                                              {"low": 20, "high": 30, "step": 2},
                                                              None,
                                                              numpy.arange(30, 35),
                                                              numpy.arange(35, 40),
                                                              numpy.arange(40, 45),
                                                              [45, 46, 47, 48, 49]],
                                                  gene_type=[numpy.float16, 1])

def test_nested_gene_space_nested_gene_type():
    ga_instance = validate_gene_type_and_rounding(gene_space=[[0, 1, 2, 3, 4],
                                                              numpy.arange(5, 10),
                                                              range(10, 15),
                                                              {"low": 15, "high": 20},
                                                              {"low": 20, "high": 30, "step": 2},
                                                              None,
                                                              numpy.arange(30, 35),
                                                              numpy.arange(35, 40),
                                                              numpy.arange(40, 45),
                                                              [45, 46, 47, 48, 49]],
                                                  gene_type=[int,
                                                             float,
                                                             numpy.float64,
                                                             [float, 3],
                                                             [float, 4],
                                                             numpy.int16,
                                                             [numpy.float32, 1],
                                                             int,
                                                             float,
                                                             [float, 3]])

def test_single_gene_space_nested_gene_type():
    ga_instance = validate_gene_type_and_rounding(gene_space=numpy.arange(0, 100),
                                                  gene_type=[int,
                                                             float,
                                                             numpy.float64,
                                                             [float, 3],
                                                             [float, 4],
                                                             numpy.int16,
                                                             [numpy.float32, 1],
                                                             int,
                                                             float,
                                                             [float, 3]])

def test_custom_initial_population_single_gene_type():
    global initial_population
    ga_instance = validate_gene_type_and_rounding(initial_population=initial_population,
                                                  gene_type=[numpy.float16, 2])

def test_custom_initial_population_nested_gene_type():
    global initial_population
    ga_instance = validate_gene_type_and_rounding(initial_population=initial_population,
                                                  gene_type=[int,
                                                             float,
                                                             numpy.float64,
                                                             [float, 3],
                                                             [float, 4],
                                                             numpy.int16,
                                                             [numpy.float32, 1],
                                                             int,
                                                             float,
                                                             [float, 3]])

if __name__ == "__main__":
    #### Single-objective
    print()
    test_nested_gene_type()
    print()

    test_single_gene_type_float16()
    print()

    test_single_gene_type_int32()
    print()

    test_single_gene_space_single_gene_type()
    print()

    test_nested_gene_space_single_gene_type()
    print()

    test_nested_gene_space_nested_gene_type()
    print()

    test_single_gene_space_nested_gene_type()
    print()

    test_custom_initial_population_single_gene_type()
    print()

    test_custom_initial_population_nested_gene_type()
    print()
