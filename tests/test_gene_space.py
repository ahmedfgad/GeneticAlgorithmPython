"""
This script is identical to the test_gene_space_allow_duplicate_genes.py script except for:
    Setting allow_duplicate_genes=True instead of False.
"""

import pygad
import random
import numpy

num_generations = 100

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

# Test single gene space with nested gene type.

def number_respect_gene_space(gene_space=None,
                              gene_type=float,
                              num_genes=10,
                              mutation_by_replacement=False,
                              random_mutation_min_val=-1,
                              random_mutation_max_val=1,
                              init_range_low=-4,
                              init_range_high=4,
                              mutation_type="random",
                              mutation_percent_genes="default",
                              mutation_probability=None,
                              initial_population=None,
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
                           initial_population=initial_population,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=True,
                           mutation_by_replacement=mutation_by_replacement,
                           parent_selection_type=parent_selection_type,
                           save_solutions=True,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           mutation_probability=mutation_probability,
                           suppress_warnings=True,
                           random_seed=2)

    ga_instance.run()
    ga_instance.solutions = numpy.array(ga_instance.solutions, 
                                        dtype=object)

    # gene_space_unpacked = ga_instance.unpack_gene_space(num_values_from_inf_range=100)
    num_outside = 0
    if ga_instance.gene_space_nested == True:
        for gene_idx in range(ga_instance.num_genes):

            if type(ga_instance.init_range_low) in ga_instance.supported_int_float_types:
                range_min_init = ga_instance.init_range_low
                range_max_init = ga_instance.init_range_high
            else:
                range_min_init = ga_instance.init_range_low[gene_idx]
                range_max_init = ga_instance.init_range_high[gene_idx]
            if type(ga_instance.random_mutation_min_val) in ga_instance.supported_int_float_types:
                range_min_mutation = ga_instance.random_mutation_min_val
                range_max_mutation = ga_instance.random_mutation_max_val
            else:
                range_min_mutation = ga_instance.random_mutation_min_val[gene_idx]
                range_max_mutation = ga_instance.random_mutation_max_val[gene_idx]

            all_gene_values = ga_instance.solutions[:, gene_idx]
            if type(ga_instance.gene_space[gene_idx]) in [list, tuple, range, numpy.ndarray]:
                current_gene_space = list(ga_instance.gene_space[gene_idx])
                # print("current_gene_space", current_gene_space)
                for val_idx, val in enumerate(all_gene_values):
                    if None in current_gene_space:
                        if (val in current_gene_space) or (val >= range_min_init and val < range_max_init) or (val >= range_min_mutation and val < range_max_mutation):
                            pass
                        else:
                            # print("###########")
                            # print(gene_idx, val)
                            # print(current_gene_space)
                            # print(range_min_mutation, range_max_mutation)
                            # print("\n\n")
                            num_outside += 1
                    elif val in current_gene_space:
                        # print("val, current_gene_space", val, current_gene_space)
                        pass
                    else:
                        # print(gene_idx, val, current_gene_space)
                        num_outside += 1
            elif type(ga_instance.gene_space[gene_idx]) is dict:
                if not "step" in ga_instance.gene_space[gene_idx].keys():
                    for val in all_gene_values:
                        if val >= ga_instance.gene_space[gene_idx]["low"] and val < ga_instance.gene_space[gene_idx]["high"]:
                            pass
                        else:
                            print(gene_idx, val, current_gene_space)
                            num_outside += 1
                else:
                    gene_space_values = numpy.arange(ga_instance.gene_space[gene_idx]["low"],
                                                     ga_instance.gene_space[gene_idx]["high"],
                                                     ga_instance.gene_space[gene_idx]["step"])
                    for val in all_gene_values:
                        if val in gene_space_values:
                            pass
                        else:
                            num_outside += 1
            elif type(ga_instance.gene_space[gene_idx]) in ga_instance.supported_int_float_types:
                for val in all_gene_values:
                    if val == ga_instance.gene_space[gene_idx]:
                        pass
                    else:
                        num_outside += 1
            elif ga_instance.gene_space[gene_idx] is None:
                for val in all_gene_values:
                    # print(val)
                    if (val >= range_min_init and val < range_max_init) or (val >= range_min_mutation and val < range_max_mutation):
                        pass
                    else:
                        # print("###########")
                        # print(gene_idx, val)
                        # print(ga_instance.gene_space[gene_idx])
                        # print(range_min_init, range_max_init)
                        # print(range_min_mutation, range_max_mutation)
                        # print("\n\n")
                        num_outside += 1
    else:
        for gene_idx in range(ga_instance.num_genes):

            if type(ga_instance.init_range_low) in ga_instance.supported_int_float_types:
                range_min_init = ga_instance.init_range_low
                range_max_init = ga_instance.init_range_high
            else:
                range_min_init = ga_instance.init_range_low[gene_idx]
                range_max_init = ga_instance.init_range_high[gene_idx]
            if type(ga_instance.random_mutation_min_val) in ga_instance.supported_int_float_types:
                range_min_mutation = ga_instance.random_mutation_min_val
                range_max_mutation = ga_instance.random_mutation_max_val
            else:
                range_min_mutation = ga_instance.random_mutation_min_val[gene_idx]
                range_max_mutation = ga_instance.random_mutation_max_val[gene_idx]

            all_gene_values = ga_instance.solutions[:, gene_idx]
            # print("all_gene_values", gene_idx, all_gene_values)
            if type(ga_instance.gene_space) in [list, tuple, range, numpy.ndarray]:
                current_gene_space = list(ga_instance.gene_space)
                for val in all_gene_values:
                    if None in current_gene_space:
                        if (val in current_gene_space) or (val >= range_min_init and val < range_max_init) or (val >= range_min_mutation and val < range_max_mutation):
                            pass
                        else:
                            # print("###########")
                            # print(gene_idx, val)
                            # print(current_gene_space)
                            # print(range_min_mutation, range_max_mutation)
                            # print("\n\n")
                            num_outside += 1
                    elif val in current_gene_space:
                        pass
                    else:
                        num_outside += 1
            elif type(ga_instance.gene_space) is dict:
                if not "step" in ga_instance.gene_space.keys():
                    for val in all_gene_values:
                        if val >= ga_instance.gene_space["low"] and val < ga_instance.gene_space["high"]:
                            pass
                        else:
                            num_outside += 1
                else:
                    gene_space_values = numpy.arange(ga_instance.gene_space["low"],
                                                     ga_instance.gene_space["high"],
                                                     ga_instance.gene_space["step"])
                    for val in all_gene_values:
                        if val in gene_space_values:
                            pass
                        else:
                            num_outside += 1

    print(f"Number of outside range is {num_outside}.")
    return num_outside, ga_instance

#### Single-Objective
def test_gene_space_range():
    num_outside, _ = number_respect_gene_space(gene_space=range(10))
    
    assert num_outside == 0

def test_gene_space_numpy_arange():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10))

    assert num_outside == 0

def test_gene_space_list():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)))
    
    assert num_outside == 0

def test_gene_space_list_None():
    num_outside, _ = number_respect_gene_space(gene_space=[30, None, 40, 50, None, 60, 70, None, None, None])
    
    assert num_outside == 0

def test_gene_space_numpy():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))))
    
    assert num_outside == 0

def test_gene_space_dict_without_step():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10})

    assert num_outside == 0

def test_gene_space_dict_with_step():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2})

    assert num_outside == 0

def test_gene_space_list_single_value():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5])

    assert num_outside == 0

def test_gene_space_range_nested_gene_type():
    num_outside, _ = number_respect_gene_space(gene_space=range(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])
    
    assert num_outside == 0

def test_gene_space_numpy_arange_nested_gene_type():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])
    
    assert num_outside == 0

def test_gene_space_list_nested_gene_type():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])
    
    assert num_outside == 0

def test_gene_space_numpy_nested_gene_type():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])
    
    assert num_outside == 0

def test_gene_space_dict_without_step_nested_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert num_outside == 0

def test_gene_space_dict_with_step_nested_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert num_outside == 0

def test_gene_space_list_single_value_nested_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert num_outside == 0

def test_nested_gene_space_range():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[range(0, 10),
                                                                     range(10, 20),
                                                                     range(20, 30),
                                                                     range(30, 40),
                                                                     range(40, 50),
                                                                     range(50, 60),
                                                                     range(60, 70),
                                                                     range(70, 80),
                                                                     range(80, 90),
                                                                     range(90, 100)])

    assert num_outside == 0

def test_nested_gene_space_dict_without_step():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}])

    assert num_outside == 0

def test_nested_gene_space_dict_without_step_float_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}],
                                                         gene_type=[float, 3])

    assert num_outside == 0

def test_nested_gene_space_dict_with_step():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10, "step": 1},
                                                                     {"low": 10, "high": 20, "step": 1.5},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     {"low": 30, "high": 40, "step": 2.5},
                                                                     {"low": 40, "high": 50, "step": 3},
                                                                     {"low": 50, "high": 60, "step": 3.5},
                                                                     {"low": 60, "high": 70, "step": 4},
                                                                     {"low": 70, "high": 80, "step": 4.5},
                                                                     {"low": 80, "high": 90, "step": 5},
                                                                     {"low": 90, "high": 100, "step": 5.5}])

    assert num_outside == 0


def test_nested_gene_space_numpy_arange():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[numpy.arange(0, 10),
                                                                     numpy.arange(10, 20),
                                                                     numpy.arange(20, 30),
                                                                     numpy.arange(30, 40),
                                                                     numpy.arange(40, 50),
                                                                     numpy.arange(50, 60),
                                                                     numpy.arange(60, 70),
                                                                     numpy.arange(70, 80),
                                                                     numpy.arange(80, 90),
                                                                     numpy.arange(90, 100)])

    assert num_outside == 0

def test_nested_gene_space_list():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                     [-10, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                                                     [-11, 11, 22, 33, 44, 55, 66, 77, 88, 99],
                                                                     [-100, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                                                                     [-4.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                                     [-5.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                                                                     [-10.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
                                                                     [-15, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                                                                     [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                                                                     [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])

    assert num_outside == 0

def test_nested_gene_space_list2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1], 
                                                                     [1, 2], 
                                                                     [2, 3],
                                                                     [3, 4],
                                                                     [4, 5],
                                                                     [5, 6],
                                                                     [6, 7],
                                                                     [7, 8],
                                                                     [8, 9],
                                                                     [9, 10]])

    assert num_outside == 0

def test_nested_gene_space_list3_None():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         mutation_by_replacement=True)

    assert num_outside == 0

def test_nested_gene_space_list4_None_custom_mutation_range():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         random_mutation_min_val=20,
                                                         random_mutation_max_val=40,
                                                         mutation_by_replacement=True)

    assert num_outside == 0

def test_nested_gene_space_mix():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=int,
                                                         mutation_by_replacement=True)

    assert num_outside == 0

def test_nested_gene_space_mix_nested_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], int, numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 00, "high": 10, "step": 1},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]])

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population_single_gene_type():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[float, 4])

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive")

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation_probability():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive")

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive")

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation_probability():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive")
    assert num_outside == 0

#### Multi-Objective
def test_gene_space_range_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=range(10),
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_numpy_arange_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10),
                                               multi_objective=True)

    assert num_outside == 0

def test_gene_space_list_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)),
                                               multi_objective=True)

    assert num_outside == 0

def test_gene_space_list_None_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=[30, None, 40, 50, None, 60, 70, None, None, None],
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_numpy_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))),
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_dict_without_step_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10},
                                                         multi_objective=True)

    assert num_outside == 0

def test_gene_space_dict_with_step_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2},
                                                         multi_objective=True)

    assert num_outside == 0

def test_gene_space_list_single_value_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5],
                                                         multi_objective=True)

    assert num_outside == 0

def test_gene_space_range_nested_gene_type_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=range(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_numpy_arange_nested_gene_type_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_list_nested_gene_type_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_numpy_nested_gene_type_multi_objective():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True)
    
    assert num_outside == 0

def test_gene_space_dict_without_step_nested_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_gene_space_dict_with_step_nested_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_gene_space_list_single_value_nested_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_range_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[range(0, 10),
                                                                     range(10, 20),
                                                                     range(20, 30),
                                                                     range(30, 40),
                                                                     range(40, 50),
                                                                     range(50, 60),
                                                                     range(60, 70),
                                                                     range(70, 80),
                                                                     range(80, 90),
                                                                     range(90, 100)],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_dict_without_step_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_dict_without_step_float_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}],
                                                         gene_type=[float, 3],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_dict_with_step_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10, "step": 1},
                                                                     {"low": 10, "high": 20, "step": 1.5},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     {"low": 30, "high": 40, "step": 2.5},
                                                                     {"low": 40, "high": 50, "step": 3},
                                                                     {"low": 50, "high": 60, "step": 3.5},
                                                                     {"low": 60, "high": 70, "step": 4},
                                                                     {"low": 70, "high": 80, "step": 4.5},
                                                                     {"low": 80, "high": 90, "step": 5},
                                                                     {"low": 90, "high": 100, "step": 5.5}],
                                                         multi_objective=True)

    assert num_outside == 0


def test_nested_gene_space_numpy_arange_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[numpy.arange(0, 10),
                                                                     numpy.arange(10, 20),
                                                                     numpy.arange(20, 30),
                                                                     numpy.arange(30, 40),
                                                                     numpy.arange(40, 50),
                                                                     numpy.arange(50, 60),
                                                                     numpy.arange(60, 70),
                                                                     numpy.arange(70, 80),
                                                                     numpy.arange(80, 90),
                                                                     numpy.arange(90, 100)],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_list_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                     [-10, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                                                     [-11, 11, 22, 33, 44, 55, 66, 77, 88, 99],
                                                                     [-100, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                                                                     [-4.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                                     [-5.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                                                                     [-10.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
                                                                     [-15, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                                                                     [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                                                                     [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_list2_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1], 
                                                                     [1, 2], 
                                                                     [2, 3],
                                                                     [3, 4],
                                                                     [4, 5],
                                                                     [5, 6],
                                                                     [6, 7],
                                                                     [7, 8],
                                                                     [8, 9],
                                                                     [9, 10]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_list3_None_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         mutation_by_replacement=True,
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_list4_None_custom_mutation_range_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         random_mutation_min_val=20,
                                                         random_mutation_max_val=40,
                                                         mutation_by_replacement=True,
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_mix_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=int,
                                                         mutation_by_replacement=True,
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_mix_nested_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], int, numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 00, "high": 10, "step": 1},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population_single_gene_type_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[float, 4],
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive",
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation_probability_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive",
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive",
                                                         multi_objective=True)

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation_probability_multi_objective():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive",
                                                         multi_objective=True)
    assert num_outside == 0

#### Multi-Objective NSGA-II Parent Selection
def test_gene_space_range_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=range(10),
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_numpy_arange_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10),
                                               multi_objective=True,
                                               parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_list_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)),
                                               multi_objective=True,
                                               parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_list_None_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=[30, None, 40, 50, None, 60, 70, None, None, None],
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_numpy_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))),
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_dict_without_step_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10},
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_dict_with_step_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2},
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_list_single_value_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_range_nested_gene_type_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=range(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_numpy_arange_nested_gene_type_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.arange(10),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_list_nested_gene_type_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=list(range(10)),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_numpy_nested_gene_type_multi_objective_nsga2():
    num_outside, _ = number_respect_gene_space(gene_space=numpy.array(list(range(10))),
                                               gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                               multi_objective=True,
                                               parent_selection_type='nsga2')
    
    assert num_outside == 0

def test_gene_space_dict_without_step_nested_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_dict_with_step_nested_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space={"low": 0, "high": 10, "step": 2},
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_gene_space_list_single_value_nested_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[5],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_range_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[range(0, 10),
                                                                     range(10, 20),
                                                                     range(20, 30),
                                                                     range(30, 40),
                                                                     range(40, 50),
                                                                     range(50, 60),
                                                                     range(60, 70),
                                                                     range(70, 80),
                                                                     range(80, 90),
                                                                     range(90, 100)],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_dict_without_step_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_dict_without_step_float_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10},
                                                                     {"low": 10, "high": 20},
                                                                     {"low": 20, "high": 30},
                                                                     {"low": 30, "high": 40},
                                                                     {"low": 40, "high": 50},
                                                                     {"low": 50, "high": 60},
                                                                     {"low": 60, "high": 70},
                                                                     {"low": 70, "high": 80},
                                                                     {"low": 80, "high": 90},
                                                                     {"low": 90, "high": 100}],
                                                         gene_type=[float, 3],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_dict_with_step_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[{"low": 0, "high": 10, "step": 1},
                                                                     {"low": 10, "high": 20, "step": 1.5},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     {"low": 30, "high": 40, "step": 2.5},
                                                                     {"low": 40, "high": 50, "step": 3},
                                                                     {"low": 50, "high": 60, "step": 3.5},
                                                                     {"low": 60, "high": 70, "step": 4},
                                                                     {"low": 70, "high": 80, "step": 4.5},
                                                                     {"low": 80, "high": 90, "step": 5},
                                                                     {"low": 90, "high": 100, "step": 5.5}],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0


def test_nested_gene_space_numpy_arange_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[numpy.arange(0, 10),
                                                                     numpy.arange(10, 20),
                                                                     numpy.arange(20, 30),
                                                                     numpy.arange(30, 40),
                                                                     numpy.arange(40, 50),
                                                                     numpy.arange(50, 60),
                                                                     numpy.arange(60, 70),
                                                                     numpy.arange(70, 80),
                                                                     numpy.arange(80, 90),
                                                                     numpy.arange(90, 100)],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_list_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                     [-10, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                                                                     [-11, 11, 22, 33, 44, 55, 66, 77, 88, 99],
                                                                     [-100, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                                                                     [-4.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                                     [-5.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                                                                     [-10.5, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9],
                                                                     [-15, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                                                                     [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                                                                     [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_list2_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1], 
                                                                     [1, 2], 
                                                                     [2, 3],
                                                                     [3, 4],
                                                                     [4, 5],
                                                                     [5, 6],
                                                                     [6, 7],
                                                                     [7, 8],
                                                                     [8, 9],
                                                                     [9, 10]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_list3_None_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         mutation_by_replacement=True,
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_list4_None_custom_mutation_range_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, None], 
                                                                     [1, 2], 
                                                                     [2, None],
                                                                     [3, 4],
                                                                     [None, 5],
                                                                     None,
                                                                     [None, 7],
                                                                     [None, None],
                                                                     [8, 9],
                                                                     None],
                                                         random_mutation_min_val=20,
                                                         random_mutation_max_val=40,
                                                         mutation_by_replacement=True,
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_mix_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=int,
                                                         mutation_by_replacement=True,
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_mix_nested_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4], 
                                                                     numpy.arange(5, 10), 
                                                                     range(10, 15),
                                                                     {"low": 15, "high": 20},
                                                                     {"low": 20, "high": 30, "step": 2},
                                                                     None,
                                                                     numpy.arange(30, 35),
                                                                     numpy.arange(35, 40),
                                                                     numpy.arange(40, 45),
                                                                     [45, 46, 47, 48, 49]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], int, numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 00, "high": 10, "step": 1},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_mix_initial_population_single_gene_type_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[float, 4],
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive",
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_single_gene_type_adaptive_mutation_probability_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         # Due to rounding the genes, a gene at index 4 will have a value of 10 (outside the dict range) if [float, 2] is used.
                                                         gene_type=[float, 4],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive",
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_percent_genes=[70, 50],
                                                         mutation_type="adaptive",
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')

    assert num_outside == 0

def test_nested_gene_space_nested_gene_type_adaptive_mutation_probability_multi_objective_nsga2():
    num_outside, ga_instance = number_respect_gene_space(gene_space=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                                                     numpy.arange(0, 10), 
                                                                     range(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     {"low": 0, "high": 10},
                                                                     range(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     numpy.arange(0, 10),
                                                                     {"low": 0, "high": 10},
                                                                     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                                                         gene_type=[int, float, numpy.float64, [float, 3], [float, 4], numpy.int16, [numpy.float32, 1], int, float, [float, 3]],
                                                         mutation_probability=[0.7, 0.5],
                                                         mutation_type="adaptive",
                                                         multi_objective=True,
                                                         parent_selection_type='nsga2')
    assert num_outside == 0




if __name__ == "__main__":
    #### Single-objective
    print()
    test_gene_space_range()
    print()
    test_gene_space_range_nested_gene_type()
    print()

    test_gene_space_numpy_arange()
    print()
    test_gene_space_numpy_arange_nested_gene_type()
    print()

    test_gene_space_list()
    print()
    test_gene_space_list_None()
    print()
    test_gene_space_list_nested_gene_type()
    print()

    test_gene_space_list_single_value()
    print()
    test_gene_space_list_single_value_nested_gene_type()
    print()

    test_gene_space_numpy()
    print()
    test_gene_space_numpy_nested_gene_type()
    print()

    test_gene_space_dict_without_step()
    print()
    test_gene_space_dict_without_step_nested_gene_type()
    print()

    test_gene_space_dict_with_step()
    print()
    test_gene_space_dict_with_step_nested_gene_type()
    print()

    test_nested_gene_space_range()
    print()

    test_nested_gene_space_dict_without_step()
    print()

    test_nested_gene_space_dict_without_step_float_gene_type()
    print()

    test_nested_gene_space_dict_with_step()
    print()

    test_nested_gene_space_numpy_arange()
    print()

    test_nested_gene_space_list()
    print()

    test_nested_gene_space_list2()
    print()

    test_nested_gene_space_list3_None()
    print()

    test_nested_gene_space_list4_None_custom_mutation_range()
    print()

    test_nested_gene_space_mix()
    print()

    test_nested_gene_space_mix_nested_gene_type()
    print()

    test_nested_gene_space_mix_initial_population()
    print()

    test_nested_gene_space_mix_initial_population_single_gene_type()
    print()

    test_nested_gene_space_single_gene_type_adaptive_mutation()
    print()
    test_nested_gene_space_single_gene_type_adaptive_mutation_probability()
    print()

    test_nested_gene_space_nested_gene_type_adaptive_mutation()
    print()
    test_nested_gene_space_nested_gene_type_adaptive_mutation_probability()
    print()

    #### Multi-objective
    print()
    test_gene_space_range_multi_objective()
    print()
    test_gene_space_range_nested_gene_type_multi_objective()
    print()

    test_gene_space_numpy_arange_multi_objective()
    print()
    test_gene_space_numpy_arange_nested_gene_type_multi_objective()
    print()

    test_gene_space_list_multi_objective()
    print()
    test_gene_space_list_None_multi_objective()
    print()
    test_gene_space_list_nested_gene_type_multi_objective()
    print()

    test_gene_space_list_single_value_multi_objective()
    print()
    test_gene_space_list_single_value_nested_gene_type_multi_objective()
    print()

    test_gene_space_numpy_multi_objective()
    print()
    test_gene_space_numpy_nested_gene_type_multi_objective()
    print()

    test_gene_space_dict_without_step_multi_objective()
    print()
    test_gene_space_dict_without_step_nested_gene_type_multi_objective()
    print()

    test_gene_space_dict_with_step_multi_objective()
    print()
    test_gene_space_dict_with_step_nested_gene_type_multi_objective()
    print()

    test_nested_gene_space_range_multi_objective()
    print()

    test_nested_gene_space_dict_without_step_multi_objective()
    print()

    test_nested_gene_space_dict_without_step_float_gene_type_multi_objective()
    print()

    test_nested_gene_space_dict_with_step_multi_objective()
    print()

    test_nested_gene_space_numpy_arange_multi_objective()
    print()

    test_nested_gene_space_list_multi_objective()
    print()

    test_nested_gene_space_list2_multi_objective()
    print()

    test_nested_gene_space_list3_None_multi_objective()
    print()

    test_nested_gene_space_list4_None_custom_mutation_range_multi_objective()
    print()

    test_nested_gene_space_mix_multi_objective()
    print()

    test_nested_gene_space_mix_nested_gene_type_multi_objective()
    print()

    test_nested_gene_space_mix_initial_population_multi_objective()
    print()

    test_nested_gene_space_mix_initial_population_single_gene_type_multi_objective()
    print()

    test_nested_gene_space_single_gene_type_adaptive_mutation_multi_objective()
    print()
    test_nested_gene_space_single_gene_type_adaptive_mutation_probability_multi_objective()
    print()

    test_nested_gene_space_nested_gene_type_adaptive_mutation_multi_objective()
    print()
    test_nested_gene_space_nested_gene_type_adaptive_mutation_probability_multi_objective()
    print()


    #### Multi-objective NSGA-II Parent Selection
    print()
    test_gene_space_range_multi_objective_nsga2()
    print()
    test_gene_space_range_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_numpy_arange_multi_objective_nsga2()
    print()
    test_gene_space_numpy_arange_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_list_multi_objective_nsga2()
    print()
    test_gene_space_list_None_multi_objective_nsga2()
    print()
    test_gene_space_list_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_list_single_value_multi_objective_nsga2()
    print()
    test_gene_space_list_single_value_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_numpy_multi_objective_nsga2()
    print()
    test_gene_space_numpy_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_dict_without_step_multi_objective_nsga2()
    print()
    test_gene_space_dict_without_step_nested_gene_type_multi_objective_nsga2()
    print()

    test_gene_space_dict_with_step_multi_objective_nsga2()
    print()
    test_gene_space_dict_with_step_nested_gene_type_multi_objective_nsga2()
    print()

    test_nested_gene_space_range_multi_objective_nsga2()
    print()

    test_nested_gene_space_dict_without_step_multi_objective_nsga2()
    print()

    test_nested_gene_space_dict_without_step_float_gene_type_multi_objective_nsga2()
    print()

    test_nested_gene_space_dict_with_step_multi_objective_nsga2()
    print()

    test_nested_gene_space_numpy_arange_multi_objective_nsga2()
    print()

    test_nested_gene_space_list_multi_objective_nsga2()
    print()

    test_nested_gene_space_list2_multi_objective_nsga2()
    print()

    test_nested_gene_space_list3_None_multi_objective_nsga2()
    print()

    test_nested_gene_space_list4_None_custom_mutation_range_multi_objective_nsga2()
    print()

    test_nested_gene_space_mix_multi_objective_nsga2()
    print()

    test_nested_gene_space_mix_nested_gene_type_multi_objective_nsga2()
    print()

    test_nested_gene_space_mix_initial_population_multi_objective_nsga2()
    print()

    test_nested_gene_space_mix_initial_population_single_gene_type_multi_objective_nsga2()
    print()

    test_nested_gene_space_single_gene_type_adaptive_mutation_multi_objective_nsga2()
    print()
    test_nested_gene_space_single_gene_type_adaptive_mutation_probability_multi_objective_nsga2()
    print()

    test_nested_gene_space_nested_gene_type_adaptive_mutation_multi_objective_nsga2()
    print()
    test_nested_gene_space_nested_gene_type_adaptive_mutation_probability_multi_objective_nsga2()
    print()
