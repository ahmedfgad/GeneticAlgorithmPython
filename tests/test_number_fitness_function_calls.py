import pygad
import random
import numpy

actual_num_fitness_calls_default_keep = 0
actual_num_fitness_calls_no_keep = 0
actual_num_fitness_calls_keep_elitism = 0
actual_num_fitness_calls_keep_parents = 0

num_generations = 100
sol_per_pop = 10
num_parents_mating = 5

# TODO: Calculate the number when fitness_batch_size is used.

def number_calls_fitness_function(keep_elitism=1, 
                                  keep_parents=-1,
                                  mutation_type="random",
                                  mutation_percent_genes="default",
                                  parent_selection_type='sss',
                                  multi_objective=False,
                                  fitness_batch_size=None):

    actual_num_fitness_calls = 0
    def fitness_func_no_batch_single(ga, solution, idx):
        nonlocal actual_num_fitness_calls
        actual_num_fitness_calls = actual_num_fitness_calls + 1
        return random.random()

    def fitness_func_no_batch_multi(ga_instance, solution, solution_idx):
        nonlocal actual_num_fitness_calls
        actual_num_fitness_calls = actual_num_fitness_calls + 1
        return [random.random(), random.random()]

    def fitness_func_batch_single(ga_instance, solution, solution_idx):
        nonlocal actual_num_fitness_calls
        actual_num_fitness_calls = actual_num_fitness_calls + 1
        f = []
        for sol in solution:
            f.append(random.random())
        return f

    def fitness_func_batch_multi(ga_instance, solution, solution_idx):
        nonlocal actual_num_fitness_calls
        actual_num_fitness_calls = actual_num_fitness_calls + 1
        f = []
        for sol in solution:
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

    ga_optimizer = pygad.GA(num_generations=num_generations,
                            sol_per_pop=sol_per_pop,
                            num_genes=6,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func,
                            mutation_type=mutation_type,
                            parent_selection_type=parent_selection_type,
                            mutation_percent_genes=mutation_percent_genes,
                            keep_elitism=keep_elitism,
                            keep_parents=keep_parents,
                            suppress_warnings=True,
                            fitness_batch_size=fitness_batch_size)

    ga_optimizer.run()

    if fitness_batch_size is None:
        if keep_elitism == 0:
            if keep_parents == 0:
                # 10 (for initial population) + 100*10 (for other generations) = 1010
                expected_num_fitness_calls = sol_per_pop + num_generations * sol_per_pop
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * sol_per_pop
            elif keep_parents == -1:
                # 10 (for initial population) + 100*num_parents_mating (for other generations)
                expected_num_fitness_calls = sol_per_pop + num_generations * (sol_per_pop - num_parents_mating)
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * (sol_per_pop - num_parents_mating)
            else:
                # 10 (for initial population) + 100*keep_parents (for other generations)
                expected_num_fitness_calls = sol_per_pop + num_generations * (sol_per_pop - keep_parents)
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * (sol_per_pop - keep_parents)
        else:
            # 10 (for initial population) + 100*keep_elitism (for other generations)
            expected_num_fitness_calls = sol_per_pop + num_generations * (sol_per_pop - keep_elitism)
            if mutation_type == "adaptive":
                expected_num_fitness_calls += num_generations * (sol_per_pop - keep_elitism)
    else:
        if keep_elitism == 0:
            if keep_parents == 0:
                # 10 (for initial population) + 100*10 (for other generations) = 1010
                expected_num_fitness_calls = int(numpy.ceil(sol_per_pop/fitness_batch_size)) + num_generations * int(numpy.ceil(sol_per_pop/fitness_batch_size))
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * int(numpy.ceil(sol_per_pop/fitness_batch_size))
            elif keep_parents == -1:
                # 10 (for initial population) + 100*num_parents_mating (for other generations)
                expected_num_fitness_calls = int(numpy.ceil(sol_per_pop/fitness_batch_size)) + num_generations * int(numpy.ceil((sol_per_pop - num_parents_mating)/fitness_batch_size))
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * int(numpy.ceil((sol_per_pop - num_parents_mating)/fitness_batch_size))
            else:
                # 10 (for initial population) + 100*keep_parents (for other generations)
                expected_num_fitness_calls = int(numpy.ceil(sol_per_pop/fitness_batch_size)) + num_generations * int(numpy.ceil((sol_per_pop - keep_parents)/fitness_batch_size))
                if mutation_type == "adaptive":
                    expected_num_fitness_calls += num_generations * int(numpy.ceil((sol_per_pop - keep_parents)/fitness_batch_size))
        else:
            # 10 (for initial population) + 100*keep_elitism (for other generations)
            expected_num_fitness_calls = int(numpy.ceil(sol_per_pop/fitness_batch_size)) + num_generations * int(numpy.ceil((sol_per_pop - keep_elitism)/fitness_batch_size))
            if mutation_type == "adaptive":
                expected_num_fitness_calls += num_generations * int(numpy.ceil((sol_per_pop - keep_elitism)/fitness_batch_size))

    print(f"Expected number of fitness function calls is {expected_num_fitness_calls}.")
    print(f"Actual number of fitness function calls is {actual_num_fitness_calls}.")
    return actual_num_fitness_calls, expected_num_fitness_calls

def test_number_calls_fitness_function_default_keep():
    actual, expected = number_calls_fitness_function()
    assert actual == expected

def test_number_calls_fitness_function_no_keep():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5])
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5])
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5])
    assert actual == expected

#### Multi Objective
def test_number_calls_fitness_function_no_keep_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation_multi_objective():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected

#### Multi Objective NSGA-II Parent Selection
def test_number_calls_fitness_function_no_keep_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_nsga2():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True)
    assert actual == expected


######## Batch Fitness Calculation
#### Single Objective
def test_number_calls_fitness_function_no_keep_batch_1():
    actual, expected = number_calls_fitness_function(fitness_batch_size=1)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_batch_4():
    actual, expected = number_calls_fitness_function(fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_batch_9():
    actual, expected = number_calls_fitness_function(fitness_batch_size=9)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_batch_10():
    actual, expected = number_calls_fitness_function(fitness_batch_size=10)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation_batch_4():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     fitness_batch_size=4)
    assert actual == expected

#### Multi Objective
def test_number_calls_fitness_function_no_keep_multi_objective_batch_1():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=1)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_batch_9():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=9)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_batch_10():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=10)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

#### Multi Objective NSGA-II Parent Selection
def test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_1():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=1)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_9():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=9)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_10():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=10)
    assert actual == expected

def test_number_calls_fitness_function_keep_elitism_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=0,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_keep_parents_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=4,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=0, 
                                                     keep_parents=0,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected

def test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_nsga2_batch_4():
    actual, expected = number_calls_fitness_function(keep_elitism=3, 
                                                     keep_parents=4,
                                                     mutation_type="adaptive",
                                                     mutation_percent_genes=[10, 5],
                                                     multi_objective=True,
                                                     fitness_batch_size=4)
    assert actual == expected


if __name__ == "__main__":
    #### Single-objective
    print()
    test_number_calls_fitness_function_default_keep()
    print()
    test_number_calls_fitness_function_no_keep()
    print()
    test_number_calls_fitness_function_keep_elitism()
    print()
    test_number_calls_fitness_function_keep_parents()
    print()
    test_number_calls_fitness_function_both_keep()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation()
    print()

    #### Multi-Objective
    print()
    test_number_calls_fitness_function_no_keep_multi_objective()
    print()
    test_number_calls_fitness_function_keep_elitism_multi_objective()
    print()
    test_number_calls_fitness_function_keep_parents_multi_objective()
    print()
    test_number_calls_fitness_function_both_keep_multi_objective()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation_multi_objective()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective()
    print()

    #### Multi-Objective NSGA-II Parent Selection
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_keep_elitism_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_keep_parents_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_both_keep_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_nsga2()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_nsga2()
    print()


    ######## Batch Fitness
    #### Single-objective
    print()
    test_number_calls_fitness_function_no_keep_batch_1()
    print()
    test_number_calls_fitness_function_no_keep_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_batch_9()
    print()
    test_number_calls_fitness_function_no_keep_batch_10()
    print()
    test_number_calls_fitness_function_keep_elitism_batch_4()
    print()
    test_number_calls_fitness_function_keep_parents_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation_batch_4()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation_batch_4()
    print()

    #### Multi-Objective
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_batch_1()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_batch_9()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_batch_10()
    print()
    test_number_calls_fitness_function_keep_elitism_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_keep_parents_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_batch_4()
    print()

    #### Multi-Objective NSGA-II Parent Selection
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_1()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_9()
    print()
    test_number_calls_fitness_function_no_keep_multi_objective_nsga2_batch_10()
    print()
    test_number_calls_fitness_function_keep_elitism_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_keep_parents_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_no_keep_adaptive_mutation_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_default_adaptive_mutation_multi_objective_nsga2_batch_4()
    print()
    test_number_calls_fitness_function_both_keep_adaptive_mutation_multi_objective_nsga2_batch_4()
    print()

