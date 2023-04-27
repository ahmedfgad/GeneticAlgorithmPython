import pygad

actual_num_fitness_calls_default_keep = 0
actual_num_fitness_calls_no_keep = 0
actual_num_fitness_calls_keep_elitism = 0
actual_num_fitness_calls_keep_parents = 0

num_generations = 100
sol_per_pop = 10
num_parents_mating = 5

def number_calls_fitness_function(keep_elitism=1, 
                                  keep_parents=-1,
                                  mutation_type="random",
                                  mutation_percent_genes="default"):

    actual_num_fitness_calls = 0
    def fitness_func(ga, solution, idx):
        nonlocal actual_num_fitness_calls
        actual_num_fitness_calls = actual_num_fitness_calls + 1
        return 1

    ga_optimizer = pygad.GA(num_generations=num_generations,
                            sol_per_pop=sol_per_pop,
                            num_genes=6,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes,
                            keep_elitism=keep_elitism,
                            keep_parents=keep_parents,
                            suppress_warnings=True)

    ga_optimizer.run()
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

    print("Expected number of fitness function calls is {expected_num_fitness_calls}.".format(expected_num_fitness_calls=expected_num_fitness_calls))
    print("Actual number of fitness function calls is {actual_num_fitness_calls}.".format(actual_num_fitness_calls=actual_num_fitness_calls))
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

if __name__ == "__main__":
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
