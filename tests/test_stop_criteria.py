import pygad
import numpy

actual_num_fitness_calls_default_keep = 0
actual_num_fitness_calls_no_keep = 0
actual_num_fitness_calls_keep_elitism = 0
actual_num_fitness_calls_keep_parents = 0

num_generations = 100
sol_per_pop = 10
num_parents_mating = 5

def multi_objective_problem(keep_elitism=1, 
                            keep_parents=-1,
                            fitness_batch_size=None,
                            stop_criteria=None,
                            parent_selection_type='sss',
                            mutation_type="random",
                            mutation_percent_genes="default",
                            multi_objective=False):

    function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
    function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
    desired_output1 = 50 # Function 1 output.
    desired_output2 = 30 # Function 2 output.

    def fitness_func_batch_multi(ga_instance, solution, solution_idx):
        f = []
        for sol in solution:
            output1 = numpy.sum(sol*function_inputs1)
            output2 = numpy.sum(sol*function_inputs2)
            fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
            fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
            f.append([fitness1, fitness2])
        return f

    def fitness_func_no_batch_multi(ga_instance, solution, solution_idx):
        output1 = numpy.sum(solution*function_inputs1)
        output2 = numpy.sum(solution*function_inputs2)
        fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
        fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
        return [fitness1, fitness2]

    def fitness_func_batch_single(ga_instance, solution, solution_idx):
        f = []
        for sol in solution:
            output = numpy.sum(solution*function_inputs1)
            fitness = 1.0 / (numpy.abs(output - desired_output1) + 0.000001)
            f.append(fitness)
        return f

    def fitness_func_no_batch_single(ga_instance, solution, solution_idx):
        output = numpy.sum(solution*function_inputs1)
        fitness = 1.0 / (numpy.abs(output - desired_output1) + 0.000001)
        return fitness

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
                            fitness_batch_size=fitness_batch_size,
                            mutation_type=mutation_type,
                            mutation_percent_genes=mutation_percent_genes,
                            keep_elitism=keep_elitism,
                            keep_parents=keep_parents,
                            stop_criteria=stop_criteria,
                            parent_selection_type=parent_selection_type,
                            suppress_warnings=True)

    ga_optimizer.run()
    
    return ga_optimizer.generations_completed, ga_optimizer.best_solutions_fitness, ga_optimizer.last_generation_fitness, stop_criteria

def test_number_calls_fitness_function_default_keep():
    multi_objective_problem()

def test_number_calls_fitness_function_stop_criteria_reach(multi_objective=False,
                                                           fitness_batch_size=None, 
                                                           num=10):
    generations_completed, best_solutions_fitness, last_generation_fitness, stop_criteria = multi_objective_problem(multi_objective=multi_objective,
                                                                                           fitness_batch_size=fitness_batch_size,
                                                                                           stop_criteria=f"reach_{num}")
    # Verify that the GA stops when meeting the criterion.
    criterion = stop_criteria.split('_')
    stop_word = criterion[0]
    if generations_completed < num_generations:
        if stop_word == 'reach':
            if len(criterion) > 2:
                # multi-objective problem.
                for idx, num in enumerate(criterion[1:]):
                    criterion[idx + 1] = float(num)
            else:
                criterion[1] = float(criterion[1])

            # Single-objective
            if type(last_generation_fitness[0]) in pygad.GA.supported_int_float_types:
                assert max(last_generation_fitness) >= criterion[1]
            # Multi-objective
            elif type(last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                # Validate the value passed to the criterion.
                if len(criterion[1:]) == 1:
                    # There is a single value used across all the objectives.
                    pass
                elif len(criterion[1:]) > 1:
                    # There are multiple values. The number of values must be equal to the number of objectives.
                    if len(criterion[1:]) == len(last_generation_fitness[0]):
                        pass
                    else:
                        raise ValueError("Error")

                for obj_idx in range(len(last_generation_fitness[0])):
                    # Use the objective index to return the proper value for the criterion.
                    if len(criterion[1:]) == len(last_generation_fitness[0]):
                        reach_fitness_value = criterion[obj_idx + 1]
                    elif len(criterion[1:]) == 1:
                        reach_fitness_value = criterion[1]

                    assert max(last_generation_fitness[:, obj_idx]) >= reach_fitness_value

def test_number_calls_fitness_function_stop_criteria_saturate(multi_objective=False,
                                                              fitness_batch_size=None,
                                                              num=5):
    generations_completed, best_solutions_fitness, last_generation_fitness, stop_criteria = multi_objective_problem(multi_objective=multi_objective,
                                                                                           fitness_batch_size=fitness_batch_size,
                                                                                           stop_criteria=f"saturate_{num}")
    # Verify that the GA stops when meeting the criterion.
    criterion = stop_criteria.split('_')
    stop_word = criterion[0]
    number = criterion[1]
    if generations_completed < num_generations:
        if stop_word == 'saturate':
            number = int(number)
            if type(last_generation_fitness[0]) in pygad.GA.supported_int_float_types:
                assert best_solutions_fitness[generations_completed - number] == best_solutions_fitness[generations_completed - 1]
            elif type(last_generation_fitness[0]) in [list, tuple, numpy.ndarray]:
                for obj_idx in range(len(best_solutions_fitness[0])):
                    assert best_solutions_fitness[generations_completed - number][obj_idx] == best_solutions_fitness[generations_completed - 1][obj_idx]

if __name__ == "__main__":
    #### Single-objective problem with a single numeric value with stop_criteria.
    print()
    test_number_calls_fitness_function_default_keep()
    print()
    test_number_calls_fitness_function_stop_criteria_reach()
    print()
    test_number_calls_fitness_function_stop_criteria_reach(num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate()
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(fitness_batch_size=4)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(fitness_batch_size=4,
                                                           num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(fitness_batch_size=4)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(fitness_batch_size=4,
                                                              num=2)
    print()


    #### Multi-objective problem with a single numeric value with stop_criteria.
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(multi_objective=True)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(multi_objective=True, 
                                                              num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           fitness_batch_size=4)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           fitness_batch_size=4,
                                                           num=2)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(multi_objective=True, 
                                                              fitness_batch_size=4)
    print()
    test_number_calls_fitness_function_stop_criteria_saturate(multi_objective=True, 
                                                              fitness_batch_size=4,
                                                              num=50)
    print()


    #### Multi-objective problem with multiple numeric values with stop_criteria.
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           num="2_5")
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           fitness_batch_size=4)
    print()
    test_number_calls_fitness_function_stop_criteria_reach(multi_objective=True, 
                                                           fitness_batch_size=4,
                                                           num="10_20")

