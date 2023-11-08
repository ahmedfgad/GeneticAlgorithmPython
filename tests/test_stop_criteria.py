import pygad
import numpy

actual_num_fitness_calls_default_keep = 0
actual_num_fitness_calls_no_keep = 0
actual_num_fitness_calls_keep_elitism = 0
actual_num_fitness_calls_keep_parents = 0

num_generations = 50
sol_per_pop = 10
num_parents_mating = 5

function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
desired_output1 = 50 # Function 1 output.
desired_output2 = 30 # Function 2 output.

#### Define the fitness functions in the top-level of the module so that they are picklable and usable in the process-based parallel processing works.
#### If the functions are defined inside a class/method/function, they are not picklable and this error is raised: AttributeError: Can't pickle local object
#### Process-based parallel processing must have the used functions picklable.
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


def multi_objective_problem(keep_elitism=1, 
                            keep_parents=-1,
                            fitness_batch_size=None,
                            stop_criteria=None,
                            parent_selection_type='sss',
                            mutation_type="random",
                            mutation_percent_genes="default",
                            multi_objective=False,
                            parallel_processing=None):

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
                            parallel_processing=parallel_processing,
                            suppress_warnings=True)

    ga_optimizer.run()
    
    return None

def test_number_calls_fitness_function_no_parallel_processing():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=None)

def test_number_calls_fitness_function_parallel_processing_thread_1():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['thread', 1])

def test_number_calls_fitness_function_parallel_processing_thread_2():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['thread', 2])

def test_number_calls_fitness_function_parallel_processing_thread_5():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['thread', 5])

def test_number_calls_fitness_function_parallel_processing_thread_5_patch_4():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=4,
                            parallel_processing=['thread', 5])

def test_number_calls_fitness_function_parallel_processing_thread_5_patch_4_multi_objective():
    multi_objective_problem(multi_objective=True,
                            fitness_batch_size=4,
                            parallel_processing=['thread', 5])

def test_number_calls_fitness_function_parallel_processing_process_1():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['process', 1])

def test_number_calls_fitness_function_parallel_processing_process_2():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['process', 2])

def test_number_calls_fitness_function_parallel_processing_process_5():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=None,
                            parallel_processing=['process', 5])

def test_number_calls_fitness_function_parallel_processing_process_5_patch_4():
    multi_objective_problem(multi_objective=False,
                            fitness_batch_size=4,
                            parallel_processing=['process', 5])

def test_number_calls_fitness_function_parallel_processing_process_5_patch_4_multi_objective():
    multi_objective_problem(multi_objective=True,
                            fitness_batch_size=4,
                            parallel_processing=['process', 5])

if __name__ == "__main__":
    print()
    test_number_calls_fitness_function_no_parallel_processing()
    print()

    #### Thread-based Parallel Processing
    test_number_calls_fitness_function_parallel_processing_thread_1()
    print()
    test_number_calls_fitness_function_parallel_processing_thread_2()
    print()
    test_number_calls_fitness_function_parallel_processing_thread_5()
    print()
    test_number_calls_fitness_function_parallel_processing_thread_5_patch_4()
    print()
    test_number_calls_fitness_function_parallel_processing_thread_5_patch_4_multi_objective()
    print()

    #### Thread-based Parallel Processing
    test_number_calls_fitness_function_parallel_processing_process_1()
    print()
    test_number_calls_fitness_function_parallel_processing_process_2()
    print()
    test_number_calls_fitness_function_parallel_processing_process_5()
    print()
    test_number_calls_fitness_function_parallel_processing_process_5_patch_4()
    print()
    test_number_calls_fitness_function_parallel_processing_process_5_patch_4_multi_objective()
    print()
