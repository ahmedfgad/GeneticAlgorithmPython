import pygad
import numpy

"""
All the callback functions/methods in PyGAD have limits in the number of arguments passed.
For example, the fitness function accepts only 3 arguments:
    1. The pygad.GA instance.
    2. The solution(s).
    3. The index (indices) of the passed solution(s).
If it is necessary to pass extra arguments to the fitness function, for example, then follow these steps:
    1. Create a wrapper function that accepts only the number of arguments meeded by PyGAD.
    2. Define the extra arguments in the body of the wrapper function.
    3. Create an inner fitness function inside the wrapper function with whatever extra arguments needed.
    4. Call the inner fitness function from the wrapper function while passing the extra arguments.

This is an example that passes a list ([10, 20, 30]) to the inner fitness function. The list has 3 numbers.
A number is randomly selected from the list and added to the calculated fitness.
"""

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func_wrapper(ga_instanse, solution, solution_idx):
    def fitness_func(ga_instanse, solution, solution_idx, *args):
        output = numpy.sum(solution*function_inputs)
        output += numpy.random.choice(args)
        fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
        return fitness
    args = [10, 20, 30]
    fitness = fitness_func(ga_instanse, solution, solution_idx, *args)
    return fitness

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=5,
                       fitness_func=fitness_func_wrapper,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       suppress_warnings=True)

ga_instance.run()
ga_instance.plot_fitness()
