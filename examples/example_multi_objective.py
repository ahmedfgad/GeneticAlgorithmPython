import pygad
import numpy

"""
Given these 2 functions:
    y1 = f(w1:w6) = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + 6wx6
    y2 = f(w1:w6) = w1x7 + w2x8 + w3x9 + w4x10 + w5x11 + 6wx12
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7) and y=50
    and   (x7,x8,x9,x10,x11,x12)=(-2,0.7,-9,1.4,3,5) and y=30
What are the best values for the 6 weights (w1 to w6)? We are going to use the genetic algorithm to optimize these 2 functions.
This is a multi-objective optimization problem.

PyGAD considers the problem as multi-objective if the fitness function returns:
    1) List.
    2) Or tuple.
    3) Or numpy.ndarray.
"""

function_inputs1 = [4,-2,3.5,5,-11,-4.7] # Function 1 inputs.
function_inputs2 = [-2,0.7,-9,1.4,3,5] # Function 2 inputs.
desired_output1 = 50 # Function 1 output.
desired_output2 = 30 # Function 2 output.

def fitness_func(ga_instance, solution, solution_idx):
    output1 = numpy.sum(solution*function_inputs1)
    output2 = numpy.sum(solution*function_inputs2)
    fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
    fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
    return [fitness1, fitness2]

num_generations = 100 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 20 # Number of solutions in the population.
num_genes = len(function_inputs1)

last_fitness = 0
def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       parent_selection_type='nsga2',
                       on_generation=on_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

ga_instance.plot_fitness(labels=['Obj 1', 'Obj 2'])

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

prediction = numpy.sum(numpy.array(function_inputs1)*solution)
print(f"Predicted output 1 based on the best solution : {prediction}")
prediction = numpy.sum(numpy.array(function_inputs2)*solution)
print(f"Predicted output 2 based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

