import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

num_generations = 100 # Number of generations.
num_parents_mating = 10 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 20 # Number of solutions in the population.
num_genes = len(function_inputs)

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       gene_type=int,
                       gene_space=range(1000),
                       mutation_by_replacement=True,
                       sample_size=1000000,
                       gene_constraint=[lambda x: x[0]>5, lambda x: x[0]>5, lambda x: x[0]>5, lambda x: x[0]>5, lambda x: x[0]>5, lambda x: x[0]>5],
                       allow_duplicate_genes=False)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# ga_instance.plot_fitness()
