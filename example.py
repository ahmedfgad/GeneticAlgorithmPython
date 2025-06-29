import pygad
import numpy

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

num_genes = len(function_inputs)

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=10,
                       sol_per_pop=20,
                       num_genes=num_genes,
                       mutation_num_genes=6,
                       fitness_func=fitness_func,
                       init_range_low=4,
                       init_range_high=10,
                       # suppress_warnings=True,
                       random_mutation_min_val=4,
                       random_mutation_max_val=10,
                       mutation_by_replacement=True,
                       gene_type=int,
                       # mutation_probability=0.4,
                       gene_constraint=[lambda x: x[0]>=8,None,None,None,None,None])

# ga_instance.run()
