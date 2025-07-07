import pygad
import numpy

"""
An example of using the gene_constraint parameter.

"""

function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

ga_instance = pygad.GA(num_generations=100,
                       num_parents_mating=5,
                       sol_per_pop=10,
                       num_genes=len(function_inputs),
                       mutation_num_genes=6,
                       fitness_func=fitness_func,
                       allow_duplicate_genes=False,
                       gene_space=range(100),
                       gene_type=int,
                       sample_size=100,
                       random_seed=10,
                       gene_constraint=[lambda x, v: [val for val in v if val>=98],
                                        lambda x, v: [val for val in v if val>=98],
                                        lambda x, v: [val for val in v if 80<val<90],
                                        None,
                                        lambda x, v: [val for val in v if 20<val<40],
                                        None]
                       )

ga_instance.run()
print(ga_instance.population)
