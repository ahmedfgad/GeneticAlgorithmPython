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
                       init_range_low=1,
                       init_range_high=100,
                       # suppress_warnings=True,
                       random_mutation_min_val=1,
                       random_mutation_max_val=100,
                       mutation_by_replacement=True,
                       gene_type=[float, 1],
                       save_solutions=True,
                       allow_duplicate_genes=False,
                       # gene_space=numpy.unique(numpy.random.uniform(1, 100, size=100)),
                       gene_space=[range(10), {"low": 1, "high": 5}, 2.5891221, [1,2,3,4], None, numpy.unique(numpy.random.uniform(1, 100, size=4))],
                       gene_constraint=[lambda x: x[0]>=98,lambda x: x[1]>=98,lambda x: x[2]<98,lambda x: x[3]<98,lambda x: x[4]<98,lambda x: x[5]<98],
                       )

ga_instance.run()

print(ga_instance.gene_space_unpacked)
print(ga_instance.population)
