from functools import partial

import numpy
import pygad


def fitness_func(ga_instance, solution, solution_idx, target):
    return 1.0 / (1.0 + abs(numpy.sum(solution) - target))


def mutation_func(offspring, ga_instance):
    # Use the GA's own generator so random_seed also covers this operator.
    for solution in offspring:
        gene_index = ga_instance.numpy_random_generator.randint(ga_instance.num_genes)
        lower, upper = ga_instance.get_initial_population_range(gene_index)
        solution[gene_index] = ga_instance.numpy_random_generator.uniform(lower, upper)
    return offspring


def create_ga(random_seed):
    return pygad.GA(num_generations=numpy.uint8(5),
                    num_parents_mating=numpy.int64(2),
                    fitness_func=partial(fitness_func, target=3.0),
                    sol_per_pop=6,
                    num_genes=3,
                    init_range_low=0.0,
                    init_range_high=2.0,
                    gene_type=[float, 2],
                    mutation_type=mutation_func,
                    mutation_num_genes=1,
                    random_seed=random_seed)


first_ga = create_ga(numpy.int64(7))
second_ga = create_ga(7)

first_ga.run()

# Another GA and global random draws do not change second_ga's generator states.
other_ga = create_ga(99)
other_ga.run()
numpy.random.seed(100)
numpy.random.random(10)

second_ga.run()
numpy.testing.assert_array_equal(first_ga.population, second_ga.population)
print("Independent instances with the same seed produce the same population.")
print(first_ga.population)

# Count arithmetic uses Python integers even when NumPy counts are supplied.
first_ga.run()
print("Generations after two runs:", first_ga.generations_completed)
