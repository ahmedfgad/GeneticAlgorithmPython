"""Return one fitness value per solution, including a shorter final batch."""

import numpy
import pygad


def fitness_func(ga_instance, solutions, solutions_indices):
    print(f"Batch size: {len(solutions)}")
    return numpy.sum(solutions, axis=1)


if __name__ == "__main__":
    initial_population = numpy.arange(40, dtype=float).reshape(20, 2)
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=5,
                           initial_population=initial_population,
                           fitness_func=fitness_func,
                           fitness_batch_size=10,
                           keep_elitism=1,
                           mutation_num_genes=1,
                           random_seed=17)
    ga_instance.run()
