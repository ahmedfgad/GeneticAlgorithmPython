import numpy
import pygad


def fitness_func(ga_instance, solution, solution_idx):
    # Keep the integer exact while comparing it with the target.
    integer_error = abs(solution[0] - (2**53 + 1))
    return 1.0 / (1.0 + integer_error + abs(solution[1] - 1.25))


def mutation_func(offspring, ga_instance):
    # Object arrays preserve mixed integer and floating-point values.
    offspring = offspring.copy()
    offspring[:, 1] = [float(value) + 0.126 for value in offspring[:, 1]]
    # PyGAD applies each gene's type and precision to the returned values.
    return offspring


initial_population = [[2**53 + 1, 1.236, 3.9],
                      [2**53 + 2, 1.754, 4.1],
                      [2**53 + 3, 2.345, 5.8],
                      [2**53 + 4, 2.876, 6.2]]

ga_instance = pygad.GA(num_generations=3,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       initial_population=initial_population,
                       gene_type=[int, [numpy.float32, 2], numpy.int8],
                       mutation_type=mutation_func,
                       mutation_num_genes=1,
                       save_best_solutions=True,
                       random_seed=7)

print("Initial Population")
print(ga_instance.initial_population)

ga_instance.run()

print("Final Population")
print(ga_instance.population)
print("Gene Types")
print([type(value) for value in ga_instance.population[0]])
print("Saved Best Solutions")
print(ga_instance.best_solutions)
