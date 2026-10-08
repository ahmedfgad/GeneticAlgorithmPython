"""Replace a loaded fitness function and start fresh when the objective changes."""

import numpy
import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


def updated_fitness_func(ga_instance, solution, solution_idx):
    # The formula is unchanged, so previously calculated fitness remains valid.
    fitness = float(numpy.sum(solution))
    print(f"Updated fitness function: solution {solution_idx}, fitness {fitness}")
    return fitness


def new_objective_fitness_func(ga_instance, solution, solution_idx):
    # This different objective needs fresh fitness values for every solution.
    return -float(numpy.sum(solution ** 2))


if __name__ == "__main__":
    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=2,
                           sol_per_pop=6,
                           num_genes=4,
                           fitness_func=fitness_func,
                           mutation_num_genes=1,
                           random_seed=17)
    ga_instance.run()
    ga_instance.save("saved_ga")

    # Loading restores the saved function, even if the script has been edited.
    loaded_ga_instance = pygad.load("saved_ga")
    loaded_ga_instance.fitness_func = updated_fitness_func
    loaded_ga_instance.run()

    # Reuse the chromosomes with a fresh GA when changing the objective.
    new_ga_instance = pygad.GA(num_generations=2,
                               num_parents_mating=2,
                               initial_population=loaded_ga_instance.population.copy(),
                               fitness_func=new_objective_fitness_func,
                               mutation_num_genes=1,
                               random_seed=17)
    new_ga_instance.run()
    solution, solution_fitness, solution_idx = new_ga_instance.best_solution(
        pop_fitness=new_ga_instance.last_generation_fitness)
    print(f"New objective: best fitness {solution_fitness}")
