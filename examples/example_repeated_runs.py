"""Continue a GA from a checkpoint and inspect its saved generation numbers."""

import os
import tempfile

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return int(solution[0])


def mutation_func(offspring, ga_instance):
    return offspring + 10


def main():
    ga_instance = pygad.GA(num_generations=2,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           initial_population=[[0], [1]],
                           gene_type=int,
                           crossover_type=None,
                           mutation_type=mutation_func,
                           keep_parents=0,
                           keep_elitism=0,
                           save_best_solutions=True,
                           save_solutions=True,
                           suppress_warnings=True,
                           random_seed=7)
    ga_instance.run()

    with tempfile.TemporaryDirectory() as directory:
        filename = os.path.join(directory, 'repeated_runs')
        ga_instance.save(filename)
        resumed_instance = pygad.load(filename)
        resumed_instance.run()

    print('Completed generations:', resumed_instance.generations_completed)
    print('Best solution generation:', resumed_instance.best_solution_generation)
    print('Saved best generation numbers:', resumed_instance.best_solutions_generations)
    for generation, solution, fitness in zip(resumed_instance.best_solutions_generations,
                                             resumed_instance.best_solutions,
                                             resumed_instance.best_solutions_fitness):
        print('Generation:', generation, 'Best solution:', solution, 'Fitness:', fitness)
    # The final population of the first run and the starting population of
    # the resumed run are both kept, with the same generation number (2).
    assert resumed_instance.best_solutions_generations == [0, 1, 2, 2, 3, 4]
    assert resumed_instance.best_solution_generation == 4


if __name__ == '__main__':
    main()
