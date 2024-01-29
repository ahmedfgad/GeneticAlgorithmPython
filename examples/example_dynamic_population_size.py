import pygad
import numpy

"""
This is an example to dynamically change the population size (i.e. number of solutions/chromosomes per population) during runtime.

The user has to carefully inspect the parameters and instance attributes to select those that must be changed to be consistent with the new population size.
Check this link for more information: https://pygad.readthedocs.io/en/latest/pygad_more.html#change-population-size-during-runtime
"""

def update_GA(ga_i, 
              pop_size):
    """
    Update the parameters and instance attributes to match the new population size.

    Parameters
    ----------
    ga_i : TYPE
        The pygad.GA instance.
    pop_size : TYPE
        The new population size.

    Returns
    -------
    None.
    """

    ga_i.pop_size = pop_size
    ga_i.sol_per_pop = ga_i.pop_size[0]
    ga_i.num_parents_mating = int(ga_i.pop_size[0]/2)

    # Calculate the new value for the num_offspring parameter.
    if ga_i.keep_elitism != 0:
        ga_i.num_offspring = ga_i.sol_per_pop - ga_i.keep_elitism
    elif ga_i.keep_parents != 0:
        if ga_i.keep_parents == -1:
            ga_i.num_offspring = ga_i.sol_per_pop - ga_i.num_parents_mating
        else:
            ga_i.num_offspring = ga_i.sol_per_pop - ga_i.keep_parents            

    ga_i.num_genes = ga_i.pop_size[1]
    ga_i.population = numpy.random.uniform(low=ga_i.init_range_low,
                                           high=ga_i.init_range_low,
                                           size=ga_i.pop_size)
    fitness = []
    for solution, solution_idx in enumerate(ga_i.population):
        fitness.append(fitness_func(ga_i, solution, solution_idx))
    ga_i.last_generation_fitness = numpy.array(fitness)
    parents, parents_fitness = ga_i.steady_state_selection(ga_i.last_generation_fitness, 
                                                           ga_i.num_parents_mating)
    ga_i.last_generation_elitism = parents[:ga_i.keep_elitism]
    ga_i.last_generation_elitism_indices = parents_fitness[:ga_i.keep_elitism]

    ga_i.last_generation_parents = parents
    ga_i.last_generation_parents_indices = parents_fitness

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.sum(solution)

def on_generation(ga_i):
    # The population starts with 20 solutions.
    print(ga_i.generations_completed, ga_i.population.shape)
    # At generation 15, set the population size to 30 solutions and 10 genes.
    if ga_i.generations_completed >= 15:
        ga_i.pop_size = (30, 10)
        update_GA(ga_i=ga_i, 
                  pop_size=(30, 10))
    # At generation 10, set the population size to 15 solutions and 8 genes.
    elif ga_i.generations_completed >= 10:
        update_GA(ga_i=ga_i, 
                  pop_size=(15, 8))
    # At generation 5, set the population size to 10 solutions and 3 genes.
    elif ga_i.generations_completed >= 5:
        update_GA(ga_i=ga_i, 
                  pop_size=(10, 3))

ga_instance = pygad.GA(num_generations=20,
                       sol_per_pop=20,
                       num_genes=6,
                       num_parents_mating=10,
                       fitness_func=fitness_func,
                       on_generation=on_generation)

ga_instance.run()

