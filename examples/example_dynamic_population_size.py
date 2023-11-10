import pygad
import numpy

"""
This is an example to dynamically change the population size (i.e. number of solutions/chromosomes per population) during runtime.
The following 2 instance attributes must be changed to meet the new desired population size:
    1) population: This is a NumPy array holding the population.
    2) num_offspring: This represents the number of offspring to produce during crossover.
For example, if the population initially has 20 solutions and 6 genes. To change it to have 30 solutions, then:
    1)population: Create a new NumPy array with the desired size (30, 6) and assign it to the population instance attribute.
    2)num_offspring: Set the num_offspring attribute accordingly (e.g. 29 assuming that keep_elitism has the default value of 1).
"""

def fitness_func(ga_instance, solution, solution_idx):
    return [numpy.random.rand(), numpy.random.rand()]

def on_generation(ga_i):
    # The population starts with 20 solutions.
    print(ga_i.generations_completed, ga_i.num_offspring, ga_i.population.shape)
    # At generation 15, increase the population size to 40 solutions.
    if ga_i.generations_completed >= 15:
        ga_i.num_offspring = 49
        new_population = numpy.zeros(shape=(ga_i.num_offspring+1, ga_i.population.shape[1]), dtype=ga_i.population.dtype)
        new_population[:ga_i.population.shape[0], :] = ga_i.population
        ga_i.population = new_population
    elif ga_i.generations_completed >= 10:
        ga_i.num_offspring = 39
        new_population = numpy.zeros(shape=(ga_i.num_offspring+1, ga_i.population.shape[1]), dtype=ga_i.population.dtype)
        new_population[:ga_i.population.shape[0], :] = ga_i.population
        ga_i.population = new_population
    # At generation 10, increase the population size to 30 solutions.
    elif ga_i.generations_completed >= 5:
        ga_i.num_offspring = 29
        new_population = numpy.zeros(shape=(ga_i.num_offspring+1, ga_i.population.shape[1]), dtype=ga_i.population.dtype)
        new_population[:ga_i.population.shape[0], :] = ga_i.population
        ga_i.population = new_population

ga_instance = pygad.GA(num_generations=20,
                       sol_per_pop=20,
                       num_genes=6,
                       num_parents_mating=10,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       parent_selection_type='nsga2')

ga_instance.run()
