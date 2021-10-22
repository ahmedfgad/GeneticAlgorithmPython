from numpy.random import uniform
from numpy import inf, empty, sum, where, max, uint8

def cal_pop_fitness(variables, population):
    """ Calculating the fitness value of each solution in the current population.
        The fitness function calulates the sum of products between each input
        and its corresponding weight. """
    return sum(population * variables, axis=1)

def select_mating_pool(pop, fitness, num_parents):
    """ Selecting the best individuals in the current generation as parents for
        producing the offspring of the next generation. """
    parents = empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_index = where(fitness == max(fitness))[0][0]
        parents[parent_num, :] = pop[max_fitness_index, :]
        fitness[max_fitness_index] = -inf
    return parents

def crossover(parents, offspring_size):
    """ The point at which crossover takes place between two parents.  Usually,
        it is at the center.  Index of the first parent to mate.  Index of the
        second parent to mate.  The new offspring will have its first half of
        its genes taken from the first parent.  The new offspring will have its
        second half of its genes taken from the second parent. """
    offspring = empty(offspring_size)
    co_point = uint8(offspring_size[1] / 2)
    
    for i in range(offspring_size[0]):
        offspring[i, :co_point] = parents[i % parents.shape[0], :co_point]
        offspring[i, co_point:] = parents[(i + 1) % parents.shape[0], co_point:]
    return offspring

def mutation(offspring_crossover, mutations=1):
    """ Mutation changes a number of genes as defined by the mutations argument.
        The changes are random.  The random value to be added to the gene."""
    mutations_counter = uint8(offspring_crossover.shape[1] / mutations)
    for i in range(offspring_crossover.shape[0] * mutations):
        gene_i = mutations_counter - 1 if i % mutations else gene_i
        offspring_crossover[i // mutations, gene_i] += uniform(-1.0, 1.0, 1)
        gene_i += mutations_counter
    return offspring_crossover
