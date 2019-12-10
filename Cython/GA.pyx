import numpy
cimport numpy
import time
import cython
from libc.stdlib cimport rand, RAND_MAX

cdef double DOUBLE_RAND_MAX = RAND_MAX # a double variable holding the maximum random integer in C

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef cal_pop_fitness(numpy.ndarray[numpy.double_t, ndim=1] equation_inputs, numpy.ndarray[numpy.double_t, ndim=2] pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    cdef numpy.ndarray[numpy.double_t, ndim=1] fitness
    fitness = numpy.zeros(pop.shape[0])
    # fitness = numpy.sum(pop*equation_inputs, axis=1) # slower than looping.
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            fitness[i] += pop[i, j]*equation_inputs[j]
    return fitness

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] select_mating_pool(numpy.ndarray[numpy.double_t, ndim=2] pop, numpy.ndarray[numpy.double_t, ndim=1] fitness, int num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    cdef numpy.ndarray[numpy.double_t, ndim=2] parents
    cdef int parent_num, max_fitness_idx, min_val, max_fitness, a

    min_val = -99999999999

    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = 0
        # numpy.where(fitness == numpy.max(fitness)) # slower than argmax() by 150 ms.
        # max_fitness_idx = numpy.argmax(fitness) # slower than looping by 100 ms.
        for a in range(1, fitness.shape[0]):
            if fitness[a] > fitness[max_fitness_idx]:
                max_fitness_idx = a

        # parents[parent_num, :] = pop[max_fitness_idx, :] # slower han looping by 20 ms
        for a in range(parents.shape[1]):
            parents[parent_num, a] = pop[max_fitness_idx, a]
        fitness[max_fitness_idx] = min_val
    return parents

@cython.wraparound(True)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] crossover(numpy.ndarray[numpy.double_t, ndim=2] parents, tuple offspring_size):
    cdef numpy.ndarray[numpy.double_t, ndim=2] offspring
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    cdef int k, parent1_idx, parent2_idx
    cdef numpy.int_t crossover_point
    crossover_point = (int) (offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        
        for m in range(crossover_point):
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, m] = parents[parent1_idx, m]
        for m in range(crossover_point-1, -1):
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, m] = parents[parent2_idx, m]

        # The next 2 lines are slower than using the above loops because they run with C speed.
        # offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef numpy.ndarray[numpy.double_t, ndim=2] mutation(numpy.ndarray[numpy.double_t, ndim=2] offspring_crossover, int num_mutations=1):
    cdef int idx, mutation_num, gene_idx
    cdef double random_value
    cdef Py_ssize_t mutations_counter
    mutations_counter = (int) (offspring_crossover.shape[1] / num_mutations) # using numpy.uint8() is slower than using (int)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    cdef double rand_num
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            # random_value = numpy.random.uniform(-1.0, 1.0, 1) # Slower by 300 ms compared to native C rand() function.
            rand_double = rand()/DOUBLE_RAND_MAX
            random_value = rand_double * (1.0 - (-1.0)) + (-1.0); # The new range is from 1.0 to -1.0.
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.boundscheck(False)
cpdef optimize():
    # Inputs of the equation.
    cdef numpy.ndarray equation_inputs, parents, new_population, fitness, offspring_crossover, offspring_mutation
    cdef int num_weights, sol_per_pop, num_parents_mating, num_generations
    cdef list pop_size
    cdef double t1, t2, t

    equation_inputs = numpy.array([4,-2,3.5,5,-11,-4.7])
    num_weights = equation_inputs.shape[0]

    # Number of the weights we are looking to optimize.
    sol_per_pop = 8
    num_weights = equation_inputs.shape[0]
    num_parents_mating = 4
    
    # Defining the population size.
    pop_size = [sol_per_pop,num_weights] # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
    #Creating the initial population.
    new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

    num_generations = 1000000
    t1 = time.time()
    for generation in range(num_generations):
        # Measuring the fitness of each chromosome in the population.
        fitness = cal_pop_fitness(equation_inputs, new_population)
    
        # Selecting the best parents in the population for mating.
        parents = select_mating_pool(new_population, fitness, 
                                          num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = crossover(parents,
                                           offspring_size=(pop_size[0]-parents.shape[0], num_weights))

        # Adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, num_mutations=2)
    
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    t2 = time.time()
    t = t2-t1
    print("Total Time %.20f" % t)
    print(cal_pop_fitness(equation_inputs, new_population))
