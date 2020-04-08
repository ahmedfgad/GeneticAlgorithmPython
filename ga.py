import numpy
import random
import sys
import matplotlib.pyplot

class GA:
    # Parameters of the genetic algorithm.
    sol_per_pop = None # Number of solutions in the population.
    num_parents_mating = None # Number of solutions to be selected as parents in the mating pool.
    num_generations = None # Number of generations.
    pop_size = None # Population size = (number of chromosomes, number of genes per chromosome)
    
    population = None # A NumPy array holding the opulation.
    
    # NumPy arrays holding information about the generations.
    best_outputs = [] # A list holding the value of the best solution for each generation.
    best_outputs_fitness = [] # A list holding the fitness value of the best solution for each generation.

    # Parameters of the function to be optimized.
    function_inputs = None # Inputs of the function to be optimized.
    function_output = None # Desired outuput of the function.
    num_weights = None

    # Mutation parameters.
    mutation_percent_genes=None
    mutation_num_genes=None
    mutation_min_val=None
    mutation_max_val=None

    def __init__(self, num_generations, 
                 sol_per_pop, 
                 num_parents_mating, 
                 function_inputs, 
                 function_output,
                 mutation_percent_genes=10,
                 mutation_num_genes=None,
                 mutation_min_val=-1.0,
                 mutation_max_val=1.0):

        # Parameters of the genetic algorithm.
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        
        # Properties of the function to be optimized.
        self.function_inputs = function_inputs # Funtion inputs.
        self.function_output = function_output # Function output.
        self.num_weights = len(self.function_inputs) # Number of parameters in the function.
        
        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes
        self.mutation_min_val = mutation_min_val
        self.mutation_max_val = mutation_max_val
        
        # Initializing the population.
        self.initialize_population()

    def initialize_population(self):
        self.pop_size = (self.sol_per_pop,self.num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
        # Creating the initial population randomly.
        self.population = numpy.random.uniform(low=-4.0, high=4.0, size=self.pop_size)

    def train(self):
        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness()

            # Selecting the best parents in the population for mating.
            parents = self.select_mating_pool(fitness)

            # Generating next generation using crossover.
            offspring_crossover = self.crossover(parents,
                                               offspring_size=(self.pop_size[0]-parents.shape[0], self.num_weights))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)

            if (len(offspring_mutation) == 2):
                print(offspring_mutation[1])
                sys.exit(1)

            # Creating the new population based on the parents and offspring.
            self.population[0:parents.shape[0], :] = parents
            self.population[parents.shape[0]:, :] = offspring_mutation

    def cal_pop_fitness(self):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function calulates the sum of products between each input and its corresponding weight.
        outputs = numpy.sum(self.population*self.function_inputs, axis=1)
        fitness = 1.0 / numpy.abs(outputs - self.function_output)
        
        # The best result in the current iteration.
        self.best_outputs.append(outputs[numpy.where(fitness == numpy.max(fitness))[0][0]])
        self.best_outputs_fitness.append(numpy.max(fitness))

        return fitness

    def select_mating_pool(self, fitness):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((self.num_parents_mating, self.population.shape[1]))
        for parent_num in range(self.num_parents_mating):
            max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = self.population[max_fitness_idx, :]
            fitness[max_fitness_idx] = -99999999999
        return parents

    def crossover(self, parents, offspring_size):
        offspring = numpy.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = numpy.uint8(offspring_size[1]/2)
    
        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring):
        """
        The mutation() method applies mutation over the offspring.
        The parameters are:
            -offspring: The offspring to which mutation is applied.
            -percent_genes: Defaults to 10 and refers to the percentage of genes to which mutation is applied. Based on the percentage, the number of genes are calculated.
            -num_genes: The number of genes to which mutaiton is applied. If None, then the number of genes is calculated based on the argument percent_genes. If not None, then it overrides the value of the argument percent_genes.
            -mutation_min_val & mutation_max_val: The range from which random values are selected and added to the selected genes.
        """
        if self.mutation_num_genes == None:
            if self.mutation_percent_genes <= 0:
                return None, "ERROR: The percentage of genes for which mutation is applied must be > 0. Please specify a valid value for the percent_genes argument." 
            self.mutation_num_genes = numpy.uint32((self.mutation_percent_genes*offspring.shape[1])/100)
            # Based on the percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
            if self.mutation_num_genes == 0:
                self.mutation_num_genes = 1
        if self.mutation_num_genes <= 0:
            return None, "ERROR: Number of genes for mutation must be > 0. Please specify a valid value for the mutation_num_genes argument."
        else:
            self.mutation_num_genes = int(self.mutation_num_genes)
        mutation_indices = numpy.array(random.sample(range(0, offspring.shape[1]), self.mutation_num_genes))
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring.shape[0]):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(self.mutation_min_val, self.mutation_max_val, 1)
            offspring[idx, mutation_indices] = offspring[idx, mutation_indices] + random_value
        return offspring
    
    def best_solution(self):
        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))

        best_solution = self.population[best_match_idx, :][0][0]
        best_solution_fitness = fitness[best_match_idx][0]

        return best_solution, best_solution_fitness

    def plot_result(self):
        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_outputs)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Outputs")
        matplotlib.pyplot.show()

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_outputs_fitness)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Fitness")
        matplotlib.pyplot.show()
