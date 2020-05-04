import numpy
import random
import matplotlib.pyplot
import pickle

class GA:
    def __init__(self, num_generations, 
                 sol_per_pop, 
                 num_parents_mating, 
                 num_genes, 
                 fitness_func,
                 init_range_low=-4,
                 init_range_high=4,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 K_tournament=3,
                 crossover_type="single_point",
                 mutation_type="random",
                 mutation_percent_genes=10,
                 mutation_num_genes=None,
                 random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0):

        """
        # A list of all parameters necessary for building an instance of the genetic algorithm.

        # Parameters of the genetic algorithm:
        num_generations = None # Number of generations.
        sol_per_pop = None # Number of solutions in the population.
        num_parents_mating = None # Number of solutions to be selected as parents in the mating pool.
        pop_size = None # Population size = (number of chromosomes, number of genes per chromosome)
        keep_parents = -1 # If 0, this means the parents of the current populaiton will not be used at all in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. In some cases, the parents are of high quality and thus we do not want to loose such some high quality solutions. If some parent selection operators like roulette wheel selection (RWS), the parents may not be of high quality and thus keeping the parents might degarde the quality of the population.

        fitness_func = None

        # Initial population parameters:
        # It is OK to set the value of any of the 2 parameters to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).
        init_range_low = -4 # The lower value of the random range from which the gene values in the initial population are selected.
        init_range_high = 4 # The upper value of the random range from which the gene values in the initial population are selected.

        # Parameters about parent selection:
        parent_selection_type = None # Type of parent selection.
        select_parents = None # Refers to a method that selects the parents based on the parent selection type specified in parent_selection_type.

        K_tournament = None # For tournament selection, a parent is selected out of K randomly selected solutions.

        population = None # A NumPy array holding the opulation.

        # Crossover parameters:           
        crossover_type = None # Type of the crossover opreator.
        crossover = None # A method that applies the crossover operator based on the selected type of crossover in the crossover_type property.

        # Mutation parameters:           
        mutation_type = None # Type of the mutation opreator.
        mutation = None # A method that applies the mutation operator based on the selected type of mutation in the mutation_type property.

        best_solution_fitness = [] # A list holding the fitness value of the best solution for each generation.

        # Parameters of the function to be optimized:
        num_genes = None # Number of parameters in the function.

        # Mutation parameters:
        mutation_percent_genes=None # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
        mutation_num_genes=None # Number of genes to mutate. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes.
        random_mutation_min_val=None
        random_mutation_max_val=None

        # Some flags:
        run_completed = False # Set to True only when the run() method completes gracefully.
        valid_parameters = False # Set to True when all the paremeters passed in the GA class construtor are valid.
        """

        # Validating the number of solutions in the population (sol_per_pop) and the number of parents to be selected for mating (num_parents_mating)
        if (sol_per_pop <= 0 or num_parents_mating <= 0):
            self.valid_parameters = False
            raise ValueError("ERROR creating an instance of the GA class with invalid parameters. \nThe following parameters must be > 0: \n1) Population size (i.e. number of solutions per population) (sol_per_pop).\n2) Number of selected parents in the mating pool (num_parents_mating).\n")

        # Validating the number of gene.      
        if (num_genes <= 0):
            self.valid_parameters = False
            raise ValueError("ERROR: Number of genes cannot be <= 0. \n")

        self.num_genes = num_genes # Number of genes in the solution.

        if (mutation_num_genes == None):
            if (mutation_percent_genes < 0 or mutation_percent_genes > 100):
                self.valid_parameters = False
                raise ValueError("ERROR: Percentage of selected genes for mutation (mutation_percent_genes) must be >= 0 and <= 100 inclusive.\n")
        elif (mutation_num_genes <= 0 ):
            self.valid_parameters = False
            raise ValueError("ERROR: Number of selected genes for mutation (mutation_num_genes) cannot be <= 0.\n")
        elif (mutation_num_genes > self.num_genes):
            self.valid_parameters = False
            raise ValueError("ERROR: Number of selected genes for mutation (mutation_num_genes) cannot be greater than the number of parameters in the function.\n")
        elif (type(mutation_num_genes) is not int):
            self.valid_parameters = False
            raise ValueError("Error: Number of selected genes for mutation (mutation_num_genes) must be a positive integer >= 1.\n")

        # Validating the number of parents to be selected for mating: num_parents_mating
        if (num_parents_mating > sol_per_pop):
            self.valid_parameters = False
            raise ValueError("ERROR creating an instance of the GA class with invalid parameters. \nThe number of parents to select for mating cannot be greater than the number of solutions in the population (i.e., num_parents_mating must always be <= sol_per_pop).\n")

        # Validating the crossover type: crossover_type
        if (crossover_type == "single_point"):
            self.crossover = self.single_point_crossover
        elif (crossover_type == "two_points"):
            self.crossover = self.two_points_crossover
        elif (crossover_type == "uniform"):
            self.crossover = self.uniform_crossover
        else:
            self.valid_parameters = False
            raise ValueError("ERROR: undefined crossover type. \nThe assigned value to the crossover_type argument does not refer to one of the supported crossover types which are: \n-single_point (for single point crossover)\n-two_points (for two points crossover)\n-uniform (for uniform crossover).\n")

        self.crossover_type = crossover_type

        # Validating the mutation type: mutation_type
        if (mutation_type == "random"):
            self.mutation = self.random_mutation
        elif (mutation_type == "swap"):
            self.mutation = self.swap_mutation
        elif (mutation_type == "scramble"):
            self.mutation = self.scramble_mutation
        elif (mutation_type == "inversion"):
            self.mutation = self.inversion_mutation
        else:
            self.valid_parameters = False
            raise ValueError("ERROR: undefined mutation type. \nThe assigned value to the mutation_type argument does not refer to one of the supported mutation types which are: \n-random (for random mutation)\n-swap (for swap mutation)\n-inversion (for inversion mutation)\n-scramble (for scramble mutation).\n")

        self.mutation_type = mutation_type

        # Validating the selected type of parent selection: parent_selection_type
        if (parent_selection_type == "sss"):
            self.select_parents = self.steady_state_selection
        elif (parent_selection_type == "rws"):
            self.select_parents = self.roulette_wheel_selection
        elif (parent_selection_type == "sus"):
            self.select_parents = self.stochastic_universal_selection
        elif (parent_selection_type == "random"):
            self.select_parents = self.random_selection
        elif (parent_selection_type == "tournament"):
            self.select_parents = self.tournament_selection
        elif (parent_selection_type == "rank"):
            self.select_parents = self.rank_selection
        else:
            self.valid_parameters = False
            raise ValueError("ERROR: undefined parent selection type. \nThe assigned value to the parent_selection_type argument does not refer to one of the supported parent selection techniques which are: \n-sss (for steady state selection)\n-rws (for roulette wheel selection)\n-sus (for stochastic universal selection)\n-rank (for rank selection)\n-random (for random selection)\n-tournament (for tournament selection).\n")

        if(parent_selection_type == "tournament"):
            if (K_tournament > sol_per_pop):
                K_tournament = sol_per_pop
                print("Warining: K of the tournament selection should not be greater than the number of solutions within the population.\nK will be clipped to be equal to the number of solutions in the population (sol_per_pop).\n")
            elif (K_tournament <= 0):
                self.valid_parameters = False
                raise ValueError("ERROR: K of the tournament selection cannot be <=0.\n")

        self.K_tournament = K_tournament

        # Validating the number of parents to keep in the next population: keep_parents
        if (keep_parents > sol_per_pop or keep_parents > num_parents_mating or keep_parents < -1):
            self.valid_parameters = False
            raise ValueError("ERROR: Incorrect value to the keep_parents parameter. \nThe assigned value to the keep_parent parameter must satisfy the following conditions: \n1) Less than or equal to sol_per_pop\n2) Less than or equal to num_parents_mating\n3) Greater than or equal to -1.\n")

        self.keep_parents = keep_parents

        if (self.keep_parents == -1): # Keep all parents in the next population.
            self.num_offspring = sol_per_pop - num_parents_mating
        elif (self.keep_parents == 0): # Keep no parents in the next population.
            self.num_offspring = sol_per_pop
        elif (self.keep_parents > 0): # Keep the specified number of parents in the next population.
            self.num_offspring = sol_per_pop - self.keep_parents

        # Check if the fitness function accepts only a single paramater.
        if (fitness_func.__code__.co_argcount == 1):
            self.fitness_func = fitness_func
        else:
            self.valid_parameters = False
            raise ValueError("The fitness function must accept only a single parameter representing the solution to which the fitness value is calculated.\nThe passed fitness function named '{funcname}' accepts {argcount} argument(s).".format(funcname=fitness_func.__code__.co_name, argcount=fitness_func.__code__.co_argcount))

        self.init_range_low = init_range_low
        self.init_range_high = init_range_high

        # At this point, all necessary parameters validation is done successfully and we are sure that the parameters are valid.
        self.valid_parameters = True

        # Parameters of the genetic algorithm.
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.num_generations = abs(num_generations)
        self.parent_selection_type = parent_selection_type
        
        # Parameters of the mutation operation.
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_num_genes = mutation_num_genes
        self.random_mutation_min_val = random_mutation_min_val
        self.random_mutation_max_val = random_mutation_max_val

        # Even such this parameter is declared in the class header, it is assigned to the object here to access it after saving the object.
        self.best_solution_fitness = []

        # Initializing the population.
        self.initialize_population(self.init_range_low, self.init_range_high)

    def initialize_population(self, low, high):
        """
        Creates an initial population randomly as a NumPy array. The array is saved in the instance attribute named 'population'.
        """
        self.pop_size = (self.sol_per_pop,self.num_genes) # The population will have sol_per_pop chromosome where each chromosome has num_genes genes.
        # Creating the initial population randomly.
        self.population = numpy.random.uniform(low=low, high=high, size=self.pop_size)

    def run(self):
        """
        Running the genetic algorithm. This is the main method in which the genetic algorithm is evolved through a number of generations.
        """
        if self.valid_parameters == False:
            raise ValueError("ERROR calling the run() method: \nThe run() method cannot be executed with invalid parameters. Please check the parameters passed while creating an instance of the GA class.\n")

        for generation in range(self.num_generations):
            # Measuring the fitness of each chromosome in the population.
            fitness = self.cal_pop_fitness()

            # Selecting the best parents in the population for mating.
            parents = self.select_parents(fitness, num_parents=self.num_parents_mating)

            # Generating next generation using crossover.
            offspring_crossover = self.crossover(parents,
                                                 offspring_size=(self.num_offspring, self.num_genes))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)

            if (self.keep_parents == 0):
                self.population = offspring_mutation
            elif (self.keep_parents == -1):
                # Creating the new population based on the parents and offspring.
                self.population[0:parents.shape[0], :] = parents
                self.population[parents.shape[0]:, :] = offspring_mutation
            elif (self.keep_parents > 0):
                parents_to_keep = self.steady_state_selection(fitness, num_parents=self.keep_parents)
                self.population[0:parents_to_keep.shape[0], :] = parents_to_keep
                self.population[parents_to_keep.shape[0]:, :] = offspring_mutation

        # After the run() method completes, the run_completed flag is changed from False to True.
        self.run_completed = True

    def cal_pop_fitness(self):
        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
        """
        if self.valid_parameters == False:
            raise ValueError("ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

        pop_fitness = []
        # Calculating the fitness value of each solution in the current population.
        for sol in self.population:
            fitness = self.fitness_func(sol)
            pop_fitness.append(fitness)
        
        pop_fitness = numpy.array(pop_fitness)

        # The best result in the current iteration.
        self.best_solution_fitness.append(numpy.max(pop_fitness))

        return pop_fitness

    def steady_state_selection(self, fitness, num_parents):
        """
        Applies the steady state selection of the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
        return parents

    def rank_selection(self, fitness, num_parents):
        """
        Applies the rank selection of the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :]
        return parents

    def random_selection(self, fitness, num_parents):
        """
        Randomly selecting the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        parents = numpy.empty((num_parents, self.population.shape[1]))

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :]
        return parents

    def tournament_selection(self, fitness, num_parents):
        """
        Applies the tournament selection of the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :]
        return parents

    def roulette_wheel_selection(self, fitness, num_parents):
        """
        Applies the roulette wheel selection of the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :]
                    break
        return parents

    def stochastic_universal_selection(self, fitness, num_parents):
        """
        Applies the stochastic universal selection of the parents that will mate to produce the offspring. 
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns:
            -parents: The selected parents to mate.
        """
        # https://en.wikipedia.org/wiki/Stochastic_universal_sampling
        # https://books.google.com.eg/books?id=gwUwIEPqk30C&pg=PA60&lpg=PA60&dq=Roulette+Wheel+genetic+algorithm+select+more+than+once&source=bl&ots=GLr2DrPcj4&sig=ACfU3U0jVOGXhzsla8mVqhi5x1giPRL4ew&hl=en&sa=X&ved=2ahUKEwim25rMvdzoAhWa8uAKHbt0AdgQ6AEwA3oECAYQLQ#v=onepage&q=Roulette%20Wheel%20&f=false
        # https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_parent_selection.htm
        # https://www.obitko.com/tutorials/genetic-algorithms/selection.php
        fitness_sum = numpy.sum(fitness)
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=numpy.float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        pointers_distance = 1.0 / self.num_parents_mating # Distance between different pointers.
        first_pointer = numpy.random.uniform(low=0.0, high=pointers_distance, size=1) # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = numpy.empty((num_parents, self.population.shape[1]))
        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num*pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :]
                    break
        return parents

    def single_point_crossover(self, parents, offspring_size):
        """
        Applies the single point crossover. It selects a point randomly at which crossover takes place between two parents.
        It accepts 2 parameters:
            -parents: The parents to mate and create the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns:
            -offspring: The produced offspring after the parents mate.
        """
        offspring = numpy.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def two_points_crossover(self, parents, offspring_size):
        """
        Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between two parents.
        It accepts 2 parameters:
            -parents: The parents to mate and create the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns:
            -offspring: The produced offspring after the parents mate.
        """
        offspring = numpy.empty(offspring_size)
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_point1 = 0
        else:
            crossover_point1 = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1]/2 + 1), size=1)[0]

        crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
            # The genes between the 2 points are copied from the second parent.
            offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
        return offspring

    def uniform_crossover(self, parents, offspring_size):
        """
        Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.
        It accepts 2 parameters:
            -parents: The parents to mate and create the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns:
            -offspring: The produced offspring after the parents mate.
        """
        offspring = numpy.empty(offspring_size)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

            genes_source = numpy.random.randint(low=0, high=2, size=offspring_size[1])
            for gene_idx in range(offspring_size[1]):
                if (genes_source[gene_idx] == 0):
                    # The gene will be copied from the first parent if the current gene index is 0.
                    offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
                elif (genes_source[gene_idx] == 1):
                    # The gene will be copied from the second parent if the current gene index is 1.
                    offspring[k, gene_idx] = parents[parent2_idx, gene_idx]
        return offspring

    def random_mutation(self, offspring):
        """
        Applies the random mutation which changes the values of a number of genes randomly by selecting a random value between random_mutation_min_val and random_mutation_max_val to be added to the selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns:
            -offspring: The offspring after mutation.
        """
        if self.mutation_num_genes == None:
            self.mutation_num_genes = numpy.uint32((self.mutation_percent_genes*offspring.shape[1])/100)
            # Based on the percentage of genes, if the number of selected genes for mutation is less than the least possible value which is 1, then the number will be set to 1.
            if self.mutation_num_genes == 0:
                self.mutation_num_genes = 1
        mutation_indices = numpy.array(random.sample(range(0, offspring.shape[1]), self.mutation_num_genes))
        # Random mutation changes a single gene in each offspring randomly.
        for idx in range(offspring.shape[0]):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(self.random_mutation_min_val, self.random_mutation_max_val, 1)
            offspring[idx, mutation_indices] = offspring[idx, mutation_indices] + random_value
        return offspring

    def swap_mutation(self, offspring):
        """
        Applies the swap mutation which selects interchanges the values of 2 randomly selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns:
            -offspring: The offspring after mutation.
        """
        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=offspring.shape[1]/2, size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            temp = offspring[idx, mutation_gene1]
            offspring[idx, mutation_gene1] = offspring[idx, mutation_gene2]
            offspring[idx, mutation_gene2] = temp
        return offspring

    def inversion_mutation(self, offspring):
        """
        Applies the inversion mutation which selects a subset of genes and invert them.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns:
            -offspring: The offspring after mutation.
        """
        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            genes_to_scramble = numpy.flip(offspring[idx, mutation_gene1:mutation_gene2])
            offspring[idx, mutation_gene1:mutation_gene2] = genes_to_scramble
        return offspring

    def scramble_mutation(self, offspring):
        """
        Applies the scramble mutation which selects a subset of genes and shuffles their order randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns:
            -offspring: The offspring after mutation.
        """
        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)
            genes_range = numpy.arange(start=mutation_gene1, stop=mutation_gene2)
            numpy.random.shuffle(genes_range)
            
            genes_to_scramble = numpy.flip(offspring[idx, genes_range])
            offspring[idx, genes_range] = genes_to_scramble
        return offspring

    def best_solution(self):
        """
        Calculates the fitness values for the current population. 
        If the run() method is not called, then it returns 2 empty lists. Otherwise, it returns the following:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
        """
        if self.run_completed == False:
            raise ValueError("Warning calling the best_solution() method: \nThe run() method is not yet called and thus the GA did not evolve the solutions. Thus, the best solution is retireved from the initial random population without being evolved.\n")

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))

        best_solution = self.population[best_match_idx, :][0][0]
        best_solution_fitness = fitness[best_match_idx][0]

        return best_solution, best_solution_fitness

    def plot_result(self):
        """
        Creating 2 plots that summarizes how the solutions evolved.
        The first plot is between the iteration number and the function output based on the current parameters for the best solution.
        The second plot is between the iteration number and the fitness value of the best solution.
        """
        if self.run_completed == False:
            print("Warning calling the plot_result() method: \nGA is not executed yet and there are no results to display. Please call the run() method before calling the plot_result() method.\n")

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(self.best_solution_fitness)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Fitness")
        matplotlib.pyplot.show()

    def save(self, filename):
        """
        Saving the genetic algorithm instance:
            -filename: Name of the file to save the instance. It must have no extension.
        """
        with open(filename + ".pkl", 'wb') as file:
            pickle.dump(self, file)

def load(filename):
    """
    Reading a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. It must have no extension.
    Returns the genetic algorithm instance.
    """
    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = pickle.load(file)
    except (FileNotFoundError):
        print("Error loading the file. Please check if the file exists.")
    return ga_in
