"""
The pygad.utils.parent_selection module has all the built-in parent selection operators.
"""

import numpy

class ParentSelection:
    def steady_state_selection(self, fitness, num_parents):
    
        """
        Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
        fitness_sorted.reverse()

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[fitness_sorted[parent_num], :].copy()

        return parents, numpy.array(fitness_sorted[:num_parents])

    def rank_selection(self, fitness, num_parents):

        """
        Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
        Rank selection gives a rank from 1 to N (number of solutions) to each solution based on its fitness.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        # This has the index of each solution in the population.
        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])

        # Rank the solutions based on their fitness. The worst is gives the rank 1. The best has the rank N.
        rank = numpy.arange(1, self.sol_per_pop+1)

        probs = rank / numpy.sum(rank)

        probs_start, probs_end, parents = self.wheel_cumulative_probs(probs=probs.copy(), 
                                                                      num_parents=num_parents)

        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    # The variable idx has the rank of solution but not its index in the population.
                    # Return the correct index of the solution.
                    mapped_idx = fitness_sorted[idx]
                    parents[parent_num, :] = self.population[mapped_idx, :].copy()
                    parents_indices.append(mapped_idx)
                    break

        return parents, numpy.array(parents_indices)

    def random_selection(self, fitness, num_parents):
    
        """
        Selects the parents randomly. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        rand_indices = numpy.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

        for parent_num in range(num_parents):
            parents[parent_num, :] = self.population[rand_indices[parent_num], :].copy()

        return parents, rand_indices

    def tournament_selection(self, fitness, num_parents):

        """
        Selects the parents using the tournament selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
    
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)
    
        parents_indices = []
    
        for parent_num in range(num_parents):
            rand_indices = numpy.random.randint(low=0.0, high=len(fitness), size=self.K_tournament)
            K_fitnesses = fitness[rand_indices]
            selected_parent_idx = numpy.where(K_fitnesses == numpy.max(K_fitnesses))[0][0]
            parents_indices.append(rand_indices[selected_parent_idx])
            parents[parent_num, :] = self.population[rand_indices[selected_parent_idx], :].copy()
    
        return parents, numpy.array(parents_indices)
    
    def roulette_wheel_selection(self, fitness, num_parents):
    
        """
        Selects the parents using the roulette wheel selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
    
        fitness_sum = numpy.sum(fitness)
        if fitness_sum == 0:
            self.logger.error("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")

        probs = fitness / fitness_sum

        probs_start, probs_end, parents = self.wheel_cumulative_probs(probs=probs.copy(), 
                                                                      num_parents=num_parents)

        parents_indices = []

        for parent_num in range(num_parents):
            rand_prob = numpy.random.rand()
            for idx in range(probs.shape[0]):
                if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    parents_indices.append(idx)
                    break

        return parents, numpy.array(parents_indices)

    def wheel_cumulative_probs(self, probs, num_parents):
        """
        A helper function to calculate the wheel probabilities for these 2 methods:
            1) roulette_wheel_selection
            2) rank_selection
        It accepts a single 1D array representing the probabilities of selecting each solution.
        It returns 2 1D arrays:
            1) probs_start has the start of each range.
            2) probs_start has the end of each range.
        It also returns an empty array for the parents.
        """

        probs_start = numpy.zeros(probs.shape, dtype=float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        return probs_start, probs_end, parents

    def stochastic_universal_selection(self, fitness, num_parents):

        """
        Selects the parents using the stochastic universal selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """

        fitness_sum = numpy.sum(fitness)
        if fitness_sum == 0:
            self.logger.error("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
            raise ZeroDivisionError("Cannot proceed because the sum of fitness values is zero. Cannot divide by zero.")
        probs = fitness / fitness_sum
        probs_start = numpy.zeros(probs.shape, dtype=float) # An array holding the start values of the ranges of probabilities.
        probs_end = numpy.zeros(probs.shape, dtype=float) # An array holding the end values of the ranges of probabilities.

        curr = 0.0

        # Calculating the probabilities of the solutions to form a roulette wheel.
        for _ in range(probs.shape[0]):
            min_probs_idx = numpy.where(probs == numpy.min(probs))[0][0]
            probs_start[min_probs_idx] = curr
            curr = curr + probs[min_probs_idx]
            probs_end[min_probs_idx] = curr
            probs[min_probs_idx] = 99999999999

        pointers_distance = 1.0 / self.num_parents_mating # Distance between different pointers.
        first_pointer = numpy.random.uniform(low=0.0, 
                                             high=pointers_distance, 
                                             size=1)[0] # Location of the first pointer.

        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        if self.gene_type_single == True:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]), dtype=object)

        parents_indices = []

        for parent_num in range(num_parents):
            rand_pointer = first_pointer + parent_num*pointers_distance
            for idx in range(probs.shape[0]):
                if (rand_pointer >= probs_start[idx] and rand_pointer < probs_end[idx]):
                    parents[parent_num, :] = self.population[idx, :].copy()
                    parents_indices.append(idx)
                    break

        return parents, numpy.array(parents_indices)
