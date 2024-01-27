"""
The pygad.utils.mutation module has all the built-in mutation operators.
"""

import numpy
import random

import pygad
import concurrent.futures

class Mutation:

    def __init__():
        pass

    def random_mutation(self, offspring):

        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.mutation_by_space(offspring)
            else:
                offspring = self.mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.mutation_probs_by_space(offspring)
            else:
                offspring = self.mutation_probs_randomly(offspring)

        return offspring

    def mutation_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if self.gene_space_nested:
                    # Returning the current gene space from the 'gene_space' attribute.
                    if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                        curr_gene_space = self.gene_space[gene_idx].copy()
                    else:
                        curr_gene_space = self.gene_space[gene_idx]

                    # If the gene space has only a single value, use it as the new gene value.
                    if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                        value_from_space = curr_gene_space
                    # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                    elif curr_gene_space is None:
                        rand_val = numpy.random.uniform(low=range_min,
                                                        high=range_max,
                                                        size=1)[0]
                        if self.mutation_by_replacement:
                            value_from_space = rand_val
                        else:
                            value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                    elif type(curr_gene_space) is dict:
                        # The gene's space of type dict specifies the lower and upper limits of a gene.
                        if 'step' in curr_gene_space.keys():
                            # The numpy.random.choice() and numpy.random.uniform() functions return a NumPy array as the output even if the array has a single value.
                            # We have to return the output at index 0 to force a numeric value to be returned not an object of type numpy.ndarray. 
                            # If numpy.ndarray is returned, then it will cause an issue later while using the set() function.
                            # Randomly select a value from a discrete range.
                            value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                stop=curr_gene_space['high'],
                                                                                step=curr_gene_space['step']),
                                                                   size=1)[0]
                        else:
                            # Return the current gene value.
                            value_from_space = offspring[offspring_idx, gene_idx]
                            # Generate a random value to be added to the current gene value.
                            rand_val = numpy.random.uniform(low=range_min,
                                                            high=range_max,
                                                            size=1)[0]
                            # The objective is to have a new gene value that respects the gene_space boundaries.
                            # The next if-else block checks if adding the random value keeps the new gene value within the gene_space boundaries.
                            temp_val = value_from_space + rand_val
                            if temp_val < curr_gene_space['low']:
                                # Restrict the new value to be > curr_gene_space['low']
                                # If subtracting the random value makes the new gene value outside the boundaries [low, high), then use the lower boundary the gene value.
                                if curr_gene_space['low'] <= value_from_space - rand_val < curr_gene_space['high']:
                                    # Because subtracting the random value keeps the new gene value within the boundaries [low, high), then use such a value as the gene value.
                                    temp_val = value_from_space - rand_val
                                else:
                                    # Because subtracting the random value makes the new gene value outside the boundaries [low, high), then use the lower boundary as the gene value.
                                    temp_val = curr_gene_space['low']
                            elif temp_val >= curr_gene_space['high']:
                                # Restrict the new value to be < curr_gene_space['high']
                                # If subtracting the random value makes the new gene value outside the boundaries [low, high), then use such a value as the gene value.
                                if curr_gene_space['low'] <= value_from_space - rand_val < curr_gene_space['high']:
                                    # Because subtracting the random value keeps the new value within the boundaries [low, high), then use such a value as the gene value.
                                    temp_val = value_from_space - rand_val
                                else:
                                    # Because subtracting the random value makes the new gene value outside the boundaries [low, high), then use the lower boundary as the gene value.
                                    temp_val = curr_gene_space['low']
                            value_from_space = temp_val
                    else:
                        # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                        # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                        if len(curr_gene_space) == 1:
                            value_from_space = curr_gene_space[0]
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        else:
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))

                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)[0]
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)[0]
                    else:
                        # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))

                        if len(values_to_select_from) == 0:
                            value_from_space = offspring[offspring_idx, gene_idx]
                        else:
                            value_from_space = random.choice(values_to_select_from)
                    # value_from_space = random.choice(self.gene_space)

                if value_from_space is None:
                    # TODO: Return index 0.
                    # TODO: Check if this if statement is necessary.
                    value_from_space = numpy.random.uniform(low=range_min, 
                                                            high=range_max, 
                                                            size=1)[0]

                # Assinging the selected value from the space to the gene.
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                         self.gene_type[1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                         self.gene_type[gene_idx][1])

                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring

    def mutation_probs_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space and the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated based on the mutation space.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if probs[gene_idx] <= self.mutation_probability:
                    if self.gene_space_nested:
                        # Returning the current gene space from the 'gene_space' attribute.
                        if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                            curr_gene_space = self.gene_space[gene_idx].copy()
                        else:
                            curr_gene_space = self.gene_space[gene_idx]
        
                        # If the gene space has only a single value, use it as the new gene value.
                        if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                            value_from_space = curr_gene_space
                        # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                        elif curr_gene_space is None:
                            rand_val = numpy.random.uniform(low=range_min,
                                                            high=range_max,
                                                            size=1)[0]
                            if self.mutation_by_replacement:
                                value_from_space = rand_val
                            else:
                                value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                        elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)[0]
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)[0]
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                            if len(curr_gene_space) == 1:
                                value_from_space = curr_gene_space[0]
                            # If the gene space has more than 1 value, then select a new one that is different from the current value.
                            else:
                                values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))

                                if len(values_to_select_from) == 0:
                                    value_from_space = offspring[offspring_idx, gene_idx]
                                else:
                                    value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            if 'step' in self.gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                    stop=self.gene_space['high'],
                                                                                    step=self.gene_space['step']),
                                                                       size=1)[0]
                            else:
                                value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                        high=self.gene_space['high'],
                                                                        size=1)[0]
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))

                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)

                    # Assigning the selected value from the space to the gene.
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                             self.gene_type[1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                             self.gene_type[gene_idx][1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def mutation_randomly(self, offspring):

        """
        Applies the random mutation the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                # Generating a random value.
                random_value = numpy.random.uniform(low=range_min, 
                                                    high=range_max, 
                                                    size=1)[0]
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]
               # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]

                # Round the gene
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        random_value = numpy.round(random_value, self.gene_type[1])
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)

        return offspring

    def mutation_probs_randomly(self, offspring):

        """
        Applies the random mutation using the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more gene in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if probs[gene_idx] <= self.mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=range_min, 
                                                        high=range_max, 
                                                        size=1)[0]
                    # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                    if self.mutation_by_replacement:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]

                    # Round the gene
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            random_value = numpy.round(random_value, self.gene_type[1])
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def swap_mutation(self, offspring):

        """
        Applies the swap mutation which interchanges the values of 2 randomly selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
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
        Applies the inversion mutation which selects a subset of genes and inverts them (in order).
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
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
        It returns an array of the mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)
            genes_range = numpy.arange(start=mutation_gene1, stop=mutation_gene2)
            numpy.random.shuffle(genes_range)
            
            genes_to_scramble = numpy.flip(offspring[idx, genes_range])
            offspring[idx, genes_range] = genes_to_scramble
        return offspring

    def adaptive_mutation_population_fitness(self, offspring):

        """
        A helper method to calculate the average fitness of the solutions before applying the adaptive mutation.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns the average fitness to be used in adaptive mutation.
        """        

        fitness = self.last_generation_fitness.copy()
        temp_population = numpy.zeros_like(self.population)

        if (self.keep_elitism == 0):
            if (self.keep_parents == 0):
                parents_to_keep = []
            elif (self.keep_parents == -1):
                parents_to_keep = self.last_generation_parents.copy()
                temp_population[0:len(parents_to_keep), :] = parents_to_keep
            elif (self.keep_parents > 0):
                parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_parents)
                temp_population[0:len(parents_to_keep), :] = parents_to_keep
        else:
            parents_to_keep, _ = self.steady_state_selection(self.last_generation_fitness, num_parents=self.keep_elitism)
            temp_population[0:len(parents_to_keep), :] = parents_to_keep

        temp_population[len(parents_to_keep):, :] = offspring

        fitness[:self.last_generation_parents.shape[0]] = self.last_generation_fitness[self.last_generation_parents_indices]

        first_idx = len(parents_to_keep)
        last_idx = fitness.shape[0]
        if len(fitness.shape) > 1:
            # TODO This is a multi-objective optimization problem.
            # fitness[first_idx:last_idx] = [0]*(last_idx - first_idx)
            fitness[first_idx:last_idx] = numpy.zeros(shape=(last_idx - first_idx, fitness.shape[1]))
            # raise ValueError('Edit adaptive mutation to work with multi-objective optimization problems.')
        else:
            # This is a single-objective optimization problem.
            fitness[first_idx:last_idx] = [0]*(last_idx - first_idx)

        # # No parallel processing.
        if self.parallel_processing is None:
            if self.fitness_batch_size in [1, None]:
                # Calculate the fitness for each individual solution.
                for idx in range(first_idx, last_idx):
                    # We cannot return the index of the solution within the population.
                    # Because the new solution (offspring) does not yet exist in the population.
                    # The user should handle this situation if the solution index is used anywhere.
                    fitness[idx] = self.fitness_func(self, 
                                                      temp_population[idx], 
                                                      None)
            else:
                # Calculate the fitness for batch of solutions.
    
                # Number of batches.
                num_batches = int(numpy.ceil((last_idx - first_idx) / self.fitness_batch_size))
    
                for batch_idx in range(num_batches):
                    # The index of the first solution in the current batch.
                    batch_first_index = first_idx + batch_idx * self.fitness_batch_size
                    # The index of the last solution in the current batch.
                    if batch_idx == (num_batches - 1):
                        batch_last_index = last_idx
                    else:
                        batch_last_index = first_idx + (batch_idx + 1) * self.fitness_batch_size
    
                    # Calculate the fitness values for the batch.
                    # We cannot return the index/indices of the solution(s) within the population.
                    # Because the new solution(s) (offspring) do(es) not yet exist in the population.
                    # The user should handle this situation if the solution index is used anywhere.
                    fitness_temp = self.fitness_func(self, 
                                                     temp_population[batch_first_index:batch_last_index], 
                                                     None) 
                    # Insert the fitness of each solution at the proper index.
                    for idx in range(batch_first_index, batch_last_index):
                        fitness[idx] = fitness_temp[idx - batch_first_index]

        else:
            # Parallel processing
            # Decide which class to use based on whether the user selected "process" or "thread"
            # TODO Add ExecutorClass as an instance attribute in the pygad.GA instances. Then retrieve this instance here instead of creating a new one.
            if self.parallel_processing[0] == "process":
                ExecutorClass = concurrent.futures.ProcessPoolExecutor
            else:
                ExecutorClass = concurrent.futures.ThreadPoolExecutor
    
            # We can use a with statement to ensure threads are cleaned up promptly (https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor-example)
            with ExecutorClass(max_workers=self.parallel_processing[1]) as executor:
                # Indices of the solutions to calculate its fitness.
                solutions_to_submit_indices = list(range(first_idx, last_idx))
                # The solutions to calculate its fitness.
                solutions_to_submit = [temp_population[sol_idx].copy() for sol_idx in solutions_to_submit_indices]
                if self.fitness_batch_size in [1, None]:
                    # Use parallel processing to calculate the fitness of the solutions.
                    for index, sol_fitness in zip(solutions_to_submit_indices, executor.map(self.fitness_func, [self]*len(solutions_to_submit_indices), solutions_to_submit, solutions_to_submit_indices)):
                        if type(sol_fitness) in self.supported_int_float_types:
                            # The fitness function returns a single numeric value.
                            # This is a single-objective optimization problem.
                            fitness[index] = sol_fitness
                        elif type(sol_fitness) in [list, tuple, numpy.ndarray]:
                            # The fitness function returns a list/tuple/numpy.ndarray.
                            # This is a multi-objective optimization problem.
                            fitness[index] = sol_fitness
                        else:
                            raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value {sol_fitness} of type {type(sol_fitness)} found.")
                else:
                    # Reaching this point means that batch processing is in effect to calculate the fitness values.
                    # Number of batches.
                    num_batches = int(numpy.ceil(len(solutions_to_submit_indices) / self.fitness_batch_size))
                    # Each element of the `batches_solutions` list represents the solutions in one batch.
                    batches_solutions = []
                    # Each element of the `batches_indices` list represents the solutions' indices in one batch.
                    batches_indices = []
                    # For each batch, get its indices and call the fitness function.
                    for batch_idx in range(num_batches):
                        batch_first_index = batch_idx * self.fitness_batch_size
                        batch_last_index = (batch_idx + 1) * self.fitness_batch_size
                        batch_indices = solutions_to_submit_indices[batch_first_index:batch_last_index]
                        batch_solutions = self.population[batch_indices, :]
    
                        batches_solutions.append(batch_solutions)
                        batches_indices.append(batch_indices)

                    for batch_indices, batch_fitness in zip(batches_indices, executor.map(self.fitness_func, [self]*len(solutions_to_submit_indices), batches_solutions, batches_indices)):
                        if type(batch_fitness) not in [list, tuple, numpy.ndarray]:
                            raise TypeError(f"Expected to receive a list, tuple, or numpy.ndarray from the fitness function but the value ({batch_fitness}) of type {type(batch_fitness)}.")
                        elif len(numpy.array(batch_fitness)) != len(batch_indices):
                            raise ValueError(f"There is a mismatch between the number of solutions passed to the fitness function ({len(batch_indices)}) and the number of fitness values returned ({len(batch_fitness)}). They must match.")

                        for index, sol_fitness in zip(batch_indices, batch_fitness):
                            if type(sol_fitness) in self.supported_int_float_types:
                                # The fitness function returns a single numeric value.
                                # This is a single-objective optimization problem.
                                fitness[index] = sol_fitness
                            elif type(sol_fitness) in [list, tuple, numpy.ndarray]:
                                # The fitness function returns a list/tuple/numpy.ndarray.
                                # This is a multi-objective optimization problem.
                                fitness[index] = sol_fitness
                            else:
                                raise ValueError(f"The fitness function should return a number or an iterable (list, tuple, or numpy.ndarray) but the value ({sol_fitness}) of type {type(sol_fitness)} found.")



        if len(fitness.shape) > 1:
            # TODO This is a multi-objective optimization problem.
            # Calculate the average of each objective's fitness across all solutions in the population.
            average_fitness = numpy.mean(fitness, axis=0)
        else:
            # This is a single-objective optimization problem.
            average_fitness = numpy.mean(fitness)

        return average_fitness, fitness[len(parents_to_keep):]

    def adaptive_mutation(self, offspring):

        """
        Applies the adaptive mutation which changes the values of a number of genes randomly. In adaptive mutation, the number of genes to mutate differs based on the fitness value of the solution.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # If the attribute 'gene_space' exists (i.e. not None), then the mutation values are selected from the 'gene_space' parameter according to the space of values of each gene. Otherwise, it is selected randomly based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_probs_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_probs_randomly(offspring)

        return offspring

    def adaptive_mutation_by_space(self, offspring):

        """
        Applies the adaptive mutation based on the 2 parameters 'mutation_num_genes' and 'gene_space'. 
        A number of genes equal are selected randomly for mutation. This number depends on the fitness of the solution.
        The random values are selected from the 'gene_space' parameter.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        
        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]

            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if self.gene_space_nested:
                    # Returning the current gene space from the 'gene_space' attribute.
                    if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                        curr_gene_space = self.gene_space[gene_idx].copy()
                    else:
                        curr_gene_space = self.gene_space[gene_idx]

                    # If the gene space has only a single value, use it as the new gene value.
                    if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                        value_from_space = curr_gene_space
                    # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                    elif curr_gene_space is None:
                        rand_val = numpy.random.uniform(low=range_min,
                                                        high=range_max,
                                                        size=1)[0]
                        if self.mutation_by_replacement:
                            value_from_space = rand_val
                        else:
                            value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                    elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                # The numpy.random.choice() and numpy.random.uniform() functions return a NumPy array as the output even if the array has a single value.
                                # We have to return the output at index 0 to force a numeric value to be returned not an object of type numpy.ndarray. 
                                # If numpy.ndarray is returned, then it will cause an issue later while using the set() function.
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)[0]
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)[0]
                    else:
                        # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                        # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                        if len(curr_gene_space) == 1:
                            value_from_space = curr_gene_space[0]
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        else:
                            values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))

                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)
                else:
                    # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                    if type(self.gene_space) is dict:
                        if 'step' in self.gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                stop=self.gene_space['high'],
                                                                                step=self.gene_space['step']),
                                                                   size=1)[0]
                        else:
                            value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                    high=self.gene_space['high'],
                                                                    size=1)[0]
                    else:
                        values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))

                        if len(values_to_select_from) == 0:
                            value_from_space = offspring[offspring_idx, gene_idx]
                        else:
                            value_from_space = random.choice(values_to_select_from)


                if value_from_space is None:
                    value_from_space = numpy.random.uniform(low=range_min, 
                                                            high=range_max, 
                                                            size=1)[0]

                # Assinging the selected value from the space to the gene.
                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                         self.gene_type[1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                         self.gene_type[gene_idx][1])
                    else:
                        offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring

    def adaptive_mutation_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_num_genes' parameter. 
        A number of genes equal are selected randomly for mutation. This number depends on the fitness of the solution.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if fitness_comparison:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]

            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                # Generating a random value.
                random_value = numpy.random.uniform(low=range_min, 
                                                    high=range_max, 
                                                    size=1)[0]
                # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                if self.mutation_by_replacement:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]
                # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                else:
                    if self.gene_type_single == True:
                        random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                    else:
                        random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                        if type(random_value) is numpy.ndarray:
                            random_value = random_value[0]

                if self.gene_type_single == True:
                    if not self.gene_type[1] is None:
                        random_value = numpy.round(random_value, self.gene_type[1])
                else:
                    if not self.gene_type[gene_idx][1] is None:
                        random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         num_trials=10)
        return offspring

    def adaptive_mutation_probs_by_space(self, offspring):

        """
        Applies the adaptive mutation based on the 2 parameters 'mutation_probability' and 'gene_space'.
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on space of values for each gene.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if probs[gene_idx] <= adaptive_mutation_probability:
                    if self.gene_space_nested:
                        # Returning the current gene space from the 'gene_space' attribute.
                        if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                            curr_gene_space = self.gene_space[gene_idx].copy()
                        else:
                            curr_gene_space = self.gene_space[gene_idx]
        
                        # If the gene space has only a single value, use it as the new gene value.
                        if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                            value_from_space = curr_gene_space
                        # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                        elif curr_gene_space is None:
                            rand_val = numpy.random.uniform(low=range_min,
                                                            high=range_max,
                                                            size=1)[0]
                            if self.mutation_by_replacement:
                                value_from_space = rand_val
                            else:
                                value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                        elif type(curr_gene_space) is dict:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)[0]
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)[0]
                        else:
                            # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                            # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                            if len(curr_gene_space) == 1:
                                value_from_space = curr_gene_space[0]
                            # If the gene space has more than 1 value, then select a new one that is different from the current value.
                            else:
                                values_to_select_from = list(set(curr_gene_space) - set([offspring[offspring_idx, gene_idx]]))

                                if len(values_to_select_from) == 0:
                                    value_from_space = offspring[offspring_idx, gene_idx]
                                else:
                                    value_from_space = random.choice(values_to_select_from)
                    else:
                        # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                        if type(self.gene_space) is dict:
                            if 'step' in self.gene_space.keys():
                                # The numpy.random.choice() and numpy.random.uniform() functions return a NumPy array as the output even if the array has a single value.
                                # We have to return the output at index 0 to force a numeric value to be returned not an object of type numpy.ndarray. 
                                # If numpy.ndarray is returned, then it will cause an issue later while using the set() function.
                                value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                                    stop=self.gene_space['high'],
                                                                                    step=self.gene_space['step']),
                                                                       size=1)[0]
                            else:
                                value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                                        high=self.gene_space['high'],
                                                                        size=1)[0]
                        else:
                            values_to_select_from = list(set(self.gene_space) - set([offspring[offspring_idx, gene_idx]]))

                            if len(values_to_select_from) == 0:
                                value_from_space = offspring[offspring_idx, gene_idx]
                            else:
                                value_from_space = random.choice(values_to_select_from)

                    if value_from_space is None:
                        value_from_space = numpy.random.uniform(low=range_min, 
                                                                high=range_max, 
                                                                size=1)[0]

                    # Assinging the selected value from the space to the gene.
                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[0](value_from_space),
                                                                             self.gene_type[1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[0](value_from_space)
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            offspring[offspring_idx, gene_idx] = numpy.round(self.gene_type[gene_idx][0](value_from_space),
                                                                             self.gene_type[gene_idx][1])
                        else:
                            offspring[offspring_idx, gene_idx] = self.gene_type[gene_idx][0](value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring

    def adaptive_mutation_probs_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_probability' parameter. 
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                if type(self.random_mutation_min_val) in self.supported_int_float_types:
                    range_min = self.random_mutation_min_val
                    range_max = self.random_mutation_max_val
                else:
                    range_min = self.random_mutation_min_val[gene_idx]
                    range_max = self.random_mutation_max_val[gene_idx]

                if probs[gene_idx] <= adaptive_mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=range_min, 
                                                        high=range_max, 
                                                        size=1)[0]
                    # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
                    if self.mutation_by_replacement:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]
                    # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
                    else:
                        if self.gene_type_single == True:
                            random_value = self.gene_type[0](offspring[offspring_idx, gene_idx] + random_value)
                        else:
                            random_value = self.gene_type[gene_idx][0](offspring[offspring_idx, gene_idx] + random_value)
                            if type(random_value) is numpy.ndarray:
                                random_value = random_value[0]

                    if self.gene_type_single == True:
                        if not self.gene_type[1] is None:
                            random_value = numpy.round(random_value, self.gene_type[1])
                    else:
                        if not self.gene_type[gene_idx][1] is None:
                            random_value = numpy.round(random_value, self.gene_type[gene_idx][1])

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring
