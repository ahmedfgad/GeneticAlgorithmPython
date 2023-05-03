"""
The pygad.utils.mutation module has all the built-in mutation operators.
"""

import numpy
import random

import pygad

class Mutation:

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
                        rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                        high=self.random_mutation_max_val,
                                                        size=1)[0]
                        if self.mutation_by_replacement:
                            value_from_space = rand_val
                        else:
                            value_from_space = offspring[offspring_idx, gene_idx] + rand_val
                    elif type(curr_gene_space) is dict:
                        # The gene's space of type dict specifies the lower and upper limits of a gene.
                        if 'step' in curr_gene_space.keys():
                            value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                stop=curr_gene_space['high'],
                                                                                step=curr_gene_space['step']),
                                                                   size=1)
                        else:
                            value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                    high=curr_gene_space['high'],
                                                                    size=1)[0]
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
                                                                   size=1)
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
                    value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                            high=self.random_mutation_max_val, 
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
                            rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                            high=self.random_mutation_max_val,
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
                                                                       size=1)
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
                                                                       size=1)
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
                # Generating a random value.
                random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
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
                                                                                         min_val=self.random_mutation_min_val,
                                                                                         max_val=self.random_mutation_max_val,
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
                if probs[gene_idx] <= self.mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                        high=self.random_mutation_max_val, 
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
                                                                                             min_val=self.random_mutation_min_val,
                                                                                             max_val=self.random_mutation_max_val,
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

        for idx in range(len(parents_to_keep), fitness.shape[0]):
            fitness[idx] = self.fitness_func(self, temp_population[idx], None)
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
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_num_genes = self.mutation_num_genes[0]
            else:
                adaptive_mutation_num_genes = self.mutation_num_genes[1]
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:

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
                        rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                        high=self.random_mutation_max_val,
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
                                                                       size=1)
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
                                                                   size=1)
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
                    value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                            high=self.random_mutation_max_val, 
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
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_num_genes = self.mutation_num_genes[0]
            else:
                adaptive_mutation_num_genes = self.mutation_num_genes[1]
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:
                # Generating a random value.
                random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                    high=self.random_mutation_max_val, 
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
                                                                                         min_val=self.random_mutation_min_val,
                                                                                         max_val=self.random_mutation_max_val,
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
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_probability = self.mutation_probability[0]
            else:
                adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
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
                            rand_val = numpy.random.uniform(low=self.random_mutation_min_val,
                                                            high=self.random_mutation_max_val,
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
                                                                       size=1)
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
                                                                       size=1)
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
                        value_from_space = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                                high=self.random_mutation_max_val, 
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
            if offspring_fitness[offspring_idx] < average_fitness:
                adaptive_mutation_probability = self.mutation_probability[0]
            else:
                adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= adaptive_mutation_probability:
                    # Generating a random value.
                    random_value = numpy.random.uniform(low=self.random_mutation_min_val, 
                                                        high=self.random_mutation_max_val, 
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
                                                                                             min_val=self.random_mutation_min_val,
                                                                                             max_val=self.random_mutation_max_val,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             num_trials=10)
        return offspring
