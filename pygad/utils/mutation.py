"""
The pygad.utils.mutation module has all the built-in mutation operators.
"""

import numpy
import random

import pygad
import concurrent.futures

import warnings

class Mutation:

    def __init__(self):
        pass

    def random_mutation(self, offspring):

        """
        Applies the random mutation which changes the values of a number of genes randomly.
        The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.

        It accepts:
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
        Applies the mutation using the gene_space parameter.
        It accepts:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            for gene_idx in mutation_indices:

                value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                # Before assigning the selected value from the space to the gene, change its data type and round it.
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
                                                                                         sample_size=self.sample_size,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         build_initial_pop=False)
        return offspring

    def mutation_probs_by_space(self, offspring):

        """
        Applies the random mutation using the mutation values' space and the mutation probability. For each gene, if its probability is <= that the mutation probability, then it will be mutated based on the mutation space.
        It accepts:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring using the mutation space.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                if probs[gene_idx] <= self.mutation_probability:
                    value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                        gene_idx=gene_idx,
                                                                        sample_size=self.sample_size)

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
                                                                                             sample_size=self.sample_size,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             build_initial_pop=False)
        return offspring

    def mutation_process_gene_value(self,
                                    solution,
                                    gene_idx,
                                    range_min=None,
                                    range_max=None,
                                    sample_size=100):

        """
        Generate/select values for the gene that satisfy the constraint. The values could be generated randomly or from the gene space.
        It accepts:
            -range_min: The minimum value in the range from which a value is selected.
            -range_max: The maximum value in the range from which a value is selected.
            -solution: The solution where the target gene exists.
            -gene_idx: The index of the gene in the solution.
            -sample_size: The number of random values to generate from which a value is selected. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values.
        It returns a single numeric value the satisfies the gene constraint if exists in the gene_constraint parameter. 
        """

        # Check if the gene has a constraint.
        if self.gene_constraint and self.gene_constraint[gene_idx]:
            # Generate values that meet the gene constraint. Select more than 1 value.
            # This method: 1) generates or selects the values 2) filters the values according to the constraint.
            values = self.get_valid_gene_constraint_values(range_min=range_min,
                                                           range_max=range_max,
                                                           gene_value=solution[gene_idx],
                                                           gene_idx=gene_idx,
                                                           mutation_by_replacement=self.mutation_by_replacement,
                                                           solution=solution,
                                                           sample_size=sample_size)
            if values is None:
                # No value found that satisfy the constraint.
                # Keep the old value.
                value_selected = solution[gene_idx]
            else:
                # Select a value randomly from the list of values satisfying the constraint.
                # If size is used with numpy.random.choice(), it returns an array even if it has a single value. To return a numeric value, not an array, then return index 0.
                value_selected = numpy.random.choice(values, size=1)[0]
        else:
            # The gene does not have a constraint. Just select a single value.
            value_selected = self.generate_gene_value(range_min=range_min,
                                                      range_max=range_max,
                                                      gene_value=solution[gene_idx],
                                                      gene_idx=gene_idx,
                                                      solution=solution,
                                                      mutation_by_replacement=self.mutation_by_replacement,
                                                      sample_size=1)
        # Even that its name is singular, it might have a multiple values.
        return value_selected

    def mutation_randomly(self, offspring):

        """
        Applies the random mutation.
        It accepts:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            # Return the indices of the genes to mutate.
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), 
                                                         self.mutation_num_genes))
            for gene_idx in mutation_indices:

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # Generate a random value for mutation that meet the gene constraint if exists.
                random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                range_max=range_max,
                                                                solution=offspring[offspring_idx],
                                                                gene_idx=gene_idx,
                                                                sample_size=self.sample_size)

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size)

        return offspring

    def mutation_probs_randomly(self, offspring):

        """
        Applies the random mutation using the mutation probability. For each gene, if its probability is <= that mutation probability, then it will be mutated randomly.
        It accepts:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            # The mutation probabilities for the current offspring.
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # A gene is mutated only if its mutation probability is less than or equal to the threshold.
                if probs[gene_idx] <= self.mutation_probability:

                    # Generate a random value fpr mutation that meet the gene constraint if exists.
                    random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                    range_max=range_max,
                                                                    solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size)
        return offspring

    def swap_mutation(self, offspring):

        """
        Applies the swap mutation which interchanges the values of 2 randomly selected genes.
        It accepts:
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
        It accepts:
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
        It accepts:
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
        It accepts:
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
        It accepts:
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
        It accepts:
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

                value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

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
                                                                                         sample_size=self.sample_size,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         build_initial_pop=False)
        return offspring

    def adaptive_mutation_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_num_genes' parameter. 
        A number of genes equal are selected randomly for mutation. This number depends on the fitness of the solution.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts:
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

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # Generate a random value fpr mutation that meet the gene constraint if exists.
                random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                range_max=range_max,
                                                                solution=offspring[offspring_idx],
                                                                gene_idx=gene_idx,
                                                                sample_size=self.sample_size)

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size)
        return offspring

    def adaptive_mutation_probs_by_space(self, offspring):

        """
        Applies the adaptive mutation based on the 2 parameters 'mutation_probability' and 'gene_space'.
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on space of values for each gene.
        It accepts:
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

                if probs[gene_idx] <= adaptive_mutation_probability:

                    value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                        gene_idx=gene_idx,
                                                                        sample_size=self.sample_size)

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
                                                                                             sample_size=self.sample_size,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             build_initial_pop=False)
        return offspring

    def adaptive_mutation_probs_randomly(self, offspring):

        """
        Applies the adaptive mutation based on the 'mutation_probability' parameter. 
        Based on whether the solution fitness is above or below a threshold, the mutation is applied diffrently by mutating high or low number of genes.
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts:
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

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                if probs[gene_idx] <= adaptive_mutation_probability:
                    # Generate a random value fpr mutation that meet the gene constraint if exists.
                    random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                    range_max=range_max,
                                                                    solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size)
        return offspring
