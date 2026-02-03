"""
The pygad.utils.crossover module has all the built-in crossover operators.
"""

import numpy
import random

class Crossover:

    def __init__():
        pass

    def single_point_crossover(self, parents, offspring_size):

        """
        Applies single-point crossover between pairs of parents.
        This function selects a random point at which crossover occurs between the parents, generating offspring.

        Parameters:
            parents (array-like): The parents to mate for producing the offspring.
            offspring_size (int): The number of offspring to produce.

        Returns:
            array-like: An array containing the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the K points at which crossover takes place between each two parents. The point does not have to be always at the center of the solutions.
        # This saves time by calling the numpy.random.randint() function only once.
        if self.gene_structure is None:
            crossover_points = numpy.random.randint(low=0, 
                                                    high=parents.shape[1], 
                                                    size=offspring_size[0])
        else:
            # Select random boundary index excluding the 0th index and last index to ensure split?
            # boundaries: [0, 2, 5, 10] -> valid cuts are 2, 5. So boundaries[1:-1]
            # If structure has only 1 block, crossover point is meaningless? Standard GA allows index (0..N).
            # Standard: low=0, high=N. Point K means split at index K.
            # 0 -> Empty first part, Full second part.
            # N -> Full first part, Empty second.
            # So boundaries are valid crossover points immediately.
            valid_cuts = self.boundaries
            crossover_points = numpy.random.choice(valid_cuts, size=offspring_size[0])

        for k in range(offspring_size[0]):
            # Check if the crossover_probability parameter is used.
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = list(set(numpy.where(probs <= self.crossover_probability)[0]))

                # If no parent satisfied the probability, no crossover is applied and a parent is selected as is.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(indices, 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_points[k]] = parents[parent1_idx, 0:crossover_points[k]]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_points[k]:] = parents[parent2_idx, crossover_points[k]:]

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             build_initial_pop=False)
        
        return offspring

    def two_points_crossover(self, parents, offspring_size):

        """
        Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the first K points at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        if self.gene_structure is None:
            if (parents.shape[1] == 1):
                crossover_points_1 = numpy.zeros(offspring_size[0])
            else:
                crossover_points_1 = numpy.random.randint(low=0, 
                                                          high=numpy.ceil(parents.shape[1]/2 + 1), 
                                                          size=offspring_size[0])
            # The second point must always be greater than the first point.
            crossover_points_2 = crossover_points_1 + int(parents.shape[1]/2) 
        else:
            num_logical = len(self.gene_structure)
            if num_logical < 2:
                # Can't do meaningful 2-point on < 2 blocks? Standard allows splitting anywhere.
                # If only 1 block [0, 10], Boundaries are 0, 10.
                # Cuts: 0, 10.
                crossover_points_1 = numpy.zeros(offspring_size[0], dtype=int)
                crossover_points_2 = numpy.full(offspring_size[0], parents.shape[1], dtype=int)
            else:
                # Select 2 distinct boundary indices.
                # To ensure p1 != p2 efficiently for N offspring:
                # 1. Pick p1_idx from [0, num_boundaries).
                # 2. Pick offset from [1, num_boundaries).
                # 3. p2_idx = (p1_idx + offset) % num_boundaries.
                # This guarantees p1_idx != p2_idx.
                
                num_boundaries = len(self.boundaries)
                # We need indices [0, num_boundaries-1]
                
                # p1 indices:
                p1_idx = numpy.random.randint(low=0, high=num_boundaries, size=offspring_size[0])
                
                # Offsets: at least 1, at most num_boundaries-1
                # If num_boundaries <= 1 (impossible if num_logical >= 1, boundaries has at least 0, N), 
                # but if num_logical=1, boundaries=[0, N]. len=2. offset must be 1.
                # randint(1, 2) -> returns [1]. Correct.
                
                offsets = numpy.random.randint(low=1, high=num_boundaries, size=offspring_size[0])
                
                p2_idx = (p1_idx + offsets) % num_boundaries
                
                # Gather points
                # Stack to sort
                points_idx = numpy.column_stack((p1_idx, p2_idx))
                points_idx.sort(axis=1)
                
                # Map indices to actual boundary values
                crossover_points_1 = self.boundaries[points_idx[:, 0]]
                crossover_points_2 = self.boundaries[points_idx[:, 1]] 

        for k in range(offspring_size[0]):

            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = list(set(numpy.where(probs <= self.crossover_probability)[0]))

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(indices, 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
            offspring[k, 0:crossover_points_1[k]] = parents[parent1_idx, 0:crossover_points_1[k]]
            # The genes from the second point up to the end of the chromosome are copied from the first parent.
            offspring[k, crossover_points_2[k]:] = parents[parent1_idx, crossover_points_2[k]:]
            # The genes between the 2 points are copied from the second parent.
            offspring[k, crossover_points_1[k]:crossover_points_2[k]] = parents[parent2_idx, crossover_points_1[k]:crossover_points_2[k]]

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             build_initial_pop=False)
        return offspring

    def uniform_crossover(self, parents, offspring_size):

        """
        Applies the uniform crossover. For each gene, a parent out of the 2 mating parents is selected randomly and the gene is copied from it.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the genes sources at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        # There is a list of 0 and 1 for each offspring.
        # [0, 1, 0, 0, 1, 1]: If the value is 0, then take the gene from the first parent. If 1, take it from the second parent.
        if self.gene_structure is None:
            genes_sources = numpy.random.randint(low=0, 
                                                 high=2, 
                                                 size=offspring_size)
        else:
            # Generate sources for LOGICAL blocks
            num_logical = len(self.gene_structure)
            logical_sources = numpy.random.randint(low=0, high=2, size=(offspring_size[0], num_logical))
            
            # Map logical sources to full gene mask
            genes_sources = numpy.empty(offspring_size, dtype=int)
            for k in range(offspring_size[0]):
                for b_idx in range(num_logical):
                    start = self.boundaries[b_idx]
                    end = self.boundaries[b_idx+1]
                    genes_sources[k, start:end] = logical_sources[k, b_idx]

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = list(set(numpy.where(probs <= self.crossover_probability)[0]))

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(indices, 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            for gene_idx in range(offspring_size[1]):
                if (genes_sources[k, gene_idx] == 0):
                    # The gene will be copied from the first parent if the current gene index is 0.
                    offspring[k, gene_idx] = parents[parent1_idx, gene_idx]
                elif (genes_sources[k, gene_idx] == 1):
                    # The gene will be copied from the second parent if the current gene index is 1.
                    offspring[k, gene_idx] = parents[parent2_idx, gene_idx]

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             build_initial_pop=False)

        return offspring

    def scattered_crossover(self, parents, offspring_size):

        """
        Applies the scattered crossover. It randomly selects the gene from one of the 2 parents. 
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the genes sources at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        # There is a list of 0 and 1 for each offspring.
        # [0, 1, 0, 0, 1, 1]: If the value is 0, then take the gene from the first parent. If 1, take it from the second parent.
        if self.gene_structure is None:
            genes_sources = numpy.random.randint(low=0, 
                                                 high=2, 
                                                 size=offspring_size)
        else:
            # Generate sources for LOGICAL blocks
            num_logical = len(self.gene_structure)
            logical_sources = numpy.random.randint(low=0, high=2, size=(offspring_size[0], num_logical))
            
            # Map logical sources to full gene mask
            genes_sources = numpy.empty(offspring_size, dtype=int)
            for k in range(offspring_size[0]):
                for b_idx in range(num_logical):
                    start = self.boundaries[b_idx]
                    end = self.boundaries[b_idx+1]
                    genes_sources[k, start:end] = logical_sources[k, b_idx]

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = list(set(numpy.where(probs <= self.crossover_probability)[0]))

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(indices, 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            offspring[k, :] = numpy.where(genes_sources[k] == 0, 
                                          parents[parent1_idx, :], 
                                          parents[parent2_idx, :])

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             sample_size=self.sample_size,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             build_initial_pop=False)
        return offspring
