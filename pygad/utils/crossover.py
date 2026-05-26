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
        Apply single-point crossover between pairs of parents. One
        random split point is picked per offspring; the genes before
        the point come from the first parent and the genes from the
        point onward come from the second parent.

        Parameters
        ----------
        parents : numpy.ndarray
            A 2D array of parent solutions, one row per parent.
        offspring_size : tuple
            ``(num_offspring, num_genes)``. Shape of the offspring
            array to return.

        Returns
        -------
        offspring : numpy.ndarray
            A 2D array of the produced offspring with shape
            ``offspring_size``.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the K points at which crossover takes place between each two parents. The point does not have to be always at the center of the solutions.
        # This saves time by calling the numpy.random.randint() function only once.
        crossover_points = numpy.random.randint(low=0, 
                                                high=parents.shape[1], 
                                                size=offspring_size[0])

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

            # The first half of the offspring's genes comes from the first parent.
            offspring[k, 0:crossover_points[k]] = parents[parent1_idx, 0:crossover_points[k]]
            # The second half of the offspring's genes comes from the second parent.
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
        Apply two-points crossover. Two split points are picked per
        offspring; the genes outside ``[point1, point2)`` come from the
        first parent and the genes inside come from the second parent.

        Parameters
        ----------
        parents : numpy.ndarray
            A 2D array of parent solutions, one row per parent.
        offspring_size : tuple
            ``(num_offspring, num_genes)``. Shape of the offspring
            array to return.

        Returns
        -------
        offspring : numpy.ndarray
            A 2D array of the produced offspring with shape
            ``offspring_size``.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the first K points at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_points_1 = numpy.zeros(offspring_size[0])
        else:
            crossover_points_1 = numpy.random.randint(low=0, 
                                                      high=numpy.ceil(parents.shape[1]/2 + 1), 
                                                      size=offspring_size[0])

        # The second point must always be greater than the first point.
        crossover_points_2 = crossover_points_1 + int(parents.shape[1]/2) 

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
        Apply uniform crossover. For each gene independently, the value
        is copied from one of the two mating parents picked at random.

        Parameters
        ----------
        parents : numpy.ndarray
            A 2D array of parent solutions, one row per parent.
        offspring_size : tuple
            ``(num_offspring, num_genes)``. Shape of the offspring
            array to return.

        Returns
        -------
        offspring : numpy.ndarray
            A 2D array of the produced offspring with shape
            ``offspring_size``.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the genes sources at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        # There is a list of 0 and 1 for each offspring.
        # [0, 1, 0, 0, 1, 1]: If the value is 0, then take the gene from the first parent. If 1, take it from the second parent.
        genes_sources = numpy.random.randint(low=0, 
                                             high=2, 
                                             size=offspring_size)

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

            # The gene will be copied from the first parent if the current gene index is 0.
            # The gene will be copied from the second parent if the current gene index is 1.
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

    def scattered_crossover(self, parents, offspring_size):
        """
        Apply scattered crossover. For each gene independently, the
        value is copied from one of the two mating parents picked at
        random. (Same mechanic as ``uniform_crossover`` but kept as a
        separate operator for backwards compatibility.)

        Parameters
        ----------
        parents : numpy.ndarray
            A 2D array of parent solutions, one row per parent.
        offspring_size : tuple
            ``(num_offspring, num_genes)``. Shape of the offspring
            array to return.

        Returns
        -------
        offspring : numpy.ndarray
            A 2D array of the produced offspring with shape
            ``offspring_size``.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        # Randomly generate all the genes sources at which crossover takes place between each two parents. 
        # This saves time by calling the numpy.random.randint() function only once.
        # There is a list of 0 and 1 for each offspring.
        # [0, 1, 0, 0, 1, 1]: If the value is 0, then take the gene from the first parent. If 1, take it from the second parent.
        genes_sources = numpy.random.randint(low=0, 
                                             high=2, 
                                             size=offspring_size)

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

            # The gene will be copied from the first parent if the current gene index is 0.
            # The gene will be copied from the second parent if the current gene index is 1.
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
