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
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        if self.gene_type_single == True:
            offspring = numpy.empty(offspring_size, dtype=self.gene_type[0])
        else:
            offspring = numpy.empty(offspring_size, dtype=object)

        for k in range(offspring_size[0]):
            # The point at which crossover takes place between two parents. Usually, it is at the center.
            crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]

            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(list(set(indices)), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # The new offspring has its first half of its genes from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring has its second half of its genes from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)

        
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

        for k in range(offspring_size[0]):
            if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
                crossover_point1 = 0
            else:
                crossover_point1 = numpy.random.randint(low=0, high=numpy.ceil(parents.shape[1]/2 + 1), size=1)[0]
    
            crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(list(set(indices)), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
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

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
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

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(list(set(indices)), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
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

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)

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

        for k in range(offspring_size[0]):
            if not (self.crossover_probability is None):
                probs = numpy.random.random(size=parents.shape[0])
                indices = numpy.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    offspring[k, :] = parents[k % parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    indices = random.sample(list(set(indices)), 2)
                    parent1_idx = indices[0]
                    parent2_idx = indices[1]
            else:
                # Index of the first parent to mate.
                parent1_idx = k % parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % parents.shape[0]

            # A 0/1 vector where 0 means the gene is taken from the first parent and 1 means the gene is taken from the second parent.
            gene_sources = numpy.random.randint(0, 2, size=self.num_genes)
            offspring[k, :] = numpy.where(gene_sources == 0, parents[parent1_idx, :], parents[parent2_idx, :])

            if self.allow_duplicate_genes == False:
                if self.gene_space is None:
                    offspring[k], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[k],
                                                                             min_val=self.random_mutation_min_val,
                                                                             max_val=self.random_mutation_max_val,
                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
                else:
                    offspring[k], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[k],
                                                                             gene_type=self.gene_type,
                                                                             num_trials=10)
        return offspring
