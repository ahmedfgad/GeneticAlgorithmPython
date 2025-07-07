"""
The pygad.helper.unique module has helper methods to solve duplicate genes and make sure every gene is unique.
"""

import numpy
import warnings
import random
import pygad

class Unique:

    def solve_duplicate_genes_randomly(self, 
                                       solution, 
                                       min_val, 
                                       max_val, 
                                       mutation_by_replacement, 
                                       gene_type, 
                                       sample_size=100):
        """
        Resolves duplicates in a solution by randomly selecting new values for the duplicate genes.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            min_val (int): The minimum value of the range to sample a number randomly.
            max_val (int): The maximum value of the range to sample a number randomly.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            gene_type (type): The data type of the gene (e.g., int, float).
            sample_size (int): The maximum number of random values to generate to find a unique value.

        Returns:
            tuple:
                list: The updated solution after attempting to resolve duplicates. If no duplicates are resolved, the solution remains unchanged.
                list: The indices of genes that still have duplicate values.
                int: The number of duplicates that could not be resolved.
        """

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)

        num_unsolved_duplicates = 0
        if len(not_unique_indices) > 0:
            for duplicate_index in not_unique_indices:
                dtype = self.get_gene_dtype(gene_index=duplicate_index)

                if type(min_val) in self.supported_int_float_types:
                    min_val_gene = min_val
                    max_val_gene = max_val
                else:
                    min_val_gene = min_val[duplicate_index]
                    max_val_gene = max_val[duplicate_index]

                if dtype[0] in pygad.GA.supported_int_types:
                    temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                               gene_index=duplicate_index, 
                                                               min_val=min_val_gene,
                                                               max_val=max_val_gene,
                                                               mutation_by_replacement=mutation_by_replacement, 
                                                               gene_type=gene_type)
                else:
                    temp_val = self.unique_float_gene_from_range(solution=new_solution, 
                                                                 gene_index=duplicate_index, 
                                                                 min_val=min_val_gene,
                                                                 max_val=max_val_gene,
                                                                 mutation_by_replacement=mutation_by_replacement, 
                                                                 gene_type=gene_type, 
                                                                 sample_size=sample_size)
 
                if temp_val in new_solution:
                    num_unsolved_duplicates = num_unsolved_duplicates + 1
                    if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {solution[duplicate_index]} at generation {self.generations_completed}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
                else:
                    # Unique gene value found.
                    new_solution[duplicate_index] = temp_val

        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # self.logger.info("not_unique_indices INSIDE", not_unique_indices)

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def solve_duplicate_genes_by_space(self, 
                                       solution, 
                                       gene_type, 
                                       mutation_by_replacement,
                                       sample_size=100,
                                       build_initial_pop=False):

        """
        Resolves duplicates in a solution by selecting new values for the duplicate genes from the gene space.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_type (type): The data type of the gene (e.g., int, float).
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            sample_size (int, optional): The maximum number of attempts to resolve duplicates by selecting values from the gene space.
            build_initial_pop (bool, optional): Indicates if initial population should be built.

        Returns:
            tuple:
                list: The updated solution after attempting to resolve duplicates. If no duplicates are resolved, the solution remains unchanged.
                list: The indices of genes that still have duplicate values.
                int: The number of duplicates that could not be resolved.
        """

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)

        # First try to solve the duplicates.
        # For a solution like [3 2 0 0], the indices of the 2 duplicating genes are 2 and 3.
        # The next call to the find_unique_value() method tries to change the value of the gene with index 3 to solve the duplicate.
        if len(not_unique_indices) > 0:
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(solution=new_solution,
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   sample_size=sample_size,
                                                                                                   mutation_by_replacement=mutation_by_replacement,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            return new_solution, not_unique_indices, len(not_unique_indices)

        # DEEP-DUPLICATE-REMOVAL-NEEDED
        # Search by this phrase to find where deep duplicates removal should be applied.
        # If there exist duplicate genes, then changing either of the 2 duplicating genes (with indices 2 and 3) will not solve the problem.
        # This problem can be solved by randomly changing one of the non-duplicating genes that may make a room for a unique value in one the 2 duplicating genes.
        # For example, if gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]] and the solution is [3 2 0 0], then the values of the last 2 genes duplicate.
        # There are no possible changes in the last 2 genes to solve the problem. But it could be solved by changing the second gene from 2 to 4.
        # As a result, any of the last 2 genes can take the value 2 and solve the duplicates.

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_int_gene_from_range(self, 
                                   solution, 
                                   gene_index, 
                                   min_val, 
                                   max_val, 
                                   mutation_by_replacement, 
                                   gene_type, 
                                   step=1):

        """
        Finds a unique integer value for a specific gene in a solution.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_index (int): The index of the gene for which to find a unique value.
            min_val (int): The minimum value of the range to sample an integer randomly.
            max_val (int): The maximum value of the range to sample an integer randomly.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            gene_type (type): The data type of the gene (e.g., int, int8, uint16, etc).
            step (int, optional): The step size for generating candidate values. Defaults to 1.

        Returns:
            int: The new integer value of the gene. If no unique value can be found, the original gene value is returned.
        """

        if self.gene_constraint and self.gene_constraint[gene_index]:
            # A unique value is created out of the values that satisfy the constraint.
            # sample_size=None to return all the values.
            random_values = self.get_valid_gene_constraint_values(range_min=min_val,
                                                                  range_max=max_val,
                                                                  gene_value=solution[gene_index],
                                                                  gene_idx=gene_index,
                                                                  mutation_by_replacement=mutation_by_replacement,
                                                                  solution=solution,
                                                                  sample_size=None,
                                                                  step=step)
            # If there is no value satisfying the constraint, then return the current gene value.
            if random_values is None:
                return solution[gene_index]
            else:
                pass
        else:
            # There is no constraint for the current gene. Return the same range.
            # sample_size=None to return all the values.
            random_values = self.generate_gene_value(range_min=min_val,
                                                     range_max=max_val,
                                                     gene_value=solution[gene_index],
                                                     gene_idx=gene_index,
                                                     solution=solution,
                                                     mutation_by_replacement=mutation_by_replacement,
                                                     sample_size=None,
                                                     step=step)

        selected_value = self.select_unique_value(gene_values=random_values, 
                                                  solution=solution, 
                                                  gene_index=gene_index)

        # The gene_type is of the form [type, precision]
        selected_value = gene_type[0](selected_value)
    
        return selected_value

    def unique_float_gene_from_range(self, 
                                     solution, 
                                     gene_index, 
                                     min_val, 
                                     max_val, 
                                     mutation_by_replacement, 
                                     gene_type, 
                                     sample_size=100):

        """
        Finds a unique floating-point value for a specific gene in a solution.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_index (int): The index of the gene for which to find a unique value.
            min_val (int): The minimum value of the range to sample a floating-point number randomly.
            max_val (int): The maximum value of the range to sample a floating-point number randomly.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            gene_type (type): The data type of the gene (e.g., float, float16, float32, etc).
            sample_size (int): The maximum number of random values to generate to find a unique value.

        Returns:
            int: The new floating-point value of the gene. If no unique value can be found, the original gene value is returned.
        """

        if self.gene_constraint and self.gene_constraint[gene_index]:
            # A unique value is created out of the values that satisfy the constraint.
            values = self.get_valid_gene_constraint_values(range_min=min_val,
                                                           range_max=max_val,
                                                           gene_value=solution[gene_index],
                                                           gene_idx=gene_index,
                                                           mutation_by_replacement=mutation_by_replacement,
                                                           solution=solution,
                                                           sample_size=sample_size)
            # If there is no value satisfying the constraint, then return the current gene value.
            if values is None:
                return solution[gene_index]
            else:
                pass
        else:
            # There is no constraint for the current gene. Return the same range.
            values = self.generate_gene_value(range_min=min_val,
                                              range_max=max_val,
                                              gene_value=solution[gene_index],
                                              gene_idx=gene_index,
                                              solution=solution,
                                              mutation_by_replacement=mutation_by_replacement,
                                              sample_size=sample_size)

        selected_value = self.select_unique_value(gene_values=values,
                                                  solution=solution, 
                                                  gene_index=gene_index)
        return selected_value

    def select_unique_value(self,
                            gene_values,
                            solution,
                            gene_index):

        """
        Select a unique value (if possible) from a list of gene values.

        Args:
            gene_values (NumPy Array): An array of values from which a unique value should be selected.
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_index (int): The index of the gene for which to find a unique value.

        Returns:
            selected_gene: The new (hopefully unique) value of the gene. If no unique value can be found, the original gene value is returned.
        """

        values_to_select_from = list(set(list(gene_values)) - set(solution))

        if len(values_to_select_from) == 0:
            if solution[gene_index] is None:
                # The initial population is created as an empty array (numpy.empty()).
                # If we are assigning values to the initial population, then the gene value is already None.
                # If the gene value is None, then we do not have an option other than selecting a value even if it causes duplicates.
                # If there is no value that is unique to the solution, then select any of the current values randomly from the current set of gene values.
                selected_value = random.choice(gene_values)
            else:
                # If the gene is not None, then just keep its current value as long as there are no values that make it unique.
                selected_value = solution[gene_index]
        else:
            selected_value = random.choice(values_to_select_from)
        return selected_value

    def unique_genes_by_space(self, 
                              solution,
                              gene_type, 
                              not_unique_indices,
                              mutation_by_replacement,
                              sample_size=100,
                              build_initial_pop=False):

        """
        Iterates through all duplicate genes to find unique values from their gene spaces and resolve duplicates.
        For each duplicate gene, a call is made to the `unique_gene_by_space()` function.

        Args:
            solution (list): A solution containing genes with duplicate values.
            gene_type (type): The data type of the all the genes (e.g., int, float).
            not_unique_indices (list): The indices of genes with duplicate values.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            sample_size (int): The maximum number of attempts to resolve duplicates for each gene. Only works for floating-point numbers.
            build_initial_pop (bool, optional): Indicates if initial population should be built.

        Returns:
            tuple:
                list: The updated solution after attempting to resolve all duplicates. If no duplicates are resolved, the solution remains unchanged.
                list: The indices of genes that still have duplicate values.
                int: The number of duplicates that could not be resolved.
        """

        num_unsolved_duplicates = 0
        for duplicate_index in not_unique_indices:
            temp_val = self.unique_gene_by_space(solution=solution,
                                                 gene_idx=duplicate_index, 
                                                 gene_type=gene_type,
                                                 mutation_by_replacement=mutation_by_replacement,
                                                 sample_size=sample_size,
                                                 build_initial_pop=build_initial_pop)

            if temp_val in solution:
                num_unsolved_duplicates = num_unsolved_duplicates + 1
                if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {solution[duplicate_index]} at generation {self.generations_completed+1}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
            else:
                solution[duplicate_index] = temp_val
    
        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)

        return solution, not_unique_indices, num_unsolved_duplicates

    def unique_gene_by_space(self, 
                             solution, 
                             gene_idx, 
                             gene_type,
                             mutation_by_replacement,
                             sample_size=100,
                             build_initial_pop=False):
    
        """
        Returns a unique value for a specific gene based on its value space to resolve duplicates.

        Args:
            solution (list): A solution containing genes with duplicate values.
            gene_idx (int): The index of the gene that has a duplicate value.
            gene_type (type): The data type of the gene (e.g., int, float).
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            sample_size (int): The maximum number of attempts to resolve duplicates for each gene. Only works for floating-point numbers.
            build_initial_pop (bool, optional): Indicates if initial population should be built.

        Returns:
            Any: A unique value for the gene, if one exists; otherwise, the original gene value.            
        """

        # When gene_value is None, this forces the gene value generators to select a value for use by the initial population.
        # Otherwise, it considers selecting a value for mutation.
        if build_initial_pop:
            gene_value = None
        else:
            gene_value = solution[gene_idx]

        if self.gene_constraint and self.gene_constraint[gene_idx]:
            # A unique value is created out of the values that satisfy the constraint.
            values = self.get_valid_gene_constraint_values(range_min=None,
                                                           range_max=None,
                                                           gene_value=gene_value,
                                                           gene_idx=gene_idx,
                                                           mutation_by_replacement=mutation_by_replacement,
                                                           solution=solution,
                                                           sample_size=sample_size)
            # If there is no value satisfying the constraint, then return the current gene value.
            if values is None:
                return solution[gene_idx]
            else:
                pass
        else:
            # There is no constraint for the current gene. Return the same range.
            values = self.generate_gene_value(range_min=None,
                                              range_max=None,
                                              gene_value=gene_value,
                                              gene_idx=gene_idx,
                                              solution=solution,
                                              mutation_by_replacement=mutation_by_replacement,
                                              sample_size=sample_size)

        selected_value = self.select_unique_value(gene_values=values,
                                                  solution=solution,
                                                  gene_index=gene_idx)

        return selected_value

    def find_two_duplicates(self, 
                            solution,
                            gene_space_unpacked):
        """
        Identifies the first occurrence of a duplicate gene in the solution.

        Args:
            solution: The solution containing genes with duplicate values.
            gene_space_unpacked: A list of values from the gene space to choose the values that resolve duplicates.

        Returns:
            int: The index of the first gene with a duplicate value.
            Any: The value of the duplicate gene.
        """

        for gene in set(solution):
            gene_indices = numpy.where(numpy.array(solution) == gene)[0]
            if len(gene_indices) == 1:
                continue
            for gene_idx in gene_indices:
                number_alternate_values = len(set(gene_space_unpacked[gene_idx]))
                if number_alternate_values > 1:
                    return gene_idx, gene
        # This means there is no way to solve the duplicates between the genes.
        # Because the space of the duplicates genes only has a single value and there is no alternatives.
        return None, gene

    def unpack_gene_space(self, 
                          range_min,
                          range_max,
                          sample_size_from_inf_range=100):
        """
        Unpacks the gene space for selecting a value to resolve duplicates by converting ranges into lists of values.

        Args:
            range_min (float or int): The minimum value of the range.
            range_max (float or int): The maximum value of the range.
            sample_size_from_inf_range (int): The number of values to generate for an infinite range of float values using `numpy.linspace()`.

        Returns:
            list: A list representing the unpacked gene space.
        """

        # Copy the gene_space to keep it isolated form the changes.
        if self.gene_space is None:
            return None

        if self.gene_space_nested == False:
            if type(self.gene_space) is range:
                gene_space_unpacked = list(self.gene_space)
            elif type(self.gene_space) in [numpy.ndarray, list]:
                gene_space_unpacked = self.gene_space.copy()
            elif type(self.gene_space) is dict:
                if 'step' in self.gene_space.keys():
                    gene_space_unpacked = numpy.arange(start=self.gene_space['low'],
                                                       stop=self.gene_space['high'],
                                                       step=self.gene_space['step'])
                else:
                    gene_space_unpacked = numpy.linspace(start=self.gene_space['low'],
                                                         stop=self.gene_space['high'],
                                                         num=sample_size_from_inf_range,
                                                         endpoint=False)

            if self.gene_type_single == True:
                # Change the data type.
                for idx in range(len(gene_space_unpacked)):
                    if gene_space_unpacked[idx] is None:
                        gene_space_unpacked[idx] = numpy.random.uniform(low=range_min,
                                                                        high=range_max)
                gene_space_unpacked = numpy.array(gene_space_unpacked,
                                                  dtype=self.gene_type[0])
                if not self.gene_type[1] is None:
                    # Round the values for float (non-int) data types.
                    gene_space_unpacked = numpy.round(gene_space_unpacked,
                                                      self.gene_type[1])
            else:
                temp_gene_space_unpacked = gene_space_unpacked.copy()
                gene_space_unpacked = []
                # Get the number of genes from the length of gene_type.
                # The num_genes attribute is not set yet when this method (unpack_gene_space) is called for the first time.
                for gene_idx in range(len(self.gene_type)):
                    # Change the data type.
                    gene_space_item_unpacked = numpy.array(temp_gene_space_unpacked,
                                                           self.gene_type[gene_idx][0])
                    if not self.gene_type[gene_idx][1] is None:
                        # Round the values for float (non-int) data types.
                        gene_space_item_unpacked = numpy.round(temp_gene_space_unpacked,
                                                               self.gene_type[gene_idx][1])
                    gene_space_unpacked.append(gene_space_item_unpacked)

        elif self.gene_space_nested == True:
            gene_space_unpacked = self.gene_space.copy()
            for space_idx, space in enumerate(gene_space_unpacked):
                if type(space) in pygad.GA.supported_int_float_types:
                    gene_space_unpacked[space_idx] = [space]
                elif space is None:
                    # Randomly generate the value using the mutation range.
                    gene_space_unpacked[space_idx] = numpy.arange(start=range_min,
                                                                  stop=range_max)
                elif type(space) is range:
                    # Convert the range to a list.
                    gene_space_unpacked[space_idx] = list(space)
                elif type(space) is dict:
                    # Create a list of values using the dict range.
                    # Use numpy.linspace()
                    dtype = self.get_gene_dtype(gene_index=space_idx)

                    if dtype[0] in pygad.GA.supported_int_types:
                        if 'step' in space.keys():
                            step = space['step']
                        else:
                            step = 1

                        gene_space_unpacked[space_idx] = numpy.arange(start=space['low'],
                                                                      stop=space['high'],
                                                                      step=step)
                    else:
                        if 'step' in space.keys():
                            gene_space_unpacked[space_idx] = numpy.arange(start=space['low'],
                                                                          stop=space['high'],
                                                                          step=space['step'])
                        else:
                            gene_space_unpacked[space_idx] = numpy.linspace(start=space['low'],
                                                                            stop=space['high'],
                                                                            num=sample_size_from_inf_range,
                                                                            endpoint=False)    
                elif type(space) in [numpy.ndarray, list, tuple]:
                    # list/tuple/numpy.ndarray
                    # Convert all to list
                    gene_space_unpacked[space_idx] = list(space)
    
                    # Check if there is an item with the value None. If so, replace it with a random value using the mutation range.
                    none_indices = numpy.where(numpy.array(gene_space_unpacked[space_idx]) == None)[0]
                    if len(none_indices) > 0:
                        for idx in none_indices:
                            random_value = numpy.random.uniform(low=range_min,
                                                                high=range_max,
                                                                size=1)[0]
                            gene_space_unpacked[space_idx][idx] = random_value
    
                dtype = self.get_gene_dtype(gene_index=space_idx)

                # Change the data type.
                gene_space_unpacked[space_idx] = numpy.array(gene_space_unpacked[space_idx],
                                                             dtype=dtype[0])
                if not dtype[1] is None:
                    # Round the values for float (non-int) data types.
                    gene_space_unpacked[space_idx] = numpy.round(gene_space_unpacked[space_idx],
                                                                 dtype[1])

        return gene_space_unpacked

    def solve_duplicates_deeply(self,
                                solution):
        """
        Sometimes it is impossible to solve the duplicate genes by simply selecting another value for either genes.
        This function solve the duplicates between 2 genes by searching for a third gene that can make assist in the solution.

        Args:
            solution (list): The current solution containing genes, potentially with duplicates.

        Returns:
            list or None: The updated solution with duplicates resolved, or `None` if the duplicates cannot be resolved.
        """

        # gene_space_unpacked = self.unpack_gene_space()
        # Create a copy of the gene_space_unpacked attribute because it will be changed later.
        gene_space_unpacked = self.gene_space_unpacked.copy()

        duplicate_index, duplicate_value = self.find_two_duplicates(solution, 
                                                                    gene_space_unpacked)
    
        if duplicate_index is None:
            # Impossible to solve the duplicates for the genes with value duplicate_value.
            return None
    
    
        # Without copy(), the gene will be removed from the gene_space.
        # Convert the space to list because tuples do not have copy()
        gene_other_values = list(gene_space_unpacked[duplicate_index]).copy()

        # This removes all the occurrences of this value.
        gene_other_values = [v for v in gene_other_values if v != duplicate_value]

        # Two conditions to solve the duplicates of the value D:
            # 1. From gene_other_values, select a value V such that it is available in the gene space of another gene X.
            # 2. Find an alternate value for the gene X that will not cause any duplicates.
            #    2.1 If the gene X does not have alternatives, then go back to step 1 to find another gene.
            #    2.2 Set the gene X to the value D.
            #    2.3 Set the target gene to the value V.
        # Set the space of the duplicate gene to empty list []. Do not remove it to not alter the indices of the gene spaces.
        gene_space_unpacked[duplicate_index] = []

        for other_value in gene_other_values:
            for space_idx, space in enumerate(gene_space_unpacked):
                if other_value in space:
                    if other_value in solution and list(solution).index(other_value) != space_idx:
                        continue
                    else:
                        # Find an alternate value for the third gene.
                        # Copy the space so that the original space is not changed after removing the value.
                        space_other_values = space.copy()
                        # This removes all the occurrences of this value. It is not enough to use the remove() function because it only removes the first occurrence.
                        space_other_values = [v for v in space_other_values if v != other_value]

                        for val in space_other_values:
                            if val in solution:
                                # If the value exists in another gene of the solution, then we cannot use this value as it will cause another duplicate.
                                # End the current iteration and go check another value.
                                continue
                            else:
                                solution[space_idx] = val
                                solution[duplicate_index] = other_value
                                return solution

        # Reaching here means we cannot solve the duplicate genes.
        return None
