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
                                       num_trials=10):
        """
        Resolves duplicates in a solution by randomly selecting new values for the duplicate genes.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            min_val (int): The minimum value of the range to sample a number randomly.
            max_val (int): The maximum value of the range to sample a number randomly.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            gene_type (type): The data type of the gene (e.g., int, float).
            num_trials (int): The maximum number of attempts to resolve duplicates by changing the gene values. Only works for floating-point gene types.

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
                if self.gene_type_single == True:
                    dtype = gene_type
                else:
                    dtype = gene_type[duplicate_index]

                if dtype[0] in pygad.GA.supported_int_types:
                    temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                               gene_index=duplicate_index, 
                                                               min_val=min_val, 
                                                               max_val=max_val, 
                                                               mutation_by_replacement=mutation_by_replacement, 
                                                               gene_type=gene_type)
                else:
                    temp_val = self.unique_float_gene_from_range(solution=new_solution, 
                                                                 gene_index=duplicate_index, 
                                                                 min_val=min_val, 
                                                                 max_val=max_val, 
                                                                 mutation_by_replacement=mutation_by_replacement, 
                                                                 gene_type=gene_type, 
                                                                 num_trials=num_trials)
 
                if temp_val in new_solution:
                    num_unsolved_duplicates = num_unsolved_duplicates + 1
                    if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {solution[duplicate_index]}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
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
                                       num_trials=10, 
                                       build_initial_pop=False):
    
        """
        Resolves duplicates in a solution by selecting new values for the duplicate genes from the gene space.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_type (type): The data type of the gene (e.g., int, float).
            num_trials (int): The maximum number of attempts to resolve duplicates by selecting values from the gene space.

        Returns:
            tuple:
                list: The updated solution after attempting to resolve duplicates. If no duplicates are resolved, the solution remains unchanged.
                list: The indices of genes that still have duplicate values.
                int: The number of duplicates that could not be resolved.
        """

        new_solution = solution.copy()

        _, unique_gene_indices = numpy.unique(solution, return_index=True)
        not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
        # self.logger.info("not_unique_indices OUTSIDE", not_unique_indices)

        # First try to solve the duplicates.
        # For a solution like [3 2 0 0], the indices of the 2 duplicating genes are 2 and 3.
        # The next call to the find_unique_value() method tries to change the value of the gene with index 3 to solve the duplicate.
        if len(not_unique_indices) > 0:
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            return new_solution, not_unique_indices, len(not_unique_indices)
    
        # Do another try if there exist duplicate genes.
        # If there are no possible values for the gene 3 with index 3 to solve the duplicate, try to change the value of the other gene with index 2.
        if len(not_unique_indices) > 0:
            not_unique_indices = set(numpy.where(new_solution == new_solution[list(not_unique_indices)[0]])[0]) - set([list(not_unique_indices)[0]])
            new_solution, not_unique_indices, num_unsolved_duplicates = self.unique_genes_by_space(new_solution=new_solution, 
                                                                                                   gene_type=gene_type, 
                                                                                                   not_unique_indices=not_unique_indices, 
                                                                                                   num_trials=10,
                                                                                                   build_initial_pop=build_initial_pop)
        else:
            # DEEP-DUPLICATE-REMOVAL-NEEDED
            # Search by this phrase to find where deep duplicates removal should be applied.

            # If there exist duplicate genes, then changing either of the 2 duplicating genes (with indices 2 and 3) will not solve the problem.
            # This problem can be solved by randomly changing one of the non-duplicating genes that may make a room for a unique value in one the 2 duplicating genes.
            # For example, if gene_space=[[3, 0, 1], [4, 1, 2], [0, 2], [3, 2, 0]] and the solution is [3 2 0 0], then the values of the last 2 genes duplicate.
            # There are no possible changes in the last 2 genes to solve the problem. But it could be solved by changing the second gene from 2 to 4.
            # As a result, any of the last 2 genes can take the value 2 and solve the duplicates.
            return new_solution, not_unique_indices, len(not_unique_indices)

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

        # The gene_type is of the form [type, precision]
        dtype = gene_type

        # For non-integer steps, the numpy.arange() function returns zeros if the dtype parameter is set to an integer data type. So, this returns zeros if step is non-integer and dtype is set to an int data type: numpy.arange(min_val, max_val, step, dtype=gene_type[0])
        # To solve this issue, the data type casting will not be handled inside numpy.arange(). The range is generated by numpy.arange() and then the data type is converted using the numpy.asarray() function.
        all_gene_values = numpy.asarray(numpy.arange(min_val, 
                                                     max_val, 
                                                     step), 
                                        dtype=dtype[0])

        # If mutation is by replacement, do not add the current gene value into the list.
        # This is to avoid replacing the value by itself again. We are doing nothing in this case.
        if mutation_by_replacement:
            pass
        else:
            all_gene_values = all_gene_values + solution[gene_index]

            # After adding solution[gene_index] to the list, we have to change the data type again.
            all_gene_values = numpy.asarray(all_gene_values, 
                                            dtype[0])

        values_to_select_from = list(set(list(all_gene_values)) - set(solution))
    
        if len(values_to_select_from) == 0:
            # If there are no values, then keep the current gene value.
            selected_value = solution[gene_index]
        else:
            selected_value = random.choice(values_to_select_from)

        selected_value = dtype[0](selected_value)
    
        return selected_value

    def unique_float_gene_from_range(self, 
                                     solution, 
                                     gene_index, 
                                     min_val, 
                                     max_val, 
                                     mutation_by_replacement, 
                                     gene_type, 
                                     num_trials=10):

        """
        Finds a unique floating-point value for a specific gene in a solution.

        Args:
            solution (list): A solution containing genes, potentially with duplicate values.
            gene_index (int): The index of the gene for which to find a unique value.
            min_val (int): The minimum value of the range to sample a floating-point number randomly.
            max_val (int): The maximum value of the range to sample a floating-point number randomly.
            mutation_by_replacement (bool): Indicates if mutation is performed by replacement.
            gene_type (type): The data type of the gene (e.g., float, float16, float32, etc).
            num_trials (int): The maximum number of attempts to resolve duplicates by changing the gene values.

        Returns:
            int: The new floating-point value of the gene. If no unique value can be found, the original gene value is returned.
        """

        # The gene_type is of the form [type, precision]
        dtype = gene_type

        for trial_index in range(num_trials):
            temp_val = numpy.random.uniform(low=min_val,
                                            high=max_val,
                                            size=1)[0]

            # If mutation is by replacement, do not add the current gene value into the list.
            # This is to avoid replacing the value by itself again. We are doing nothing in this case.
            if mutation_by_replacement:
                pass
            else:
                temp_val = temp_val + solution[gene_index]

            if not dtype[1] is None:
                # Precision is available and we have to round the number.
                # Convert the data type and round the number.
                temp_val = numpy.round(dtype[0](temp_val),
                                       dtype[1])
            else:
                # There is no precision and rounding the number is not needed. The type is [type, None]
                # Just convert the data type.
                temp_val = dtype[0](temp_val)

            if temp_val in solution and trial_index == (num_trials - 1):
                # If there are no values, then keep the current gene value.
                if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but cannot find a value to prevent duplicates.")
                selected_value = solution[gene_index]
            elif temp_val in solution:
                # Keep trying in the other remaining trials.
                continue
            else:
                # Unique gene value found.
                selected_value = temp_val
                break

        return selected_value

    def unique_genes_by_space(self, 
                              new_solution, 
                              gene_type, 
                              not_unique_indices, 
                              num_trials=10, 
                              build_initial_pop=False):

        """
        Iterates through all duplicate genes to find unique values from their gene spaces and resolve duplicates.
        For each duplicate gene, a call is made to the `unique_gene_by_space()` function.

        Args:
            new_solution (list): A solution containing genes with duplicate values.
            gene_type (type): The data type of the all the genes (e.g., int, float).
            not_unique_indices (list): The indices of genes with duplicate values.
            num_trials (int): The maximum number of attempts to resolve duplicates for each gene. Only works for floating-point numbers.

        Returns:
            tuple:
                list: The updated solution after attempting to resolve all duplicates. If no duplicates are resolved, the solution remains unchanged.
                list: The indices of genes that still have duplicate values.
                int: The number of duplicates that could not be resolved.
        """

        num_unsolved_duplicates = 0
        for duplicate_index in not_unique_indices:
            temp_val = self.unique_gene_by_space(solution=new_solution, 
                                                 gene_idx=duplicate_index, 
                                                 gene_type=gene_type,
                                                 build_initial_pop=build_initial_pop,
                                                 num_trials=num_trials)

            if temp_val in new_solution:
                # self.logger.info("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                num_unsolved_duplicates = num_unsolved_duplicates + 1
                if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {new_solution[duplicate_index]}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
            else:
                new_solution[duplicate_index] = temp_val
    
        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
        not_unique_indices = set(range(len(new_solution))) - set(unique_gene_indices)
        # self.logger.info("not_unique_indices INSIDE", not_unique_indices)        

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_gene_by_space(self, 
                             solution, 
                             gene_idx, 
                             gene_type, 
                             build_initial_pop=False,
                             num_trials=10):
    
            """
            Returns a unique value for a specific gene based on its value space to resolve duplicates.

            Args:
                solution (list): A solution containing genes with duplicate values.
                gene_idx (int): The index of the gene that has a duplicate value.
                gene_type (type): The data type of the gene (e.g., int, float).
                num_trials (int): The maximum number of attempts to resolve duplicates for each gene. Only works for floating-point numbers.

            Returns:
                Any: A unique value for the gene, if one exists; otherwise, the original gene value.            """

            if self.gene_space_nested:
                if type(self.gene_space[gene_idx]) in [numpy.ndarray, list, tuple]:
                    # Return the current gene space from the 'gene_space' attribute.
                    curr_gene_space = list(self.gene_space[gene_idx]).copy()
                else:
                    # Return the entire gene space from the 'gene_space' attribute.
                    # curr_gene_space = list(self.gene_space[gene_idx]).copy()
                    curr_gene_space = self.gene_space[gene_idx]

                # If the gene space has only a single value, use it as the new gene value.
                if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                    value_from_space = curr_gene_space
                    # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                elif curr_gene_space is None:
                    if self.gene_type_single == True:
                        dtype = gene_type
                    else:
                        dtype = gene_type[gene_idx]

                    if dtype[0] in pygad.GA.supported_int_types:
                        if build_initial_pop == True:
                            # If we are building the initial population, then use the range of the initial population.
                            min_val = self.init_range_low
                            max_val = self.init_range_high
                        else:
                            # If we are NOT building the initial population, then use the range of the random mutation.
                            min_val = self.random_mutation_min_val
                            max_val = self.random_mutation_max_val

                        value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                           gene_index=gene_idx, 
                                                                           min_val=min_val, 
                                                                           max_val=max_val, 
                                                                           mutation_by_replacement=True,
                                                                           gene_type=dtype)
                    else:
                        if build_initial_pop == True:
                            low = self.init_range_low
                            high = self.init_range_high
                        else:
                            low = self.random_mutation_min_val
                            high = self.random_mutation_max_val

                        """
                        value_from_space = numpy.random.uniform(low=low,
                                                                high=high,
                                                                size=1)[0]
                        """

                        value_from_space = self.unique_float_gene_from_range(solution=solution, 
                                                                             gene_index=gene_idx, 
                                                                             min_val=low, 
                                                                             max_val=high, 
                                                                             mutation_by_replacement=True, 
                                                                             gene_type=dtype, 
                                                                             num_trials=num_trials)


                elif type(curr_gene_space) is dict:
                    if self.gene_type_single == True:
                        dtype = gene_type
                    else:
                        dtype = gene_type[gene_idx]

                    # Use index 0 to return the type from the list (e.g. [int, None] or [float, 2]).
                    if dtype[0] in pygad.GA.supported_int_types:
                        if 'step' in curr_gene_space.keys():
                            step = curr_gene_space['step']
                        else:
                            step = None

                        value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                           gene_index=gene_idx, 
                                                                           min_val=curr_gene_space['low'], 
                                                                           max_val=curr_gene_space['high'], 
                                                                           step=step,
                                                                           mutation_by_replacement=True, 
                                                                           gene_type=dtype)
                    else:
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
                    # Selecting a value randomly based on the current gene's space in the 'gene_space' attribute.
                    # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                    if len(curr_gene_space) == 1:
                        value_from_space = curr_gene_space[0]
                        if not self.suppress_warnings: warnings.warn(f"You set 'allow_duplicate_genes=False' but the space of the gene with index {gene_idx} has only a single value. Thus, duplicates are possible.")
                    # If the gene space has more than 1 value, then select a new one that is different from the current value.
                    else:
                        values_to_select_from = list(set(curr_gene_space) - set(solution))
    
                        if len(values_to_select_from) == 0:
                            # DEEP-DUPLICATE-REMOVAL-NEEDED
                            # Search by this phrase to find where deep duplicates removal should be applied.

                            # Reaching this block means there is no value in the gene space of this gene to solve the duplicates.
                            # To solve the duplicate between the 2 genes, the solution is to change the value of a third gene that makes a room to solve the duplicate.

                            if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")

                            solution2 = self.solve_duplicates_deeply(solution)
                            if solution2 is None:
                                # Cannot solve duplicates. At the moment, we are changing the value of a third gene to solve the duplicates between 2 genes.
                                # Maybe a 4th, 5th, 6th, or even more genes need to be changed to solve the duplicates.
                                pass
                            else:
                                solution = solution2
                            value_from_space = solution[gene_idx]

                        else:
                            value_from_space = random.choice(values_to_select_from)
            else:
                # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
                if type(self.gene_space) is dict:
                    if self.gene_type_single == True:
                        dtype = gene_type
                    else:
                        dtype = gene_type[gene_idx]

                    if dtype[0] in pygad.GA.supported_int_types:
                        if 'step' in self.gene_space.keys():
                            step = self.gene_space['step']
                        else:
                            step = None

                        value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                           gene_index=gene_idx, 
                                                                           min_val=self.gene_space['low'], 
                                                                           max_val=self.gene_space['high'], 
                                                                           step=step,
                                                                           mutation_by_replacement=True, 
                                                                           gene_type=dtype)
                    else:
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
                    # Remove all the genes in the current solution from the gene_space.
                    # This only leaves the unique values that could be selected for the gene.
                    values_to_select_from = list(set(self.gene_space) - set(solution))
    
                    if len(values_to_select_from) == 0:
                        if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but the gene space does not have enough values to prevent duplicates.")
                        value_from_space = solution[gene_idx]
                    else:
                        value_from_space = random.choice(values_to_select_from)

            if value_from_space is None:
                if build_initial_pop == True:
                    low = self.init_range_low
                    high = self.init_range_high
                else:
                    low = self.random_mutation_min_val
                    high = self.random_mutation_max_val

                value_from_space = numpy.random.uniform(low=low,
                                                        high=high,
                                                        size=1)[0]

            # Similar to the round_genes() method in the pygad module,
            # Create a round_gene() method to round a single gene.
            if self.gene_type_single == True:
                dtype = gene_type
            else:
                dtype = gene_type[gene_idx]

            if not dtype[1] is None:
                value_from_space = numpy.round(dtype[0](value_from_space),
                                               dtype[1])
            else:
                value_from_space = dtype[0](value_from_space)

            return value_from_space

    def find_two_duplicates(self, 
                            solution,
                            gene_space_unpacked):
        """
        Identifies the first occurrence of a duplicate gene in the solution.

        Returns:
            tuple:
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
                          num_values_from_inf_range=100):
        """
        Unpacks the gene space for selecting a value to resolve duplicates by converting ranges into lists of values.

        Args:
            range_min (float or int): The minimum value of the range.
            range_max (float or int): The maximum value of the range.
            num_values_from_inf_range (int): The number of values to generate for an infinite range of float values using `numpy.linspace()`.

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
                                                         num=num_values_from_inf_range,
                                                         endpoint=False)

            if self.gene_type_single == True:
                # Change the data type.
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
                    if self.gene_type_single == True:
                        dtype = self.gene_type
                    else:
                        dtype = self.gene_type[space_idx]

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
                                                                            num=num_values_from_inf_range,
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
    
                if self.gene_type_single == True:
                    dtype = self.gene_type
                else:
                    dtype = self.gene_type[space_idx]

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
            gene_idx1 (int): The index of the first gene involved in the duplication.
            gene_idx2 (int): The index of the second gene involved in the duplication.
            assist_gene_idx (int): The index of the third gene used to assist in resolving the duplication.

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

        # The remove() function only removes the first occurrence of the value.
        # Do not use it.
        # gene_other_values.remove(duplicate_value)
    
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
