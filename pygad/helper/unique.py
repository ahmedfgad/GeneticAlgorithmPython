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
            Solves the duplicates in a solution by randomly selecting new values for the duplicating genes.

            solution: A solution with duplicate values.
            min_val: Minimum value of the range to sample a number randomly.
            max_val: Maximum value of the range to sample a number randomly.
            mutation_by_replacement: Identical to the self.mutation_by_replacement attribute.
            gene_type: Exactly the same as the self.gene_type attribute.
            num_trials: Maximum number of trials to change the gene value to solve the duplicates.

            Returns:
                new_solution: Solution after trying to solve its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
                not_unique_indices: Indices of the genes with duplicate values.
                num_unsolved_duplicates: Number of unsolved duplicates.
            """
    
            new_solution = solution.copy()
    
            _, unique_gene_indices = numpy.unique(solution, return_index=True)
            not_unique_indices = set(range(len(solution))) - set(unique_gene_indices)
    
            num_unsolved_duplicates = 0
            if len(not_unique_indices) > 0:
                for duplicate_index in not_unique_indices:
                    for trial_index in range(num_trials):
                        if self.gene_type_single == True:
                            if gene_type[0] in pygad.GA.supported_int_types:
                                temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                           gene_index=duplicate_index, 
                                                                           min_val=min_val, 
                                                                           max_val=max_val, 
                                                                           mutation_by_replacement=mutation_by_replacement, 
                                                                           gene_type=gene_type)
                            else:
                                temp_val = numpy.random.uniform(low=min_val,
                                                                high=max_val,
                                                                size=1)[0]
                                if mutation_by_replacement:
                                    pass
                                else:
                                    temp_val = new_solution[duplicate_index] + temp_val
                        else:
                            if gene_type[duplicate_index][0] in pygad.GA.supported_int_types:
                                temp_val = self.unique_int_gene_from_range(solution=new_solution, 
                                                                           gene_index=duplicate_index, 
                                                                           min_val=min_val, 
                                                                           max_val=max_val, 
                                                                           mutation_by_replacement=mutation_by_replacement, 
                                                                           gene_type=gene_type)
                            else:
                                temp_val = numpy.random.uniform(low=min_val,
                                                                high=max_val,
                                                                size=1)[0]
                                if mutation_by_replacement:
                                    pass
                                else:
                                    temp_val = new_solution[duplicate_index] + temp_val

                        # Similar to the round_genes() method in the pygad module,
                        # Create a round_gene() method to round a single gene.
                        if self.gene_type_single == True:
                            if not gene_type[1] is None:
                                temp_val = numpy.round(gene_type[0](temp_val),
                                                       gene_type[1])
                            else:
                                temp_val = gene_type[0](temp_val)
                        else:
                            if not gene_type[duplicate_index][1] is None:
                                temp_val = numpy.round(gene_type[duplicate_index][0](temp_val),
                                                       gene_type[duplicate_index][1])
                            else:
                                temp_val = gene_type[duplicate_index][0](temp_val)
    
                        if temp_val in new_solution and trial_index == (num_trials - 1):
                            num_unsolved_duplicates = num_unsolved_duplicates + 1
                            if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {solution[duplicate_index]}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
                        elif temp_val in new_solution:
                            continue
                        else:
                            new_solution[duplicate_index] = temp_val
                            break
    
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
            Solves the duplicates in a solution by selecting values for the duplicating genes from the gene space.
    
            solution: A solution with duplicate values.
            gene_type: Exactly the same as the self.gene_type attribute.
            num_trials: Maximum number of trials to change the gene value to solve the duplicates.
    
            Returns:
                new_solution: Solution after trying to solve its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
                not_unique_indices: Indices of the genes with duplicate values.
                num_unsolved_duplicates: Number of unsolved duplicates.
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
                                   step=None):

            """
            Finds a unique integer value for the gene.
    
            solution: A solution with duplicate values.
            gene_index: Index of the gene to find a unique value.
            min_val: Minimum value of the range to sample a number randomly.
            max_val: Maximum value of the range to sample a number randomly.
            mutation_by_replacement: Identical to the self.mutation_by_replacement attribute.
            gene_type: Exactly the same as the self.gene_type attribute.
    
            Returns:
                selected_value: The new value of the gene. It may be identical to the original gene value in case there are no possible unique values for the gene.
            """

            if self.gene_type_single == True:
                if step is None:
                    # all_gene_values = numpy.arange(min_val, 
                    #                                max_val, 
                    #                                dtype=gene_type[0])
                    all_gene_values = numpy.asarray(numpy.arange(min_val, max_val), 
                                                    dtype=gene_type[0])
                else:
                    # For non-integer steps, the numpy.arange() function returns zeros if the dtype parameter is set to an integer data type. So, this returns zeros if step is non-integer and dtype is set to an int data type: numpy.arange(min_val, max_val, step, dtype=gene_type[0])
                    # To solve this issue, the data type casting will not be handled inside numpy.arange(). The range is generated by numpy.arange() and then the data type is converted using the numpy.asarray() function.
                    all_gene_values = numpy.asarray(numpy.arange(min_val, 
                                                                 max_val, 
                                                                 step), 
                                                    dtype=gene_type[0])
            else:
                if step is None:
                    # all_gene_values = numpy.arange(min_val, 
                    #                                max_val, 
                    #                                dtype=gene_type[gene_index][0])
                    all_gene_values = numpy.asarray(numpy.arange(min_val, 
                                                                 max_val), 
                                                    dtype=gene_type[gene_index][0])
                else:
                    all_gene_values = numpy.asarray(numpy.arange(min_val, 
                                                                 max_val, 
                                                                 step), 
                                                    dtype=gene_type[gene_index][0])
    
            if mutation_by_replacement:
                pass
            else:
                all_gene_values = all_gene_values + solution[gene_index]

            # TODO: The gene data type is converted twine. One above and one here.
            if self.gene_type_single == True:
                # Note that we already know that the data type is integer.
                all_gene_values = numpy.asarray(all_gene_values, 
                                                dtype=gene_type[0])
            else:
                # Note that we already know that the data type is integer.
                all_gene_values = numpy.asarray(all_gene_values, 
                                                gene_type[gene_index][0])

            values_to_select_from = list(set(list(all_gene_values)) - set(solution))
    
            if len(values_to_select_from) == 0:
                # If there is no values, then keep the current gene value.
                if not self.suppress_warnings: warnings.warn("You set 'allow_duplicate_genes=False' but there is no enough values to prevent duplicates.")
                selected_value = solution[gene_index]
            else:
                selected_value = random.choice(values_to_select_from)
    
            return selected_value

    def unique_genes_by_space(self, 
                              new_solution, 
                              gene_type, 
                              not_unique_indices, 
                              num_trials=10, 
                              build_initial_pop=False):

        """
        Loops through all the duplicating genes to find unique values that from their gene spaces to solve the duplicates.
        For each duplicating gene, a call to the unique_gene_by_space() function is made.
    
        new_solution: A solution with duplicate values.
        gene_type: Exactly the same as the self.gene_type attribute.
        not_unique_indices: Indices with duplicating values.
        num_trials: Maximum number of trials to change the gene value to solve the duplicates.
    
        Returns:
            new_solution: Solution after trying to solve all of its duplicates. If no duplicates solved, then it is identical to the passed solution parameter.
            not_unique_indices: Indices of the genes with duplicate values.
            num_unsolved_duplicates: Number of unsolved duplicates.
        """

        num_unsolved_duplicates = 0
        for duplicate_index in not_unique_indices:
            for trial_index in range(num_trials):
                temp_val = self.unique_gene_by_space(solution=new_solution, 
                                                     gene_idx=duplicate_index, 
                                                     gene_type=gene_type,
                                                     build_initial_pop=build_initial_pop)

                if temp_val in new_solution and trial_index == (num_trials - 1):
                    # self.logger.info("temp_val, duplicate_index", temp_val, duplicate_index, new_solution)
                    num_unsolved_duplicates = num_unsolved_duplicates + 1
                    if not self.suppress_warnings: warnings.warn(f"Failed to find a unique value for gene with index {duplicate_index} whose value is {new_solution[duplicate_index]}. Consider adding more values in the gene space or use a wider range for initial population or random mutation.")
                elif temp_val in new_solution:
                    continue
                else:
                    new_solution[duplicate_index] = temp_val
                    # self.logger.info("SOLVED", duplicate_index)
                    break
    
        # Update the list of duplicate indices after each iteration.
        _, unique_gene_indices = numpy.unique(new_solution, return_index=True)
        not_unique_indices = set(range(len(new_solution))) - set(unique_gene_indices)
        # self.logger.info("not_unique_indices INSIDE", not_unique_indices)        

        return new_solution, not_unique_indices, num_unsolved_duplicates

    def unique_gene_by_space(self, 
                             solution, 
                             gene_idx, 
                             gene_type, 
                             build_initial_pop=False):
    
            """
            Returns a unique gene value for a single gene based on its value space to solve the duplicates.
    
            solution: A solution with duplicate values.
            gene_idx: The index of the gene that duplicates its value with another gene.
            gene_type: Exactly the same as the self.gene_type attribute.
    
            Returns:
                A unique value, if exists, for the gene.
            """

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
                        if gene_type[0] in pygad.GA.supported_int_types:
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
                                                                               gene_type=gene_type)
                        else:
                            if build_initial_pop == True:
                                low = self.init_range_low
                                high = self.init_range_high
                            else:
                                low = self.random_mutation_min_val
                                high = self.random_mutation_max_val

                            value_from_space = numpy.random.uniform(low=low,
                                                                    high=high,
                                                                    size=1)[0]
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space
                    else:
                        if gene_type[gene_idx][0] in pygad.GA.supported_int_types:
                            if build_initial_pop == True:
                                min_val = self.init_range_low
                                max_val = self.init_range_high
                            else:
                                min_val = self.random_mutation_min_val
                                max_val = self.random_mutation_max_val

                            value_from_space = self.unique_int_gene_from_range(solution=solution, 
                                                                               gene_index=gene_idx, 
                                                                               min_val=min_val, 
                                                                               max_val=max_val, 
                                                                               mutation_by_replacement=True,
                                                                               gene_type=gene_type)
                        else:
                            if build_initial_pop == True:
                                low = self.init_range_low
                                high = self.init_range_high
                            else:
                                low = self.random_mutation_min_val
                                high = self.random_mutation_max_val

                            value_from_space = numpy.random.uniform(low=low,
                                                                    high=high,
                                                                    size=1)[0]
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space
    
                elif type(curr_gene_space) is dict:
                    if self.gene_type_single == True:
                        if gene_type[0] in pygad.GA.supported_int_types:
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
                                                                               gene_type=gene_type)
                        else:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)[0]
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space
                    else:
                        # Use index 0 to return the type from the list (e.g. [int, None] or [float, 2]).
                        if gene_type[gene_idx][0] in pygad.GA.supported_int_types:
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
                                                                               gene_type=gene_type)
                        else:
                            if 'step' in curr_gene_space.keys():
                                value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                                    stop=curr_gene_space['high'],
                                                                                    step=curr_gene_space['step']),
                                                                       size=1)
                            else:
                                value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                                        high=curr_gene_space['high'],
                                                                        size=1)[0]
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space

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
                        if gene_type[0] in pygad.GA.supported_int_types:
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
                                                                               gene_type=gene_type)
                        else:
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
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space
                    else:
                        if gene_type[gene_idx][0] in pygad.GA.supported_int_types:
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
                                                                               gene_type=gene_type)
                        else:
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
                            # TODO: Remove check for mutation_by_replacement when solving duplicates. Just replace the gene by the selected value from space.
                            # if self.mutation_by_replacement:
                            #     pass
                            # else:
                            #     value_from_space = solution[gene_idx] + value_from_space
    
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
                if not gene_type[1] is None:
                    value_from_space = numpy.round(gene_type[0](value_from_space),
                                                   gene_type[1])
                else:
                    value_from_space = gene_type[0](value_from_space)
            else:
                if not gene_type[gene_idx][1] is None:
                    value_from_space = numpy.round(gene_type[gene_idx][0](value_from_space),
                                                  gene_type[gene_idx][1])
                else:
                    value_from_space = gene_type[gene_idx][0](value_from_space)

            return value_from_space

    def find_two_duplicates(self, 
                            solution,
                            gene_space_unpacked):
        """
        Returns the first occurrence of duplicate genes.
        It returns:
            The index of a gene with a duplicate value.
            The value of the gene.
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
        Unpack the gene_space for the purpose of selecting a value that solves the duplicates.
        This is by replacing each range by a list of values.
        It accepts:
            range_min: The range minimum value.
            range_min: The range maximum value.
            num_values_from_inf_range: For infinite range of float values, a fixed number of values equal to num_values_from_inf_range is selected using the numpy.linspace() function.
        It returns the unpacked gene space.
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
                    if self.gene_type_single == True: # self.gene_type_single
                        if self.gene_type[0] in pygad.GA.supported_int_types:
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
                    else:
                        if self.gene_type[space_idx][0] in pygad.GA.supported_int_types:
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
    
                if self.gene_type_single == True: # self.gene_type_single
                    # Change the data type.
                    gene_space_unpacked[space_idx] = numpy.array(gene_space_unpacked[space_idx],
                                                                 dtype=self.gene_type[0])
                    if not self.gene_type[1] is None:
                        # Round the values for float (non-int) data types.
                        gene_space_unpacked[space_idx] = numpy.round(gene_space_unpacked[space_idx],
                                                                     self.gene_type[1])
                else:
                    # Change the data type.
                    gene_space_unpacked[space_idx] = numpy.array(gene_space_unpacked[space_idx],
                                                                 self.gene_type[space_idx][0])
                    if not self.gene_type[space_idx][1] is None:
                        # Round the values for float (non-int) data types.
                        gene_space_unpacked[space_idx] = numpy.round(gene_space_unpacked[space_idx],
                                                                     self.gene_type[space_idx][1])

        return gene_space_unpacked

    def solve_duplicates_deeply(self,
                                solution):
        """
        Sometimes it is impossible to solve the duplicate genes by simply selecting another value for either genes.
        This function solve the duplicates between 2 genes by searching for a third gene that can make assist in the solution.
        It returns:
            The solution after solving the duplicates or the None if duplicates cannot be solved.
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
