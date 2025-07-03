"""
The pygad.helper.helper module has some generic helper methods.
"""

import numpy
import warnings
import random
import pygad

class Helper:

    def change_gene_value_dtype(self,
                                random_value,
                                gene_index,
                                gene_value,
                                mutation_by_replacement):
        """
        Change the data type of the random value used to apply mutation.
        It accepts 2 parameters:
            -random_value: The random value to change its data type.
            -gene_index: The index of the target gene.
            -gene_value: The gene value before mutation. Only used if mutation_by_replacement=False and gene_type_single=False.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
        It returns the new value after changing the data type.
        """

        # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
        if mutation_by_replacement:
            if self.gene_type_single == True:
                random_value = self.gene_type[0](random_value)
            else:
                random_value = self.gene_type[gene_index][0](random_value)
                if type(random_value) is numpy.ndarray:
                    random_value = random_value[0]
        # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
        else:
            if self.gene_type_single == True:
                random_value = self.gene_type[0](gene_value + random_value)
            else:
                random_value = self.gene_type[gene_index][0](gene_value + random_value)
                if type(random_value) is numpy.ndarray:
                    random_value = random_value[0]
        return random_value

    def round_gene_value(self, random_value, gene_index):
        """
        Round the random value used to apply mutation.
        It accepts 2 parameters:
            -random_value: The random value to round its value.
            -gene_index: The index of the target gene. Only used if nested gene_type is used.
        It returns the new value after being rounded.
        """

        # Round the gene
        if self.gene_type_single == True:
            if not self.gene_type[1] is None:
                random_value = numpy.round(random_value, self.gene_type[1])
        else:
            if not self.gene_type[gene_index][1] is None:
                random_value = numpy.round(random_value, self.gene_type[gene_index][1])
        return random_value

    def filter_gene_values_by_constraint(self,
                                         values,
                                         solution,
                                         gene_idx):

        """
        Filter the random values generated for mutation based on whether they meet the gene constraint in the gene_constraint parameter.
        It accepts:
            -values: The values to filter.
            -solution: The solution containing the target gene.
            -gene_idx: The index of the gene in the solution.
        It returns None if no values satisfy the constraint. Otherwise, an array of values that satisfy the constraint is returned.
        """

        # A list of the indices where the random values satisfy the constraint.
        filtered_values_indices = []
        # A temporary solution to avoid changing the original solution.
        solution_tmp = solution.copy()
        # Loop through the random values to filter the ones satisfying the constraint.
        for value_idx, value in enumerate(values):
            solution_tmp[gene_idx] = value
            # Check if the constraint is satisfied.
            if self.gene_constraint[gene_idx](solution_tmp):
                # The current value satisfies the constraint.
                filtered_values_indices.append(value_idx)

        # After going through all the values, check if any value satisfies the constraint.
        if len(filtered_values_indices) > 0:
            # At least one value was found that meets the gene constraint.
            pass
        else:
            # No value found for the current gene that satisfies the constraint.
            if not self.suppress_warnings:
                warnings.warn(f"No value found for the gene at index {gene_idx} that satisfies its gene constraint.")
            return None

        filtered_values = values[filtered_values_indices]

        return filtered_values

    def get_gene_dtype(self, gene_index):

        """
        Returns the data type of the gene by its index.
        It accepts a single parameter:
            -gene_index: The index of the gene to get its data type. Only used if each gene has its own data type.
        It returns the data type of the gene.
        """

        if self.gene_type_single == True:
            dtype = self.gene_type
        else:
            dtype = self.gene_type[gene_index]
        return dtype

    def get_random_mutation_range(self, gene_index):

        """
        Returns the minimum and maximum values of the mutation range.
        It accepts a single parameter:
            -gene_index: The index of the gene to get its range. Only used if the gene has a specific mutation range.
        It returns the minimum and maximum values of the gene mutation range.
        """

        # We can use either random_mutation_min_val or random_mutation_max_val.
        if type(self.random_mutation_min_val) in self.supported_int_float_types:
            range_min = self.random_mutation_min_val
            range_max = self.random_mutation_max_val
        else:
            range_min = self.random_mutation_min_val[gene_index]
            range_max = self.random_mutation_max_val[gene_index]
        return range_min, range_max

    def get_initial_population_range(self, gene_index):

        """
        Returns the minimum and maximum values of the initial population range.
        It accepts a single parameter:
            -gene_index: The index of the gene to get its range. Only used if the gene has a specific range
        It returns the minimum and maximum values of the gene initial population range.
        """

        # We can use either init_range_low or init_range_high.
        if type(self.init_range_low) in self.supported_int_float_types:
            range_min = self.init_range_low
            range_max = self.init_range_high
        else:
            range_min = self.init_range_low[gene_index]
            range_max = self.init_range_high[gene_index]
        return range_min, range_max

    def generate_gene_value_from_space(self,
                                       gene_value,
                                       gene_idx,
                                       mutation_by_replacement,
                                       sample_size=1):
        """
        Generate/select one or more values for the gene from the gene space.
        It accepts:
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -sample_size: The number of random values to generate. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values. For int data types, it could be None to keep all the values. For float data types, a None value returns only a single value.
            -step (int, optional): The step size for generating candidate values. Defaults to 1. Only used with genes of an integer data type.

        It returns,
            -A single numeric value if sample_size=1. Or
            -An array with number of maximum number of values equal to sample_size if sample_size>1.
        """

        range_min, range_max = self.get_random_mutation_range(gene_idx)

        if self.gene_space_nested:
            # Returning the current gene space from the 'gene_space' attribute.
            # It is used to determine the way of selecting the next gene value:
            # 1) List/NumPy Array: Whether it has only one value, multiple values, or one of its values is None.
            # 2) Fixed Numeric Value
            # 3) None
            # 4) Dict: Whether the dict has the key `step` or not.
            if type(self.gene_space[gene_idx]) in [numpy.ndarray, list]:
                # Get the gene space from the `gene_space_unpacked` property because it undergoes data type change and rounded.
                curr_gene_space = self.gene_space_unpacked[gene_idx].copy()
            elif type(self.gene_space[gene_idx]) in pygad.GA.supported_int_float_types:
                # Get the gene space from the `gene_space_unpacked` property because it undergoes data type change and rounded.
                curr_gene_space = self.gene_space_unpacked[gene_idx]
            else:
                curr_gene_space = self.gene_space[gene_idx]

            if type(curr_gene_space) in pygad.GA.supported_int_float_types:
                # If the gene space is simply a single numeric value (e.g. 5), use it as the new gene value.
                value_from_space = curr_gene_space
            elif curr_gene_space is None:
                # If the gene space is None, apply mutation by adding a random value between the range defined by the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
                rand_val = numpy.random.uniform(low=range_min,
                                                high=range_max,
                                                size=sample_size)
                if mutation_by_replacement:
                    value_from_space = rand_val
                else:
                    value_from_space = gene_value + rand_val
            elif type(curr_gene_space) is dict:
                # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                # The gene's space of type dict specifies the lower and upper limits of a gene.
                if 'step' in curr_gene_space.keys():
                    # When the `size` parameter is used, the numpy.random.choice() and numpy.random.uniform() functions return a NumPy array as the output even if the array has a single value (i.e. size=1).
                    # We have to return the output at index 0 to force a numeric value to be returned not an object of type numpy.ndarray.
                    # If numpy.ndarray is returned, then it will cause an issue later while using the set() function.
                    # Randomly select a value from a discrete range.
                    value_from_space = numpy.random.choice(numpy.arange(start=curr_gene_space['low'],
                                                                        stop=curr_gene_space['high'],
                                                                        step=curr_gene_space['step']),
                                                           size=sample_size)
                else:
                    value_from_space = numpy.random.uniform(low=curr_gene_space['low'],
                                                            high=curr_gene_space['high'],
                                                            size=sample_size)
            else:
                # Selecting a value randomly from the current gene's space in the 'gene_space' attribute.
                # If the gene space has only 1 value, then select it. The old and new values of the gene are identical.
                if len(curr_gene_space) == 1:
                    value_from_space = curr_gene_space
                else:
                    # If the gene space has more than 1 value, then select a new one that is different from the current value.
                    # To avoid selecting the current gene value again, remove it from the current gene space and do the selection.
                    value_from_space = list(set(curr_gene_space) - set([gene_value]))

                    """
                    if len(values_to_select_from) == 0:
                        # After removing the current gene value from the space, there are no more values.
                        # Then keep the current gene value.
                        value_from_space = gene_value
                    else:
                        value_from_space = random.choice(values_to_select_from)
                    """
        else:
            # Selecting a value randomly from the global gene space in the 'gene_space' attribute.
            # The gene's space of type dict specifies the lower and upper limits of a gene.
            if type(self.gene_space) is dict:
                # When the gene_space is assigned a dict object, then it specifies the lower and upper limits of all genes in the space.
                if 'step' in self.gene_space.keys():
                    value_from_space = numpy.random.choice(numpy.arange(start=self.gene_space['low'],
                                                                        stop=self.gene_space['high'],
                                                                        step=self.gene_space['step']),
                                                           size=sample_size)
                else:
                    value_from_space = numpy.random.uniform(low=self.gene_space['low'],
                                                            high=self.gene_space['high'],
                                                            size=sample_size)
            else:
                # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                # To avoid selecting the current gene value again, remove it from the current gene space and do the selection.
                value_from_space = list(set(self.gene_space) - set([gene_value]))

                """
                if len(values_to_select_from) == 0:
                    # After removing the current gene value from the space, there are no more values.
                    # Then keep the current gene value.
                    value_from_space = gene_value
                else:
                    value_from_space = random.choice(values_to_select_from)
                """

        if len(value_from_space) == 0:
            # After removing the current gene value from the space, there are no more values.
            # Then keep the current gene value.
            value_from_space = gene_value
        elif sample_size == 1:
            value_from_space = random.choice(value_from_space)

            # The gene space might be [None, 1, 7].
            # It might happen that the value None is selected.
            # In this case, generate a random value out of the mutation range.
            if value_from_space is None:
                # TODO: Return index 0.
                # TODO: Check if this if statement is necessary.
                value_from_space = numpy.random.uniform(low=range_min,
                                                        high=range_max,
                                                        size=sample_size)
        else:
            value_from_space = numpy.array(value_from_space)

        return value_from_space

    def generate_gene_value_randomly(self,
                                     range_min,
                                     range_max,
                                     gene_value,
                                     gene_idx,
                                     mutation_by_replacement,
                                     sample_size=1,
                                     step=1):
        """
        Randomly generate one or more values for the gene.
        It accepts:
            -range_min: The minimum value in the range from which a value is selected.
            -range_max: The maximum value in the range from which a value is selected.
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -sample_size: The number of random values to generate. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values. For int data types, it could be None to keep all the values. For float data types, a None value returns only a single value.
            -step (int, optional): The step size for generating candidate values. Defaults to 1. Only used with genes of an integer data type.

        It returns,
            -A single numeric value if sample_size=1. Or
            -An array with number of values equal to sample_size if sample_size>1.
        """

        gene_type = self.get_gene_dtype(gene_index=gene_idx)
        if gene_type[0] in pygad.GA.supported_int_types:
            random_value = numpy.asarray(numpy.arange(range_min, 
                                                      range_max, 
                                                      step=step), 
                                         dtype=gene_type[0])
            if sample_size is None:
                # Keep all the values.
                pass
            else:
                if sample_size >= len(random_value):
                    # Number of values is larger than or equal to the number of elements in random_value.
                    # Makes no sense to create a larger sample out of the population because it just creates redundant values.
                    pass
                else:
                    # Set replace=True to avoid selecting the same value more than once.
                    random_value = numpy.random.choice(random_value, 
                                                       size=sample_size,
                                                       replace=False)
        else:
            # Generating a random value.
            random_value = numpy.asarray(numpy.random.uniform(low=range_min, 
                                                              high=range_max, 
                                                              size=sample_size),
                                         dtype=object)

        # Change the random mutation value data type.
        for idx, val in enumerate(random_value):
            random_value[idx] = self.change_gene_value_dtype(random_value[idx],
                                                             gene_idx,
                                                             gene_value,
                                                             mutation_by_replacement=mutation_by_replacement)

            # Round the gene.
            random_value[idx] = self.round_gene_value(random_value[idx], gene_idx)

        # Rounding different values could return the same value multiple times.
        # For example, 2.8 and 2.7 will be 3.0.
        # Use the unique() function to avoid any duplicates.
        random_value = numpy.unique(random_value)

        if sample_size == 1:
            random_value = random_value[0]

        return random_value

    def generate_gene_value(self,
                            gene_value,
                            gene_idx,
                            mutation_by_replacement,
                            range_min=None,
                            range_max=None,
                            sample_size=1,
                            step=1):
        """
        Generate one or more values for the gene either randomly or from the gene space. It acts as a router.
        It accepts:
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -range_min (int, optional): The minimum value in the range from which a value is selected. It must be passed for generating the gene value randomly because we cannot decide whether it is the range for the initial population (init_range_low and init_range_high) or mutation (random_mutation_min_val and random_mutation_max_val).
            -range_max (int, optional): The maximum value in the range from which a value is selected. It must be passed for generating the gene value randomly because we cannot decide whether it is the range for the initial population (init_range_low and init_range_high) or mutation (random_mutation_min_val and random_mutation_max_val).
            -sample_size: The number of random values to generate/select and return. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values. For int data types, it could be None to keep all the values. For float data types, a None value returns only a single value.
            -step (int, optional): The step size for generating candidate values. Defaults to 1. Only used with genes of an integer data type.

        It returns,
            -A single numeric value if sample_size=1. Or
            -An array with number of values equal to sample_size if sample_size>1.
        """
        if self.gene_space is None:
            output = self.generate_gene_value_randomly(range_min=range_min,
                                                       range_max=range_max,
                                                       gene_value=gene_value,
                                                       gene_idx=gene_idx,
                                                       mutation_by_replacement=mutation_by_replacement,
                                                       sample_size=sample_size,
                                                       step=step)
        else:
            output = self.generate_gene_value_from_space(gene_value=gene_value,
                                                         gene_idx=gene_idx,
                                                         mutation_by_replacement=mutation_by_replacement,
                                                         sample_size=sample_size)
        return output

    def get_valid_gene_constraint_values(self,
                                         range_min,
                                         range_max,
                                         gene_value,
                                         gene_idx,
                                         mutation_by_replacement,
                                         solution,
                                         sample_size=100,
                                         step=1):
        """
        Generate/select values for the gene that satisfy the constraint. The values could be generated randomly or from the gene space.
        The number of returned values is at its maximum equal to the sample_size parameter.
        It accepts:
            -range_min: The minimum value in the range from which a value is selected.
            -range_max: The maximum value in the range from which a value is selected.
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -solution: The solution in which the gene exists.
            -sample_size: The number of values to generate or select. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values.
            -step (int, optional): The step size for generating candidate values. Defaults to 1. Only used with genes of an integer data type.

        It returns,
            -A single numeric value if sample_size=1. Or
            -An array with number of values equal to sample_size if sample_size>1. Or
            -None if no value found that satisfies the constraint.
        """
        # Either generate the values randomly or from the gene space.
        values = self.generate_gene_value(range_min=range_min,
                                          range_max=range_max,
                                          gene_value=gene_value,
                                          gene_idx=gene_idx,
                                          mutation_by_replacement=mutation_by_replacement,
                                          sample_size=sample_size,
                                          step=step)
        # It returns None if no value found that satisfies the constraint.
        values_filtered = self.filter_gene_values_by_constraint(values=values,
                                                                solution=solution,
                                                                gene_idx=gene_idx)
        return values_filtered
