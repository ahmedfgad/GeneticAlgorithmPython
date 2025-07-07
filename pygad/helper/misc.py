"""
The pygad.helper.misc module has some generic helper methods.
"""

import numpy
import warnings
import random
import pygad

class Helper:

    def change_population_dtype_and_round(self,
                                          population):
        """
        Change the data type of the population. It works with iterables (e.g. lists or NumPy arrays) of shape 2D.
        It does not handle single numeric values or 1D arrays.

        It accepts:
            -population: The iterable to change its dtype.

        It returns the iterable with the data type changed for all genes.
        """

        population_new = numpy.array(population.copy(), dtype=object)

        # Forcing the iterable to have the data type assigned to the gene_type parameter.
        if self.gene_type_single == True:
            # Round the numbers first then change the data type.
            # This solves issues with some data types such as numpy.float32.
            if self.gene_type[1] is None:
                pass
            else:
                # This block is reached only for non-integer data types (i.e. float).
                population_new = numpy.round(numpy.array(population_new, float),
                                             self.gene_type[1])

            population_new = numpy.array(population_new,
                                         dtype=self.gene_type[0])
        else:
            population = numpy.array(population.copy())
            population_new = numpy.zeros(shape=population.shape,
                                         dtype=object)
            for gene_idx in range(population.shape[1]):
                # Round the numbers first then change the data type.
                # This solves issues with some data types such as numpy.float32.
                if self.gene_type[gene_idx][1] is None:
                    # Do not round.
                    population_new[:, gene_idx] = population[:, gene_idx]
                else:
                    # This block is reached only for non-integer data types (i.e. float).
                    population_new[:, gene_idx] = numpy.round(numpy.array(population[:, gene_idx], float),
                                                              self.gene_type[gene_idx][1])
                # Once rounding is done, change the data type.
                # population_new[:, gene_idx] = numpy.asarray(population_new[:, gene_idx], dtype=self.gene_type[gene_idx][0])
                # Use a for loop to maintain the data type of each individual gene.
                for sol_idx in range(population.shape[0]):
                    population_new[sol_idx, gene_idx] = self.gene_type[gene_idx][0](population_new[sol_idx, gene_idx])
        return population_new

    def change_gene_dtype_and_round(self,
                                    gene_index,
                                    gene_value):
        """
        Change the data type and round a single gene value or a vector of values FOR THE SAME GENE. E.g., the input could be 6 or [6, 7, 8].

        It accepts 2 parameters:
            -gene_index: The index of the target gene.
            -gene_value: The gene value.

        If gene_value has a single value, then it returns a single number with the type changed and value rounded. If gene_value is a vector, then a vector is returned after changing the data type and rounding.
        """

        if self.gene_type_single == True:
            dtype = self.gene_type[0]
            if self.gene_type[1] is None:
                # No rounding for this gene. Use the old gene value.
                round_precision = None
            else:
                round_precision = self.gene_type[1]
        else:
            dtype = self.gene_type[gene_index][0]
            if self.gene_type[gene_index][1] is None:
                # No rounding for this gene. Use the old gene value.
                round_precision = None
            else:
                round_precision = self.gene_type[gene_index][1]

        # Sometimes the values represent the gene_space when it is not nested (e.g. gene_space=range(10))
        # Copy it to avoid changing the original gene_space.
        gene_value = [gene_value].copy()

        # Round the number before changing its data type to avoid precision loss for some data types like numpy.float32.
        if round_precision is None:
            pass
        else:
            gene_value = numpy.round(gene_value, round_precision)

        gene_value_new = numpy.asarray(gene_value, dtype=dtype)
        gene_value_new = gene_value_new[0]

        return gene_value_new

    def mutation_change_gene_dtype_and_round(self,
                                             random_value,
                                             gene_index,
                                             gene_value,
                                             mutation_by_replacement):
        """
        Change the data type and round the random value used to apply mutation.

        It accepts:
            -random_value: The random value to change its data type.
            -gene_index: The index of the target gene.
            -gene_value: The gene value before mutation. Only used if mutation_by_replacement=False and gene_type_single=False.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.

        It returns the new value after changing the data type and being rounded.
        """

        if mutation_by_replacement:
            # If the mutation_by_replacement attribute is True, then the random value replaces the current gene value.
            gene_value = random_value
        else:
            # If the mutation_by_replacement attribute is False, then the random value is added to the gene value.
            gene_value = gene_value + random_value

        gene_value_new = self.change_gene_dtype_and_round(gene_index=gene_index,
                                                          gene_value=gene_value)
        return gene_value_new

    def validate_gene_constraint_callable_output(self,
                                                 selected_values,
                                                 values):
        if type(selected_values) in [list, numpy.ndarray]:
            selected_values_set = set(selected_values)
            if selected_values_set.issubset(values):
                pass
            else:
                return False
        else:
            return False

        return True

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

        if self.gene_constraint and self.gene_constraint[gene_idx]:
            pass
        else:
            raise Exception(f"Either the gene at index {gene_idx} is not assigned a callable/function or the gene_constraint itself is not used.")

        # A temporary solution to avoid changing the original solution.
        solution_tmp = solution.copy()
        filtered_values = self.gene_constraint[gene_idx](solution_tmp, values.copy())
        result = self.validate_gene_constraint_callable_output(selected_values=filtered_values,
                                                               values=values)
        if result:
            pass
        else:
            raise Exception("The output from the gene_constraint callable/function must be a list or NumPy array that is subset of the passed values (second argument).")

        # After going through all the values, check if any value satisfies the constraint.
        if len(filtered_values) > 0:
            # At least one value was found that meets the gene constraint.
            pass
        else:
            # No value found for the current gene that satisfies the constraint.
            if not self.suppress_warnings: warnings.warn(f"Failed to find a value that satisfies its gene constraint for the gene at index {gene_idx} with value {solution[gene_idx]} at generation {self.generations_completed}.")
            return None

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
                                       gene_idx,
                                       mutation_by_replacement,
                                       solution=None,
                                       gene_value=None,
                                       sample_size=1):
        """
        Generate/select one or more values for the gene from the gene space.

        It accepts:
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -solution (iterable, optional): The solution where we need to generate a gene. Needed if you are selecting a single value (sample_size=1) to select a value that respects the allow_duplicate_genes parameter instead of selecting a value randomly. If None, then the gene value is selected randomly.
            -gene_value (int, optional): The original gene value before applying mutation. Needed if you are calling this method to apply mutation. If None, then a sample is created from the gene space without being summed to the gene value.
            -sample_size (int, optional): The number of random values to generate. It tries to generate a number of values up to a maximum of sample_size. But it is not always guaranteed because the total number of values might not be enough or the random generator creates duplicate random values. For int data types, it could be None to keep all the values. For float data types, a None value returns only a single value.

        It returns,
            -A single numeric value if sample_size=1. Or
            -An array with number of maximum number of values equal to sample_size if sample_size>1.
        """

        if gene_value is None:
            # Use the initial population range.
            range_min, range_max = self.get_initial_population_range(gene_index=gene_idx)
        else:
            # Use the mutation range. 
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
                    # Change the data type and round the generated values.
                    curr_gene_space = self.change_gene_dtype_and_round(gene_index=gene_idx,
                                                                       gene_value=curr_gene_space)

                    if gene_value is None:
                        # Just generate the value(s) without being added to the gene value specially when initializing the population.
                        value_from_space = curr_gene_space
                    else:
                        # If the gene space has more than 1 value, then select a new one that is different from the current value.
                        # To avoid selecting the current gene value again, remove it from the current gene space and do the selection.
                        value_from_space = list(set(curr_gene_space) - set([gene_value]))
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
                curr_gene_space = list(self.gene_space).copy()
                for idx in range(len(curr_gene_space)):
                    if curr_gene_space[idx] is None:
                        curr_gene_space[idx] = numpy.random.uniform(low=range_min,
                                                                    high=range_max)
                curr_gene_space = self.change_gene_dtype_and_round(gene_index=gene_idx,
                                                                   gene_value=curr_gene_space)

                if gene_value is None:
                    # Just generate the value(s) without being added to the gene value specially when initializing the population.
                    value_from_space = curr_gene_space
                else:
                    # If the space type is not of type dict, then a value is randomly selected from the gene_space attribute.
                    # To avoid selecting the current gene value again, remove it from the current gene space and do the selection.
                    value_from_space = list(set(curr_gene_space) - set([gene_value]))

        if len(value_from_space) == 0:
            if gene_value is None:
                raise ValueError(f"There are no values to select from the gene_space for the gene at index {gene_idx}.")
            else:
                # After removing the current gene value from the space, there are no more values.
                # Then keep the current gene value.
                value_from_space = gene_value
                if sample_size > 1:
                    value_from_space = numpy.array([gene_value])
        elif sample_size == 1:
            if self.allow_duplicate_genes == True:
                # Select a value randomly from the current gene space.
                value_from_space = random.choice(value_from_space)
            else:
                # We must check if the selected value will respect the allow_duplicate_genes parameter.
                # Instead of selecting a value randomly, we have to select a value that will be unique if allow_duplicate_genes=False.
                # Only select a value from the current gene space that is, hopefully, unique.
                value_from_space = self.select_unique_value(gene_values=value_from_space,
                                                            solution=solution,
                                                            gene_index=gene_idx)

            # The gene space might be [None, 1, 7].
            # It might happen that the value None is selected.
            # In this case, generate a random value out of the mutation range.
            if value_from_space is None:
                value_from_space = numpy.random.uniform(low=range_min,
                                                        high=range_max,
                                                        size=sample_size)
        else:
            value_from_space = numpy.array(value_from_space)

        # Change the data type and round the generated values.
        # It has to be called here for all the missed cases.
        value_from_space = self.change_gene_dtype_and_round(gene_index=gene_idx,
                                                            gene_value=value_from_space)
        if sample_size == 1 and type(value_from_space) not in pygad.GA.supported_int_float_types:
            value_from_space = value_from_space[0]

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
            random_value[idx] = self.mutation_change_gene_dtype_and_round(random_value[idx],
                                                                          gene_idx,
                                                                          gene_value,
                                                                          mutation_by_replacement=mutation_by_replacement)

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
                            solution=None,
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
            -solution (iterable, optional): The solution where we need to generate a gene. Needed if you are selecting a single value (sample_size=1) to select a value that respects the allow_duplicate_genes parameter instead of selecting a value randomly. If None, then the gene value is selected randomly.
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
                                                         solution=solution,
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
