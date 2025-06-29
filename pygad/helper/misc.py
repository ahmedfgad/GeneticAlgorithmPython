"""
The pygad.helper.helper module has some generic helper methods.
"""

import numpy
import warnings
import random
import pygad

class Helper:

    def get_random_mutation_range(self, gene_index):

        """
        Returns the minimum and maximum values of the mutation range.
        It accepts a single parameter:
            -gene_index: The index of the gene to get its range. Only used if the gene has a specific mutation range
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

    def generate_gene_random_value(self,
                                   range_min, 
                                   range_max, 
                                   gene_value,
                                   gene_idx, 
                                   mutation_by_replacement,
                                   num_values=1):
        """
        Randomly generate one or more values for the gene.
        It accepts:
            -range_min: The minimum value in the range from which a value is selected.
            -range_max: The maximum value in the range from which a value is selected.
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -num_values: The number of random valus to generate. It tries to generate a number of values up to a maximum of num_values. But it is not always guranteed because the total number of values might not be enough or the random generator creates duplicate random values.
        If num_values=1, it returns a single numeric value. If num_values>1, it returns an array with number of values equal to num_values.
        """

        # Generating a random value.
        random_value = numpy.asarray(numpy.random.uniform(low=range_min, 
                                                          high=range_max, 
                                                          size=num_values),
                                     dtype=object)

        # Change the random mutation value data type.
        for idx, val in enumerate(random_value):
            random_value[idx] = self.change_random_mutation_value_dtype(random_value[idx], 
                                                                        gene_idx, 
                                                                        gene_value,
                                                                        mutation_by_replacement=mutation_by_replacement)

            # Round the gene.
            random_value[idx] = self.round_random_mutation_value(random_value[idx], gene_idx)

        # Rounding different values could return the same value multiple times.
        # For example, 2.8 and 2.7 will be 3.0.
        # Use the unique() function to avoid any duplicates.
        random_value = numpy.unique(random_value)

        if num_values == 1:
            random_value = random_value[0]

        return random_value

    def get_valid_gene_constraint_values(self,
                                         range_min,
                                         range_max,
                                         gene_value,
                                         gene_idx,
                                         mutation_by_replacement,
                                         solution,
                                         num_values=100):
        """
        Randomly generate values for the gene that satisfy the constraint.
        It accepts:
            -range_min: The minimum value in the range from which a value is selected.
            -range_max: The maximum value in the range from which a value is selected.
            -gene_value: The original gene value before applying mutation.
            -gene_idx: The index of the gene in the solution.
            -mutation_by_replacement: A flag indicating whether mutation by replacement is enabled or not. The reason is to make this helper method usable while generating the initial population. In this case, mutation_by_replacement does not matter and should be considered False.
            -solution: The solution in which the gene exists.
            -num_values: The number of random valus to generate. It tries to generate a number of values up to a maximum of num_values. But it is not always guranteed because the total number of values might not be enough or the random generator creates duplicate random values.
        If num_values=1, it returns a single numeric value. If num_values>1, it returns an array with number of values equal to num_values.
        """
        random_values = self.generate_gene_random_value(range_min=range_min, 
                                                        range_max=range_max, 
                                                        gene_value=gene_value,
                                                        gene_idx=gene_idx, 
                                                        mutation_by_replacement=mutation_by_replacement,
                                                        num_values=num_values)
        random_values_filtered = self.mutation_filter_values_by_constraint(random_values=random_values,
                                                                           solution=solution,
                                                                           gene_idx=gene_idx)
        return random_values_filtered
