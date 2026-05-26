"""
The pygad.helper.misc module has some generic helper methods.
"""

import numpy
import warnings
import random
import pygad

class Helper:

    def summary(self,
                line_length=70,
                fill_character=" ",
                line_character="-",
                line_character2="=",
                columns_equal_len=False,
                print_step_parameters=True,
                print_parameters_summary=True):
        """
        Print a Keras-style summary of the PyGAD lifecycle. Each
        configured step (fitness, parent selection, crossover,
        mutation, etc.) is shown on its own row together with the
        handler name and an output-shape hint. The string written to
        the logger is also returned.

        Parameters
        ----------
        line_length : int
            Total width of a printed line in characters.
        fill_character : str
            Character used to pad cells to the column width.
        line_character : str
            Character used to draw the lighter horizontal separator
            between rows.
        line_character2 : str
            Character used to draw the heavier separator between the
            header and the body.
        columns_equal_len : bool
            If True, the three columns are split into equal widths.
            Otherwise the widths follow the longest content in each
            column.
        print_step_parameters : bool
            If True, the extra parameters of each step are printed
            inside the step's row.
        print_parameters_summary : bool
            If True, a summary block of global parameters is printed
            below the table. When ``print_step_parameters`` is False,
            the per-step extras are folded into this summary block.

        Returns
        -------
        summary_output : str
            The full summary as a single string (the same text that
            was written to the logger).
        """

        summary_output = ""

        def fill_message(msg, line_length=line_length, fill_character=fill_character):
            num_spaces = int((line_length - len(msg))/2)
            num_spaces = int(num_spaces / len(fill_character))
            msg = "{spaces}{msg}{spaces}".format(
                msg=msg, spaces=fill_character * num_spaces)
            return msg

        def line_separator(line_length=line_length, line_character=line_character):
            num_characters = int(line_length / len(line_character))
            return line_character * num_characters

        def create_row(columns, line_length=line_length, fill_character=fill_character, split_percentages=None):
            filled_columns = []
            if split_percentages is None:
                split_percentages = [int(100/len(columns))] * 3
            columns_lengths = [int((split_percentages[idx] * line_length) / 100)
                               for idx in range(len(split_percentages))]
            for column_idx, column in enumerate(columns):
                current_column_length = len(column)
                extra_characters = columns_lengths[column_idx] - \
                    current_column_length
                filled_column = column + fill_character * extra_characters
                filled_columns.append(filled_column)

            return "".join(filled_columns)

        def print_parent_selection_params():
            nonlocal summary_output
            m = f"Number of Parents: {self.num_parents_mating}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if self.parent_selection_type == "tournament":
                m = f"K Tournament: {self.K_tournament}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_fitness_params():
            nonlocal summary_output
            if not self.fitness_batch_size is None:
                m = f"Fitness batch size: {self.fitness_batch_size}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_crossover_params():
            nonlocal summary_output
            if not self.crossover_probability is None:
                m = f"Crossover probability: {self.crossover_probability}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_mutation_params():
            nonlocal summary_output
            if not self.mutation_probability is None:
                m = f"Mutation Probability: {self.mutation_probability}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            if self.mutation_percent_genes == "default":
                m = f"Mutation Percentage: {self.mutation_percent_genes}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            # Number of mutation genes is already shown above.
            m = f"Mutation Genes: {self.mutation_num_genes}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Random Mutation Range: ({self.random_mutation_min_val}, {self.random_mutation_max_val})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if not self.gene_space is None:
                m = f"Gene Space: {self.gene_space}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Mutation by Replacement: {self.mutation_by_replacement}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Allow Duplicated Genes: {self.allow_duplicate_genes}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        def print_on_generation_params():
            nonlocal summary_output
            if not self.stop_criteria is None:
                m = f"Stop Criteria: {self.stop_criteria}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"

        def print_params_summary():
            nonlocal summary_output
            m = f"Population Size: ({self.sol_per_pop}, {self.num_genes})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Number of Generations: {self.num_generations}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Initial Population Range: ({self.init_range_low}, {self.init_range_high})"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

            if not print_step_parameters:
                print_fitness_params()

            if not print_step_parameters:
                print_parent_selection_params()

            if self.keep_elitism != 0:
                m = f"Keep Elitism: {self.keep_elitism}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            else:
                m = f"Keep Parents: {self.keep_parents}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Gene DType: {self.gene_type}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

            if not print_step_parameters:
                print_crossover_params()

            if not print_step_parameters:
                print_mutation_params()

            if not print_step_parameters:
                print_on_generation_params()

            if not self.parallel_processing is None:
                m = f"Parallel Processing: {self.parallel_processing}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            if not self.random_seed is None:
                m = f"Random Seed: {self.random_seed}"
                self.logger.info(m)
                summary_output = summary_output + m + "\n"
            m = f"Save Best Solutions: {self.save_best_solutions}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            m = f"Save Solutions: {self.save_solutions}"
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        m = line_separator(line_character=line_character)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = fill_message("PyGAD Lifecycle")
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"

        lifecycle_steps = ["on_start()", "Fitness Function", "On Fitness", "Parent Selection", "On Parents",
                           "Crossover", "On Crossover", "Mutation", "On Mutation", "On Generation", "On Stop"]
        lifecycle_functions = [self.on_start, self.fitness_func, self.on_fitness, self.select_parents, self.on_parents,
                               self.crossover, self.on_crossover, self.mutation, self.on_mutation, self.on_generation, self.on_stop]
        lifecycle_functions = [getattr(
            lifecycle_func, '__name__', "None") for lifecycle_func in lifecycle_functions]
        lifecycle_functions = [lifecycle_func + "()" if lifecycle_func !=
                               "None" else "None" for lifecycle_func in lifecycle_functions]
        lifecycle_output = ["None", "(1)", "None", f"({self.num_parents_mating}, {self.num_genes})", "None",
                            f"({self.num_parents_mating}, {self.num_genes})", "None", f"({self.num_parents_mating}, {self.num_genes})", "None", "None", "None"]
        lifecycle_step_parameters = [None, print_fitness_params, None, print_parent_selection_params, None,
                                     print_crossover_params, None, print_mutation_params, None, print_on_generation_params, None]

        if not columns_equal_len:
            max_lengths = [max(list(map(len, lifecycle_steps))), max(
                list(map(len, lifecycle_functions))), max(list(map(len, lifecycle_output)))]
            split_percentages = [
                int((column_len / sum(max_lengths)) * 100) for column_len in max_lengths]
        else:
            split_percentages = None

        header_columns = ["Step", "Handler", "Output Shape"]
        header_row = create_row(
            header_columns, split_percentages=split_percentages)
        m = header_row
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"

        for lifecycle_idx in range(len(lifecycle_steps)):
            lifecycle_column = [lifecycle_steps[lifecycle_idx],
                                lifecycle_functions[lifecycle_idx], lifecycle_output[lifecycle_idx]]
            if lifecycle_column[1] == "None":
                continue
            lifecycle_row = create_row(
                lifecycle_column, split_percentages=split_percentages)
            m = lifecycle_row
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
            if print_step_parameters:
                if not lifecycle_step_parameters[lifecycle_idx] is None:
                    lifecycle_step_parameters[lifecycle_idx]()
            m = line_separator(line_character=line_character)
            self.logger.info(m)
            summary_output = summary_output + m + "\n"

        m = line_separator(line_character=line_character2)
        self.logger.info(m)
        summary_output = summary_output + m + "\n"
        if print_parameters_summary:
            print_params_summary()
            m = line_separator(line_character=line_character2)
            self.logger.info(m)
            summary_output = summary_output + m + "\n"
        return summary_output

    def initialize_parents_array(self, shape):
        """
        Allocate an empty parents (or offspring) array with the right
        dtype. Uses the dtype of the first gene type when every gene
        shares the same type, otherwise falls back to ``object``.

        Parameters
        ----------
        shape : tuple
            The shape of the array, usually
            ``(num_parents, num_genes)``.

        Returns
        -------
        array : numpy.ndarray
            An uninitialised array of the requested shape and dtype.
        """
        if self.gene_type_single:
            return numpy.empty(shape, dtype=self.gene_type[0])
        else:
            return numpy.empty(shape, dtype=object)

    def change_population_dtype_and_round(self,
                                          population):
        """
        Cast a 2D population to the dtype encoded in
        ``self.gene_type`` and round non-integer genes to the
        configured precision. When ``gene_type_single`` is True, the
        same dtype and precision are applied to every gene; otherwise
        each gene gets its own dtype and precision.

        Parameters
        ----------
        population : list or numpy.ndarray
            A 2D iterable with shape ``(num_solutions, num_genes)``.

        Returns
        -------
        population_new : numpy.ndarray
            The same data cast (and rounded) to the right type.
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
        Cast and round one or more candidate values that all belong
        to the same gene index. Useful when generating mutation
        values for a specific gene.

        Parameters
        ----------
        gene_index : int
            Index of the gene whose dtype / precision should be used.
        gene_value : numeric or iterable
            Either a single value or a vector of values for that
            gene.

        Returns
        -------
        gene_value_new : numeric
            The first (or only) value after casting and rounding.
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
        Apply a random mutation value to a gene and cast / round the
        result. If ``mutation_by_replacement`` is True, the random
        value replaces the gene; otherwise it is added to the
        existing value.

        Parameters
        ----------
        random_value : numeric
            The freshly drawn mutation value.
        gene_index : int
            Index of the gene being mutated.
        gene_value : numeric
            Gene value before mutation. Only used when
            ``mutation_by_replacement`` is False.
        mutation_by_replacement : bool
            If True, replace the gene; otherwise add the random
            value to it.

        Returns
        -------
        gene_value_new : numeric
            The mutated value after casting and rounding.
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
        """
        Check that a gene constraint callable returned a list or
        numpy array whose elements are all members of the original
        candidate ``values``.

        Parameters
        ----------
        selected_values : list, numpy.ndarray, or other
            The return value from the user-supplied constraint
            callable.
        values : iterable
            The full set of candidate values that was passed to the
            callable.

        Returns
        -------
        valid : bool
            True when ``selected_values`` is a list or numpy array
            and is a subset of ``values``. False otherwise.
        """
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
        Pass a list of candidate values through the user-supplied
        gene constraint callable and return the subset that satisfies
        the constraint.

        Parameters
        ----------
        values : list or numpy.ndarray
            Candidate values to filter.
        solution : numpy.ndarray
            The solution that owns the gene. Passed to the constraint
            callable so it can look at the other genes if needed.
        gene_idx : int
            Index of the gene inside ``solution``.

        Returns
        -------
        filtered_values : list, numpy.ndarray, or None
            The values that satisfy the constraint, or None when no
            value satisfies it (a warning is issued in that case).

        Raises
        ------
        Exception
            If the gene has no constraint, or the constraint callable
            returns a result that is not a subset of ``values``.
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
            raise Exception("The output from the gene_constraint callable/function must be a list or NumPy array that is a subset of the passed values (second argument).")

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
        Return the dtype (and optional precision) for the gene at the
        given index. When ``gene_type_single`` is True the same dtype
        is returned for every gene; otherwise the per-gene entry is
        returned.

        Parameters
        ----------
        gene_index : int
            Index of the gene whose dtype is wanted. Ignored when
            ``gene_type_single`` is True.

        Returns
        -------
        dtype : type or list
            Either a single Python / numpy type, or a
            ``[type, precision]`` pair.
        """

        if self.gene_type_single == True:
            dtype = self.gene_type
        else:
            dtype = self.gene_type[gene_index]
        return dtype

    def get_random_mutation_range(self, gene_index):
        """
        Return the random-mutation range ``(min, max)`` for the gene
        at the given index. When ``random_mutation_min_val`` is a
        scalar, the same range is used for every gene; otherwise the
        per-gene entry is returned.

        Parameters
        ----------
        gene_index : int
            Index of the gene. Ignored when the range parameters are
            scalars.

        Returns
        -------
        range_min : numeric
            Lower bound of the random delta.
        range_max : numeric
            Upper bound of the random delta.
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
        Return the initial-population range ``(min, max)`` for the
        gene at the given index. When ``init_range_low`` is a scalar,
        the same range is used for every gene; otherwise the
        per-gene entry is returned.

        Parameters
        ----------
        gene_index : int
            Index of the gene. Ignored when the range parameters are
            scalars.

        Returns
        -------
        range_min : numeric
            Lower bound for the random initial gene value.
        range_max : numeric
            Upper bound for the random initial gene value.
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
        Generate one or more candidate values for the gene from its
        ``gene_space`` entry. Handles flat spaces, nested spaces,
        ``range`` objects, and ``{low, high, step}`` dictionaries.

        Parameters
        ----------
        gene_idx : int
            Index of the gene inside the solution.
        mutation_by_replacement : bool
            If True (mutation by replacement) the generated value is
            used as-is. If False the generated value is added to
            ``gene_value``. Set to True when building the initial
            population.
        solution : iterable or None
            The solution the gene belongs to. When provided and
            ``sample_size`` is 1, the helper tries to pick a value
            that does not duplicate any existing gene.
        gene_value : numeric or None
            The current gene value. Required when applying mutation
            with ``mutation_by_replacement=False`` so the random
            value can be added on top.
        sample_size : int
            Number of candidate values to generate. ``1`` returns a
            single number; larger values return an array; ``None``
            keeps the full integer range or a single float value.

        Returns
        -------
        value : numeric or numpy.ndarray
            A single value when ``sample_size=1``; otherwise an
            array of up to ``sample_size`` values.
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
        Generate one or more candidate values for the gene by drawing
        from the random range ``[range_min, range_max)``. For integer
        gene types the helper iterates over the discrete values; for
        float types it samples uniformly.

        Parameters
        ----------
        range_min : numeric
            Lower bound of the random range.
        range_max : numeric
            Upper bound of the random range.
        gene_value : numeric
            The current gene value, used when
            ``mutation_by_replacement`` is False so the random delta
            can be added to it.
        gene_idx : int
            Index of the gene inside the solution.
        mutation_by_replacement : bool
            If True, the random value replaces the gene; otherwise it
            is added.
        sample_size : int or None
            Number of candidate values to generate. ``1`` returns a
            single number; larger values return an array of up to
            that many values; ``None`` keeps every value in the
            integer range or returns a single float.
        step : int
            Step size used when enumerating an integer range.

        Returns
        -------
        random_value : numeric or numpy.ndarray
            A single value when ``sample_size=1``; otherwise an
            array of unique values.
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
        Dispatcher that picks between
        ``generate_gene_value_from_space`` (when ``self.gene_space``
        is set) and ``generate_gene_value_randomly`` (otherwise) to
        generate one or more candidate values for the gene.

        Parameters
        ----------
        gene_value : numeric
            The current gene value, used when
            ``mutation_by_replacement`` is False.
        gene_idx : int
            Index of the gene inside the solution.
        mutation_by_replacement : bool
            See ``generate_gene_value_randomly``.
        solution : iterable or None
            The solution that owns the gene. Used to avoid creating
            duplicates when ``sample_size=1`` and
            ``allow_duplicate_genes`` is False.
        range_min : numeric or None
            Lower bound for the random range. Required when
            ``self.gene_space`` is None.
        range_max : numeric or None
            Upper bound for the random range. Required when
            ``self.gene_space`` is None.
        sample_size : int or None
            Number of candidate values to generate.
        step : int
            Step size for the integer random range.

        Returns
        -------
        output : numeric or numpy.ndarray
            A single value when ``sample_size=1``; otherwise an
            array of values.
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
        Generate up to ``sample_size`` candidate values for the gene
        (via ``generate_gene_value``) and then filter them through
        the user-supplied ``gene_constraint`` callable.

        Parameters
        ----------
        range_min : numeric or None
            Lower bound of the random range.
        range_max : numeric or None
            Upper bound of the random range.
        gene_value : numeric
            The current gene value, used when
            ``mutation_by_replacement`` is False.
        gene_idx : int
            Index of the gene inside the solution.
        mutation_by_replacement : bool
            See ``generate_gene_value_randomly``.
        solution : iterable
            The solution that owns the gene. Passed to the
            constraint callable so it can look at the other genes.
        sample_size : int
            Number of candidate values to draw before filtering.
        step : int
            Step size for the integer random range.

        Returns
        -------
        values_filtered : numpy.ndarray or None
            Values that satisfy the constraint, or None if no
            candidate satisfies it.
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
