"""
The pygad.helper.unique module has helper methods to solve duplicate genes and make sure every gene is unique.
"""

from collections import deque
import numpy
import warnings
import random
import pygad


class Unique:

    def get_duplicate_gene_indices(self, solution):
        """Return the indices after the first occurrence of each gene value."""
        seen_values = set()
        duplicate_indices = set()
        for gene_index, gene_value in enumerate(solution):
            value_key = self._gene_value_key(gene_value)
            if value_key in seen_values:
                duplicate_indices.add(gene_index)
            else:
                seen_values.add(value_key)
        return duplicate_indices

    def _gene_value_key(self, gene_value):
        """Compare exact numeric values without NumPy scalar promotion."""
        if isinstance(gene_value, numpy.generic):
            gene_value = gene_value.item()
        # Preserve the treatment of repeated NaNs as duplicates.
        if gene_value != gene_value:
            return ('nan',)
        return gene_value

    def solve_duplicate_genes_in_population(self, population, build_initial_pop=False):
        """
        Convert and round a population before repairing each solution.
        Used for initialization, population growth, and user-supplied
        operator or callback outputs. Returns a repaired copy.
        """
        population = self.change_population_dtype_and_round(population)
        for solution_index, solution in enumerate(population):
            population[solution_index], _, _ = self.solve_duplicate_genes(
                solution, build_initial_pop=build_initial_pop)
        return population

    def solve_duplicate_genes(self, solution, build_initial_pop=False,
                              mutation_by_replacement=None, sample_size=None,
                              min_val=None, max_val=None, warn=True):
        """
        Resolve duplicates using the space, type, precision, and range of
        each gene. Existing unique values are kept whenever possible.
        A chain of replacements can move an earlier gene to make room
        for a later gene, including one whose space has a single value.

        Parameters
        ----------
        solution : numpy.ndarray or list
            The solution to repair. The input is not modified.
        build_initial_pop : bool
            Use initialization ranges and replacement when True.
            Otherwise use the random-mutation ranges and mode.
        mutation_by_replacement : bool or None
            Override the mutation mode. None uses the GA setting.
        sample_size : int or None
            Number of candidates for continuous ranges. None uses
            ``self.sample_size``. Finite spaces are searched in full.
        min_val, max_val : numeric, iterable, or None
            Optional range overrides for the compatibility helpers.
        warn : bool
            Issue warnings for duplicates left after all repair attempts.

        Returns
        -------
        solution : numpy.ndarray
            A copy of the solution after repair.
        duplicate_indices : set
            Indices that still duplicate earlier genes.
        num_unsolved_duplicates : int
            The number of remaining duplicate indices.
        """
        new_solution = self.change_population_dtype_and_round([solution])[0]
        duplicate_indices = self.get_duplicate_gene_indices(new_solution)
        if not duplicate_indices:
            return new_solution, duplicate_indices, 0

        if sample_size is None:
            sample_size = self.sample_size
        if mutation_by_replacement is None:
            mutation_by_replacement = self.mutation_by_replacement
        if build_initial_pop:
            mutation_by_replacement = True

        candidate_values = []
        for gene_index, gene_value in enumerate(new_solution):
            dtype = self.get_gene_dtype(gene_index)
            if build_initial_pop and min_val is None:
                values = self.get_initial_population_gene_candidates(gene_index, sample_size)
            elif self.gene_space is None:
                if min_val is None:
                    if build_initial_pop:
                        range_min, range_max = self.get_initial_population_range(gene_index)
                    else:
                        range_min, range_max = self.get_random_mutation_range(gene_index)
                elif type(min_val) in self.supported_int_float_types:
                    range_min, range_max = min_val, max_val
                else:
                    range_min, range_max = min_val[gene_index], max_val[gene_index]
                values = self.generate_gene_value_randomly(
                    range_min=range_min, range_max=range_max,
                    gene_value=gene_value, gene_idx=gene_index,
                    mutation_by_replacement=mutation_by_replacement,
                    sample_size=None if numpy.issubdtype(numpy.dtype(dtype[0]), numpy.integer) else sample_size)
            else:
                values = self.get_gene_space_values(
                    gene_idx=gene_index,
                    gene_value=None if build_initial_pop else gene_value,
                    mutation_by_replacement=mutation_by_replacement,
                    sample_size=sample_size)

            # Compare converted values: rounding and casting can turn
            # different candidates into the same numeric value.
            values = list(dict.fromkeys((value if dtype[0] is object else dtype[0](value)) for value in numpy.atleast_1d(values)))
            values = [value for value in values if value != gene_value]
            random.shuffle(values)
            # Keep manually supplied values and values inherited from parents.
            # Only a replacement must come from the current domain.
            candidate_values.append([gene_value] + values)

        # First try values satisfying each constraint in the current
        # solution. This completely searches independent finite constraints.
        # Keep the full domains for constraints depending on changed genes.
        constrained_values = []
        for gene_index, values in enumerate(candidate_values):
            if self.gene_constraint and self.gene_constraint[gene_index]:
                selected_values = self.filter_gene_values_by_constraint(
                    numpy.array(values), new_solution, gene_index, warn=False)
                dtype = self.get_gene_dtype(gene_index)
                constrained_values.append([] if selected_values is None else [(value if dtype[0] is object else dtype[0](value)) for value in selected_values])
            else:
                constrained_values.append(values)
        repaired_solution = self._assign_unique_gene_values(new_solution, constrained_values)
        if self.solution_satisfies_gene_constraints(repaired_solution):
            new_solution = repaired_solution
        if self.get_duplicate_gene_indices(new_solution) and self.gene_constraint:
            unconstrained_solution = self._assign_unique_gene_values(new_solution, candidate_values)
            if self.solution_satisfies_gene_constraints(unconstrained_solution):
                new_solution = unconstrained_solution
            elif not self.get_duplicate_gene_indices(unconstrained_solution):
                # A dependent constraint must see the complete assignment,
                # including other positions changed by the replacement chain.
                constrained_solution = self._assign_unique_gene_values_by_constraint(
                    new_solution, candidate_values, sample_size)
                if constrained_solution is not None:
                    new_solution = constrained_solution

        duplicate_indices = self.get_duplicate_gene_indices(new_solution)
        if warn and not self.suppress_warnings:
            for gene_index in sorted(duplicate_indices):
                stage = "while creating the initial population" if build_initial_pop else f"at generation {getattr(self, 'generations_completed', 0)}"
                warnings.warn(f"Failed to find a unique value for gene with index {gene_index} whose value is {new_solution[gene_index]} {stage}. Consider adding more values in the gene space, using a wider range, or increasing sample_size for continuous ranges and gene constraints.")
        return new_solution, duplicate_indices, len(duplicate_indices)

    def _assign_unique_gene_values(self, solution, candidate_values):
        """
        Find a maximum assignment of different candidate values to genes.
        Search replacement chains iteratively so long chromosomes do not
        depend on Python's recursion limit. Without constraints, a finite
        candidate space is searched completely.
        """
        new_solution = solution.copy()
        value_owners = {}
        duplicate_indices = []
        for gene_index, gene_value in enumerate(solution):
            value_key = self._gene_value_key(gene_value)
            if value_key in value_owners:
                duplicate_indices.append(gene_index)
            else:
                value_owners[value_key] = gene_index

        for duplicate_index in duplicate_indices:
            genes_to_search = deque([duplicate_index])
            previous_genes = {duplicate_index: None}
            replacement_found = False
            while genes_to_search and not replacement_found:
                gene_index = genes_to_search.popleft()
                for value in candidate_values[gene_index]:
                    if self._gene_value_key(value) not in value_owners:
                        # Walk back from the unused value to the duplicate.
                        # Each gene releases its predecessor's needed value.
                        while True:
                            new_solution[gene_index] = value
                            value_owners[self._gene_value_key(value)] = gene_index
                            previous_gene = previous_genes[gene_index]
                            if previous_gene is None:
                                break
                            gene_index, value = previous_gene
                        replacement_found = True
                        break
                    owner_index = value_owners[self._gene_value_key(value)]
                    if owner_index not in previous_genes:
                        previous_genes[owner_index] = (gene_index, value)
                        genes_to_search.append(owner_index)
        return new_solution

    def _assign_unique_gene_values_by_constraint(self, solution, candidate_values,
                                                 sample_size):
        """
        Try alternative complete assignments when a replacement chain
        violates a dependent constraint. Limit tentative assignments to
        ``sample_size * num_genes`` to keep arbitrary user constraints
        from causing an unbounded combinatorial search.
        """
        gene_order = sorted(range(len(solution)), key=lambda index: len(candidate_values[index]))
        candidate_solution = solution.copy()
        candidate_positions = [0] * len(solution)
        selected_values = set()
        search_depth = 0
        num_attempts = 0
        max_attempts = sample_size * len(solution)
        while search_depth >= 0 and (num_attempts < max_attempts or search_depth == len(solution)):
            if search_depth == len(solution):
                if self.solution_satisfies_gene_constraints(candidate_solution):
                    return candidate_solution
                search_depth -= 1
                selected_values.remove(self._gene_value_key(candidate_solution[gene_order[search_depth]]))
                continue
            gene_index = gene_order[search_depth]
            values = candidate_values[gene_index]
            if candidate_positions[search_depth] == len(values):
                candidate_positions[search_depth] = 0
                search_depth -= 1
                if search_depth >= 0:
                    selected_values.remove(self._gene_value_key(candidate_solution[gene_order[search_depth]]))
                continue
            value = values[candidate_positions[search_depth]]
            candidate_positions[search_depth] += 1
            num_attempts += 1
            value_key = self._gene_value_key(value)
            if value_key in selected_values:
                continue
            candidate_solution[gene_index] = value
            selected_values.add(value_key)
            search_depth += 1
        return None

    def solution_satisfies_gene_constraints(self, solution):
        """Check all gene constraints against a complete candidate solution."""
        if not self.gene_constraint:
            return True
        for gene_index, constraint in enumerate(self.gene_constraint):
            if constraint is None:
                continue
            selected_values = self.filter_gene_values_by_constraint(
                numpy.array([solution[gene_index]]), solution, gene_index, warn=False)
            if selected_values is None:
                return False
        return True

    def solve_duplicate_genes_randomly(self, solution, min_val, max_val,
                                       mutation_by_replacement, gene_type,
                                       sample_size=100):
        """
        Compatibility helper for repair using explicit random ranges.
        Returns the repaired solution, remaining duplicate indices, and
        their count. Gene types are obtained from the GA configuration.
        """
        return self.solve_duplicate_genes(solution, min_val=min_val, max_val=max_val,
                                          mutation_by_replacement=mutation_by_replacement,
                                          sample_size=sample_size)

    def solve_duplicate_genes_by_space(self, solution, gene_type,
                                       mutation_by_replacement, sample_size=100,
                                       build_initial_pop=False):
        """Compatibility helper for repair using the configured gene space."""
        return self.solve_duplicate_genes(solution, build_initial_pop=build_initial_pop,
                                          mutation_by_replacement=mutation_by_replacement,
                                          sample_size=sample_size)

    def unique_int_gene_from_range(self, solution, gene_index, min_val, max_val,
                                   mutation_by_replacement, gene_type, step=1):
        """Return an unused integer candidate, or keep the gene if none exists."""
        return self._unique_gene_from_range(solution, gene_index, min_val, max_val,
                                            mutation_by_replacement, None, step)

    def unique_float_gene_from_range(self, solution, gene_index, min_val, max_val,
                                     mutation_by_replacement, gene_type,
                                     sample_size=100):
        """Return an unused float candidate, or keep the gene if none exists."""
        return self._unique_gene_from_range(solution, gene_index, min_val, max_val,
                                            mutation_by_replacement, sample_size, 1)

    def _unique_gene_from_range(self, solution, gene_index, min_val, max_val,
                                mutation_by_replacement, sample_size, step):
        """Generate, filter, and select a candidate for the compatibility helpers."""
        values = self.generate_gene_value_randomly(
            range_min=min_val, range_max=max_val, gene_value=solution[gene_index],
            gene_idx=gene_index, mutation_by_replacement=mutation_by_replacement,
            sample_size=sample_size, step=step)
        return self._select_unique_value_by_constraint(values, solution, gene_index)

    def select_unique_value(self, gene_values, solution, gene_index):
        """Select an unused value, accepting both scalar and array candidates."""
        gene_values = list(numpy.atleast_1d(gene_values))
        used_values = {self._gene_value_key(value) for value in solution}
        values_to_select_from = list({self._gene_value_key(value): value for value in gene_values
                                      if self._gene_value_key(value) not in used_values}.values())
        if values_to_select_from:
            return random.choice(values_to_select_from)
        if solution[gene_index] is None:
            if not gene_values:
                raise ValueError(f"There are no values to select for the gene at index {gene_index}.")
            return random.choice(gene_values)
        return solution[gene_index]

    def _select_unique_value_by_constraint(self, values, solution, gene_index):
        """Filter candidates before selecting an unused value for one gene."""
        values = numpy.atleast_1d(values)
        if self.gene_constraint and self.gene_constraint[gene_index]:
            values = self.filter_gene_values_by_constraint(values, solution, gene_index)
            if values is None:
                return solution[gene_index]
        return self.select_unique_value(values, solution, gene_index)

    def unique_genes_by_space(self, solution, gene_type, not_unique_indices,
                              mutation_by_replacement, sample_size=100,
                              build_initial_pop=False):
        """Compatibility helper that repairs duplicates and replacement chains."""
        return self.solve_duplicate_genes_by_space(solution, gene_type,
                                                   mutation_by_replacement,
                                                   sample_size, build_initial_pop)

    def unique_gene_by_space(self, solution, gene_idx, gene_type,
                             mutation_by_replacement, sample_size=100,
                             build_initial_pop=False):
        """Return an unused candidate from a gene's space, if available."""
        values = self.get_gene_space_values(
            gene_idx, None if build_initial_pop else solution[gene_idx],
            mutation_by_replacement, sample_size)
        return self._select_unique_value_by_constraint(values, solution, gene_idx)

    def find_two_duplicates(self, solution, gene_space_unpacked):
        """Return a duplicate gene with alternatives, or ``(None, None)``."""
        duplicate_values = {self._gene_value_key(solution[index])
                            for index in self.get_duplicate_gene_indices(solution)}
        for gene_index, gene_value in enumerate(solution):
            if self._gene_value_key(gene_value) not in duplicate_values:
                continue
            space = gene_space_unpacked[gene_index] if self.gene_space_nested or not self.gene_type_single else gene_space_unpacked
            if len({self._gene_value_key(value) for value in numpy.atleast_1d(space)}) > 1:
                return gene_index, gene_value
        return None, None

    def unpack_gene_space(self, range_min, range_max, sample_size_from_inf_range=100):
        """
        Return converted finite spaces and samples of continuous spaces.
        This attribute is a snapshot for inspection. Value generation
        reads the original space so ``None`` entries remain random and
        use the current initialization or mutation range.
        """
        if self.gene_space is None:
            return None
        if self.gene_space_nested:
            num_spaces = len(self.gene_space)
        elif not self.gene_type_single:
            num_spaces = len(self.gene_type)
        else:
            num_spaces = 1
        unpacked_spaces = []
        for gene_index in range(num_spaces):
            if type(range_min) in self.supported_int_float_types:
                low, high = range_min, range_max
            else:
                low, high = range_min[gene_index], range_max[gene_index]
            space = self.gene_space[gene_index] if self.gene_space_nested else self.gene_space
            # Continuous spaces and None entries are inspection samples.
            # They must not allocate a large integer range or consume the
            # random draws used to generate the population.
            if space is None or (type(space) is dict and 'step' not in space):
                if type(space) is dict:
                    low, high = space['low'], space['high']
                unpacked_spaces.append(self._initial_population_range_snapshot(
                    gene_index, low, high, sample_size_from_inf_range))
            elif type(space) in [list, tuple, numpy.ndarray] and any(value is None for value in space):
                values = [value for value in space if value is not None]
                values.extend(self._initial_population_range_snapshot(
                    gene_index, low, high, sample_size_from_inf_range))
                unpacked_spaces.append(numpy.unique(self.change_gene_dtype_and_round(gene_index, values)))
            else:
                unpacked_spaces.append(self.get_gene_space_values(
                    gene_index, sample_size=sample_size_from_inf_range,
                    range_min=low, range_max=high))
        if not self.gene_space_nested and self.gene_type_single:
            return unpacked_spaces[0]
        return unpacked_spaces

    def solve_duplicates_deeply(self, solution):
        """Repair replacement chains, returning None if no progress is possible."""
        repaired_solution, duplicate_indices, _ = self.solve_duplicate_genes(solution, warn=False)
        if len(duplicate_indices) < len(self.get_duplicate_gene_indices(solution)):
            return repaired_solution
        return None
