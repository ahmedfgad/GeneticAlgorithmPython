"""
The pygad.utils.mutation module has all the built-in mutation operators.
"""

import numpy
import random

import pygad

import warnings

class Mutation:

    def __init__(self):
        pass

    def random_mutation(self, offspring):
        """
        Dispatch to one of the four random-mutation backends depending
        on whether the user passed ``mutation_probability`` and whether
        ``gene_space`` is set. The replacement value for each mutated
        gene comes either from the gene space or from the
        ``random_mutation_min_val`` / ``random_mutation_max_val`` range.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.mutation_by_space(offspring)
            else:
                offspring = self.mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected from the space of values of each gene.
                offspring = self.mutation_probs_by_space(offspring)
            else:
                offspring = self.mutation_probs_randomly(offspring)

        return offspring

    def mutation_by_space(self, offspring):
        """
        Mutate ``self.mutation_num_genes`` genes per offspring by
        sampling a replacement value from the ``gene_space`` of each
        chosen gene.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), self.mutation_num_genes))
            swapped_genes = set()
            for gene_idx in mutation_indices:

                if gene_idx in swapped_genes:
                    continue

                value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                if self.allow_duplicate_genes == False and value_from_space == offspring[offspring_idx, gene_idx]:
                    # No value of the gene space is free (e.g. a permutation): swap the gene with another one instead.
                    offspring[offspring_idx] = self.swap_gene_by_space(solution=offspring[offspring_idx],
                                                                       gene_idx=gene_idx,
                                                                       swapped_genes=swapped_genes)
                    continue

                # Before assigning the selected value from the space to the gene, change its data type and round it.
                offspring[offspring_idx, gene_idx] = self.change_gene_dtype_and_round(gene_idx, value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         build_initial_pop=False)
        return offspring

    def mutation_probs_by_space(self, offspring):
        """
        Per-gene mutation that uses the ``gene_space`` for replacement
        values and ``self.mutation_probability`` to decide which genes
        to mutate. A gene is mutated when a uniform random draw is
        less than or equal to the probability threshold.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
        for offspring_idx in range(offspring.shape[0]):
            probs = numpy.random.random(size=offspring.shape[1])
            swapped_genes = set()
            for gene_idx in range(offspring.shape[1]):

                if gene_idx in swapped_genes:
                    continue

                if probs[gene_idx] <= self.mutation_probability:
                    value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                        gene_idx=gene_idx,
                                                                        sample_size=self.sample_size)

                    if self.allow_duplicate_genes == False and value_from_space == offspring[offspring_idx, gene_idx]:
                        # No value of the gene space is free (e.g. a permutation): swap the gene with another one instead.
                        offspring[offspring_idx] = self.swap_gene_by_space(solution=offspring[offspring_idx],
                                                                           gene_idx=gene_idx,
                                                                           swapped_genes=swapped_genes)
                        continue

                    # Assigning the selected value from the space to the gene.
                    offspring[offspring_idx, gene_idx] = self.change_gene_dtype_and_round(gene_idx, value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             build_initial_pop=False)
        return offspring

    def mutation_process_gene_value(self,
                                    solution,
                                    gene_idx,
                                    range_min=None,
                                    range_max=None,
                                    sample_size=100):
        """
        Pick a replacement value for a single gene. If the user passed a
        ``gene_constraint`` for that gene, the method draws up to
        ``sample_size`` candidate values, filters them through the
        constraint, and picks one of the survivors at random; if no
        candidate passes, the current value is kept.

        When no constraint exists, a single value is sampled directly
        from the gene space or the random-mutation range.

        Parameters
        ----------
        solution : numpy.ndarray
            The solution that owns the gene.
        gene_idx : int
            Index of the gene inside ``solution``.
        range_min : float, optional
            Lower bound of the random range. Used only when the gene
            has no ``gene_space``.
        range_max : float, optional
            Upper bound of the random range. Used only when the gene
            has no ``gene_space``.
        sample_size : int
            Maximum number of candidate values to draw when a gene
            constraint is in effect. The actual number can be smaller
            if the generator runs out of distinct values.

        Returns
        -------
        value_selected : numeric
            The new value for the gene. May be the old value if no
            candidate satisfied the constraint.
        """

        # Check if the gene has a constraint.
        if self.gene_constraint and self.gene_constraint[gene_idx]:
            # Generate values that meet the gene constraint. Select more than 1 value.
            # This method: 1) generates or selects the values 2) filters the values according to the constraint.
            values = self.get_valid_gene_constraint_values(range_min=range_min,
                                                           range_max=range_max,
                                                           gene_value=solution[gene_idx],
                                                           gene_idx=gene_idx,
                                                           mutation_by_replacement=self.mutation_by_replacement,
                                                           solution=solution,
                                                           sample_size=sample_size)
            if values is None:
                # No value found that satisfy the constraint.
                # Keep the old value.
                value_selected = solution[gene_idx]
            else:
                # Select a value randomly from the list of values satisfying the constraint.
                # If size is used with numpy.random.choice(), it returns an array even if it has a single value. To return a numeric value, not an array, then return index 0.
                value_selected = numpy.random.choice(values, size=1)[0]
        else:
            # The gene does not have a constraint. Just select a single value.
            value_selected = self.generate_gene_value(range_min=range_min,
                                                      range_max=range_max,
                                                      gene_value=solution[gene_idx],
                                                      gene_idx=gene_idx,
                                                      solution=solution,
                                                      mutation_by_replacement=self.mutation_by_replacement,
                                                      sample_size=1)
        # Even though its name is singular, it might hold multiple values.
        return value_selected

    def swap_gene_by_space(self,
                           solution,
                           gene_idx,
                           swapped_genes=None):
        """
        With ``allow_duplicate_genes=False``, a gene cannot take a new
        value from its space when all of them are used by other genes,
        as in a permutation (``gene_space=range(num_genes)``). To still
        change the solution, swap the gene's value with the value of
        another compatible gene, picked at random. Both values are
        cast and rounded for their destination genes. A swap is only
        allowed if this preserves their numeric values and both gene
        spaces, so it also preserves uniqueness. Genes with a
        ``gene_constraint`` are not swapped, and any constraints on
        other genes must still hold after the swap. Each gene
        participates in at most one fallback swap per mutation pass.

        Parameters
        ----------
        solution : numpy.ndarray
            The solution that owns the gene (modified in place).
        gene_idx : int
            Index of the gene inside ``solution``.
        swapped_genes : set, optional
            Indices already swapped in this offspring's mutation
            pass. Updated in place after a successful swap. Omit it
            when calling this helper independently.

        Returns
        -------
        solution : numpy.ndarray
            The solution after the swap, unchanged if no gene qualifies.
        """

        def gene_space_values(idx):
            if self.gene_space_nested or not self.gene_type_single:
                return self.gene_space_unpacked[idx]
            else:
                return self.gene_space_unpacked

        def has_constraint(idx):
            return bool(self.gene_constraint and self.gene_constraint[idx])

        if swapped_genes is None:
            swapped_genes = set()

        if gene_idx in swapped_genes or has_constraint(gene_idx):
            return solution

        gene_value = solution[gene_idx]
        candidates = []
        for other_idx, other_value in enumerate(solution):
            if (other_idx in swapped_genes or other_value == gene_value
                    or has_constraint(other_idx)):
                continue

            try:
                new_gene_value = self.change_gene_dtype_and_round(gene_idx, other_value)
                new_other_value = self.change_gene_dtype_and_round(other_idx, gene_value)
            except (OverflowError, ValueError, TypeError):
                # A value that cannot be represented by the destination
                # dtype cannot be part of a permutation-preserving swap.
                continue

            # Preserve the original set of numeric values. In particular,
            # casting or rounding must not turn a value into a duplicate.
            if new_gene_value != other_value or new_other_value != gene_value:
                continue
            if (new_gene_value not in gene_space_values(gene_idx)
                    or new_other_value not in gene_space_values(other_idx)):
                continue

            # A constraint on a different gene may depend on either
            # swapped position. Validate the complete candidate solution.
            if self.gene_constraint:
                candidate_solution = solution.copy()
                candidate_solution[gene_idx] = new_gene_value
                candidate_solution[other_idx] = new_other_value
                constraints_satisfied = True
                for idx, constraint in enumerate(self.gene_constraint):
                    if constraint is None:
                        continue
                    values = numpy.array([candidate_solution[idx]])
                    selected_values = constraint(candidate_solution.copy(), values.copy())
                    if not self.validate_gene_constraint_callable_output(selected_values, values):
                        raise Exception("The output from the gene_constraint callable/function must be a list or NumPy array that is a subset of the passed values (second argument).")
                    if len(selected_values) == 0:
                        constraints_satisfied = False
                        break
                if not constraints_satisfied:
                    continue

            candidates.append((other_idx, new_gene_value, new_other_value))

        if len(candidates) > 0:
            other_idx, new_gene_value, new_other_value = random.choice(candidates)
            solution[gene_idx] = new_gene_value
            solution[other_idx] = new_other_value
            swapped_genes.update((gene_idx, other_idx))
        return solution

    def mutation_randomly(self, offspring):
        """
        Mutate ``self.mutation_num_genes`` genes per offspring by
        drawing a new value from the random-mutation range for each
        chosen gene.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            # Return the indices of the genes to mutate.
            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), 
                                                         self.mutation_num_genes))
            for gene_idx in mutation_indices:

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # Generate a random value for mutation that meets the gene constraint, if one exists.
                random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                range_max=range_max,
                                                                solution=offspring[offspring_idx],
                                                                gene_idx=gene_idx,
                                                                sample_size=self.sample_size)

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size)

        return offspring

    def mutation_probs_randomly(self, offspring):
        """
        Per-gene mutation that uses the random-mutation range and
        ``self.mutation_probability`` to decide which genes to mutate.
        A gene is mutated when a uniform random draw is less than or
        equal to the probability threshold.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # Random mutation changes one or more genes in each offspring randomly.
        for offspring_idx in range(offspring.shape[0]):
            # The mutation probabilities for the current offspring.
            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # A gene is mutated only if its mutation probability is less than or equal to the threshold.
                if probs[gene_idx] <= self.mutation_probability:

                    # Generate a random value for mutation that meets the gene constraint, if one exists.
                    random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                    range_max=range_max,
                                                                    solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size)
        return offspring

    def polynomial_mutation(self, offspring):
        """
        Apply polynomial mutation. Each gene is mutated with
        probability ``self.mutation_probability`` (or with probability
        ``1/num_genes`` when ``mutation_probability`` is not set).

        The size of the change is set by
        ``self.polynomial_mutation_eta`` (a higher value means a
        smaller change). The per-gene bounds come from
        ``get_initial_population_range``.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (changed in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """
        eta = float(self.polynomial_mutation_eta)
        per_gene_probability = (self.mutation_probability
                                if self.mutation_probability is not None
                                else 1.0 / self.num_genes)
        eta_plus_one = eta + 1.0
        near_zero = 1e-14

        for sol_idx in range(offspring.shape[0]):
            for gene_idx in range(offspring.shape[1]):
                if numpy.random.random() > per_gene_probability:
                    continue

                range_min, range_max = self.get_initial_population_range(gene_index=gene_idx)
                lower = float(range_min)
                upper = float(range_max)
                if upper - lower < near_zero:
                    continue

                gene_value = float(offspring[sol_idx, gene_idx])
                delta_lower = (gene_value - lower) / (upper - lower)
                delta_upper = (upper - gene_value) / (upper - lower)

                rand_u = numpy.random.random()
                if rand_u <= 0.5:
                    xy = 1.0 - delta_lower
                    val = 2.0 * rand_u + (1.0 - 2.0 * rand_u) * pow(xy, eta_plus_one)
                    delta_q = pow(val, 1.0 / eta_plus_one) - 1.0
                else:
                    xy = 1.0 - delta_upper
                    val = 2.0 * (1.0 - rand_u) + 2.0 * (rand_u - 0.5) * pow(xy, eta_plus_one)
                    delta_q = 1.0 - pow(val, 1.0 / eta_plus_one)

                new_value = gene_value + delta_q * (upper - lower)
                new_value = numpy.clip(new_value, lower, upper)
                offspring[sol_idx, gene_idx] = new_value

                if self.allow_duplicate_genes == False:
                    offspring[sol_idx], _, _ = self.solve_duplicate_genes_randomly(
                        solution=offspring[sol_idx],
                        min_val=lower,
                        max_val=upper,
                        mutation_by_replacement=True,
                        gene_type=self.gene_type,
                        sample_size=self.sample_size)
        return offspring

    def swap_mutation(self, offspring):
        """
        Swap the values of two genes inside each offspring. The two
        genes are 2 different genes picked at random.
        Offspring with fewer than two genes are returned unchanged.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        if offspring.shape[1] < 2:
            return offspring

        for idx in range(offspring.shape[0]):
            mutation_gene1, mutation_gene2 = numpy.random.choice(offspring.shape[1], size=2, replace=False)

            temp = offspring[idx, mutation_gene1]
            offspring[idx, mutation_gene1] = offspring[idx, mutation_gene2]
            offspring[idx, mutation_gene2] = temp
        return offspring

    def inversion_mutation(self, offspring):
        """
        Pick a slice of genes inside each offspring and reverse the
        order of the values in that slice.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            genes_to_scramble = numpy.flip(offspring[idx, mutation_gene1:mutation_gene2])
            offspring[idx, mutation_gene1:mutation_gene2] = genes_to_scramble
        return offspring

    def scramble_mutation(self, offspring):
        """
        Pick a slice of genes inside each offspring and shuffle the
        values in that slice. The segment contains num_genes // 2 genes;
        genes outside it are unchanged. A shuffle may keep the original
        order, and segments with fewer than two genes cannot change.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        for offspring_idx in range(offspring.shape[0]):
            segment_start = numpy.random.randint(low=0, high=numpy.ceil(offspring.shape[1]/2 + 1), size=1)[0]
            segment_end = segment_start + int(offspring.shape[1]/2)
            # Shuffle values, not indices, so each permutation of the
            # selected segment is possible without a separate reversal.
            genes_to_scramble = offspring[offspring_idx, segment_start:segment_end].copy()
            numpy.random.shuffle(genes_to_scramble)
            offspring[offspring_idx, segment_start:segment_end] = genes_to_scramble
        return offspring

    def adaptive_mutation_population_fitness(self, offspring):
        """Evaluate the kept solutions and actual offspring before mutation."""
        if self.keep_elitism > 0:
            parents_to_keep, kept_indices = self.steady_state_selection(
                self.last_generation_fitness, num_parents=self.keep_elitism)
        elif self.keep_parents == -1:
            parents_to_keep = self.last_generation_parents.copy()
            kept_indices = self.last_generation_parents_indices
        elif self.keep_parents > 0:
            parents_to_keep, kept_indices = self.steady_state_selection(
                self.last_generation_fitness, num_parents=self.keep_parents)
        else:
            parents_to_keep = []
            kept_indices = []

        num_kept = len(parents_to_keep)
        temp_population = numpy.empty_like(self.population)
        if num_kept:
            temp_population[:num_kept] = parents_to_keep
        temp_population[num_kept:] = offspring
        offspring_indices = list(range(num_kept, len(temp_population)))
        offspring_fitness = self._evaluate_fitness(
            temp_population, offspring_indices, adaptive=True)
        # Offspring can have fractional fitness even when the previous
        # population's values were integers. Let NumPy infer the new dtype.
        fitness = numpy.array(list(self.last_generation_fitness[kept_indices])
                              + offspring_fitness)
        return numpy.mean(fitness, axis=0), fitness[num_kept:]

    def adaptive_mutation(self, offspring):
        """
        Dispatch to one of the four adaptive-mutation backends based on
        whether ``mutation_probability`` is set and whether
        ``gene_space`` is provided. With adaptive mutation, the
        per-solution mutation rate is high for below-average solutions
        and low for above-average solutions.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # If the attribute 'gene_space' exists (i.e. not None), then the mutation values are selected from the 'gene_space' parameter according to the space of values of each gene. Otherwise, it is selected randomly based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

        if self.mutation_probability is None:
            # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_randomly(offspring)
        else:
            # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
            if not (self.gene_space is None):
                # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
                offspring = self.adaptive_mutation_probs_by_space(offspring)
            else:
                # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
                offspring = self.adaptive_mutation_probs_randomly(offspring)

        return offspring

    def adaptive_mutation_by_space(self, offspring):
        """
        Adaptive mutation that uses ``mutation_num_genes`` and
        ``gene_space``. The number of mutated genes per offspring is
        the first element of ``mutation_num_genes`` for below-average
        solutions and the second element for above-average ones. New
        values come from the ``gene_space``.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """
        
        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]

            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            swapped_genes = set()
            for gene_idx in mutation_indices:

                if gene_idx in swapped_genes:
                    continue

                value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                if self.allow_duplicate_genes == False and value_from_space == offspring[offspring_idx, gene_idx]:
                    # No value of the gene space is free (e.g. a permutation): swap the gene with another one instead.
                    offspring[offspring_idx] = self.swap_gene_by_space(solution=offspring[offspring_idx],
                                                                       gene_idx=gene_idx,
                                                                       swapped_genes=swapped_genes)
                    continue

                # Assigning the selected value from the space to the gene.
                offspring[offspring_idx, gene_idx] = self.change_gene_dtype_and_round(gene_idx, value_from_space)

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         build_initial_pop=False)
        return offspring

    def adaptive_mutation_randomly(self, offspring):
        """
        Adaptive mutation that uses ``mutation_num_genes`` and the
        random-mutation range. The number of mutated genes per
        offspring is the first element of ``mutation_num_genes`` for
        below-average solutions and the second element for
        above-average ones. New values are sampled uniformly from the
        random-mutation range.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The number of genes to mutate depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if fitness_comparison:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_num_genes = self.mutation_num_genes[0]
                else:
                    adaptive_mutation_num_genes = self.mutation_num_genes[1]

            mutation_indices = numpy.array(random.sample(range(0, self.num_genes), adaptive_mutation_num_genes))
            for gene_idx in mutation_indices:

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                # Generate a random value fpr mutation that meet the gene constraint if exists.
                random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                range_max=range_max,
                                                                solution=offspring[offspring_idx],
                                                                gene_idx=gene_idx,
                                                                sample_size=self.sample_size)

                offspring[offspring_idx, gene_idx] = random_value

                if self.allow_duplicate_genes == False:
                    offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                         min_val=range_min,
                                                                                         max_val=range_max,
                                                                                         mutation_by_replacement=self.mutation_by_replacement,
                                                                                         gene_type=self.gene_type,
                                                                                         sample_size=self.sample_size)
        return offspring

    def adaptive_mutation_probs_by_space(self, offspring):
        """
        Adaptive mutation that uses ``mutation_probability`` and
        ``gene_space``. The probability threshold per offspring is the
        first element of ``mutation_probability`` for below-average
        solutions and the second element for above-average ones. Each
        gene is replaced with a value from its ``gene_space`` when its
        random draw falls below the chosen threshold.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        # For each offspring, a value from the gene space is selected randomly and assigned to the selected gene for mutation.

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            swapped_genes = set()
            for gene_idx in range(offspring.shape[1]):

                if gene_idx in swapped_genes:
                    continue

                if probs[gene_idx] <= adaptive_mutation_probability:

                    value_from_space = self.mutation_process_gene_value(solution=offspring[offspring_idx],
                                                                        gene_idx=gene_idx,
                                                                        sample_size=self.sample_size)

                    if self.allow_duplicate_genes == False and value_from_space == offspring[offspring_idx, gene_idx]:
                        # No value of the gene space is free (e.g. a permutation): swap the gene with another one instead.
                        offspring[offspring_idx] = self.swap_gene_by_space(solution=offspring[offspring_idx],
                                                                           gene_idx=gene_idx,
                                                                           swapped_genes=swapped_genes)
                        continue

                    # Assigning the selected value from the space to the gene.
                    offspring[offspring_idx, gene_idx] = self.change_gene_dtype_and_round(gene_idx, value_from_space)

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             build_initial_pop=False)
        return offspring

    def adaptive_mutation_probs_randomly(self, offspring):
        """
        Adaptive mutation that uses ``mutation_probability`` and the
        random-mutation range. The probability threshold per offspring
        is the first element of ``mutation_probability`` for
        below-average solutions and the second element for
        above-average ones. Each gene is replaced with a value sampled
        uniformly from the random-mutation range when its random draw
        falls below the chosen threshold.

        Parameters
        ----------
        offspring : numpy.ndarray
            The offspring solutions to mutate (modified in place).

        Returns
        -------
        offspring : numpy.ndarray
            The mutated offspring.
        """

        average_fitness, offspring_fitness = self.adaptive_mutation_population_fitness(offspring)

        # Adaptive random mutation changes one or more genes in each offspring randomly.
        # The probability of mutating a gene depends on the solution's fitness value.
        for offspring_idx in range(offspring.shape[0]):
            ## TODO Make edits to work with multi-objective optimization.
            # Compare the fitness of each offspring to the average fitness of each objective function.
            fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

            # Check if the problem is single or multi-objective optimization.
            if type(fitness_comparison) in [bool, numpy.bool_]:
                # Single-objective optimization problem.
                if offspring_fitness[offspring_idx] < average_fitness:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]
            else:
                # Multi-objective optimization problem.

                # Get the sum of the pool array (result of comparison).
                # True is considered 1 and False is 0.
                fitness_comparison_sum = sum(fitness_comparison)
                # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                # If True, then use the first percentage. 
                # If False, use the second percentage.
                if fitness_comparison_sum >= len(fitness_comparison)/2:
                    adaptive_mutation_probability = self.mutation_probability[0]
                else:
                    adaptive_mutation_probability = self.mutation_probability[1]

            probs = numpy.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):

                range_min, range_max = self.get_random_mutation_range(gene_idx)

                if probs[gene_idx] <= adaptive_mutation_probability:
                    # Generate a random value for mutation that meets the gene constraint, if one exists.
                    random_value = self.mutation_process_gene_value(range_min=range_min,
                                                                    range_max=range_max,
                                                                    solution=offspring[offspring_idx],
                                                                    gene_idx=gene_idx,
                                                                    sample_size=self.sample_size)

                    offspring[offspring_idx, gene_idx] = random_value

                    if self.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = self.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                             min_val=range_min,
                                                                                             max_val=range_max,
                                                                                             mutation_by_replacement=self.mutation_by_replacement,
                                                                                             gene_type=self.gene_type,
                                                                                             sample_size=self.sample_size)
        return offspring
