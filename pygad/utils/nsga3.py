import warnings

import numpy


# Weight used to amplify the off-axis terms in the ASF score when looking
# for the extreme point of each objective. A very small weight makes any
# deviation on a non-target axis huge so it dominates the score.
ASF_EPSILON = 1e-6

# Numbers smaller than this are treated as zero when we check for a
# singular linear system or a collapsed axis range.
INTERCEPT_NEAR_ZERO = 1e-12


class NSGA3:

    def __init__(self):
        pass

    def generate_reference_points(self, num_objectives, num_divisions):
        """
        Build the structured grid of reference points on the unit simplex
        using the Das-Dennis (stars-and-bars) method.

        Each reference point has the form (a_1/p, a_2/p, ..., a_M/p) where
        the a_i are non-negative integers that sum to num_divisions. The
        total number of points is C(M + p - 1, p).

        Parameters
        ----------
        num_objectives : int
            The number of objectives, M.
        num_divisions : int
            The number of divisions per axis, p.

        Returns
        -------
        reference_points : numpy.ndarray
            A 2D array of shape (n_points, num_objectives). Each row is one
            reference point and its values sum to 1.0.
        """
        compositions = list(_enumerate_compositions(num_objectives, num_divisions))
        as_array = numpy.array(compositions, dtype=float)
        return as_array / num_divisions

    def compute_ideal_point(self, fitness):
        """
        Return the ideal point: the best fitness value for each objective
        across the input fitness rows. PyGAD maximises, so the best value
        per objective is the column maximum.

        Parameters
        ----------
        fitness : numpy.ndarray
            A 2D array of fitness values, one row per solution.

        Returns
        -------
        ideal_point : numpy.ndarray
            A 1D array of length M with the column maximum of fitness.
        """
        return numpy.asarray(fitness).max(axis=0)

    def find_extreme_points(self, fitness, ideal_point, epsilon=ASF_EPSILON):
        """
        For each objective axis, find the solution that best represents the
        corner of that axis. This is done by running the Achievement
        Scalarizing Function (ASF) once per axis with a weight vector that
        puts weight 1.0 on the target axis and a tiny weight (epsilon) on
        every other axis. The solution with the smallest ASF score wins.

        Parameters
        ----------
        fitness : numpy.ndarray
            A 2D array of fitness values, one row per solution.
        ideal_point : numpy.ndarray
            The ideal point.
        epsilon : float
            The small weight used for off-axis objectives.

        Returns
        -------
        extreme_points : numpy.ndarray
            A 2D array of shape (M, M). Row i is the fitness vector of the
            solution selected as the extreme for objective i.
        """
        fitness = numpy.asarray(fitness, dtype=float)
        num_objectives = ideal_point.shape[0]
        # Shortfall from the ideal point on each objective. Always >= 0
        # because the ideal is the column max.
        shortfall = ideal_point - fitness
        extremes = numpy.empty((num_objectives, num_objectives), dtype=float)
        for axis in range(num_objectives):
            # Weight is 1 on the target axis, tiny on every other axis.
            weights = numpy.full(num_objectives, epsilon)
            weights[axis] = 1.0
            asf_per_solution = (shortfall / weights).max(axis=1)
            # The lowest ASF wins. argmin returns the first occurrence so
            # ties go to the lower index.
            extremes[axis] = fitness[numpy.argmin(asf_per_solution)]
        return extremes

    def compute_intercepts(self, extreme_points, ideal_point, fallback_fitness):
        """
        Fit a hyperplane through the M extreme points and return the
        intercept point on each axis. The result is the point we use to
        scale every objective to the [0, 1] range during normalization.

        The NSGA-III paper define the intercept as the point that
        normalizes to value 1 on its own axis (i.e. each extreme row lands
        on a simplex corner after normalization). The math is:

            (extreme_points - ideal_point) @ b = 1
            intercepts = ideal_point + 1 / b

        When the linear system cannot be solved, when any coefficient is
        too close to zero, or when the resulting intercept ends up on the
        wrong side of the ideal point, fall back to the worst observed
        value per objective (the column minimum under maximization).

        Two extra safety steps run after the linear solve:
          1. If an intercept value extrapolates past the worst observed
             value for that objective, clip it back to the worst value.
          2. If the gap between an intercept and the ideal point shrinks
             below INTERCEPT_NEAR_ZERO after clipping, replace that
             intercept with the worst observed value so the normalization
             denominator stays non-zero.

        Parameters
        ----------
        extreme_points : numpy.ndarray
            The M extreme points returned by find_extreme_points.
        ideal_point : numpy.ndarray
            The ideal point.
        fallback_fitness : numpy.ndarray
            The fitness pool used to compute the fallback nadir. Usually
            the same fitness array used to find the extreme points.

        Returns
        -------
        intercepts : numpy.ndarray
            A 1D array of length M with the per-axis intercept values.
        """
        ideal_point = numpy.asarray(ideal_point, dtype=float)
        extreme_points = numpy.asarray(extreme_points, dtype=float)
        fallback_fitness = numpy.asarray(fallback_fitness, dtype=float)
        # Worst per objective under maximization is the column minimum.
        worst_per_objective = fallback_fitness.min(axis=0)
        translated = extreme_points - ideal_point
        try:
            coefficients = numpy.linalg.solve(translated,
                                              numpy.ones(ideal_point.shape[0]))
        except numpy.linalg.LinAlgError:
            return worst_per_objective
        # A near-zero coefficient means 1/b is huge and the intercept is
        # essentially undefined on that axis.
        if numpy.any(numpy.abs(coefficients) < INTERCEPT_NEAR_ZERO):
            return worst_per_objective
        intercepts = ideal_point + 1.0 / coefficients
        # Under maximization a valid intercept sits strictly below the
        # ideal. If it does not, the normalization denominator would flip
        # sign and produce nonsense values.
        if numpy.any(intercepts >= ideal_point - INTERCEPT_NEAR_ZERO):
            return worst_per_objective
        # Cap the intercept at the worst observed value so we never
        # extrapolate the hyperplane past the real data range.
        overshoot = intercepts < worst_per_objective
        intercepts = numpy.where(overshoot, worst_per_objective, intercepts)
        # If capping leaves the gap |intercept - ideal| too small, reset
        # that axis to the worst observed value.
        collapsed = numpy.abs(intercepts - ideal_point) < INTERCEPT_NEAR_ZERO
        intercepts = numpy.where(collapsed, worst_per_objective, intercepts)
        return intercepts

    def normalise_fitness(self, fitness, ideal_point, intercepts):
        """
        Scale each fitness row to the [0, 1] range using the ideal point
        and the intercepts.

        For every objective i the formula is:

            f_hat_i = (f_i - ideal_i) / (intercepts_i - ideal_i)

        Values outside [0, 1] are clipped. This happens for dominated
        solutions or after a fallback intercept.

        Parameters
        ----------
        fitness : numpy.ndarray
            The fitness array to normalize.
        ideal_point : numpy.ndarray
            The ideal point.
        intercepts : numpy.ndarray
            The intercept point returned by compute_intercepts.

        Returns
        -------
        normalised : numpy.ndarray
            Fitness scaled to the unit hypercube, same shape as the input.
        """
        fitness = numpy.asarray(fitness, dtype=float)
        ideal_point = numpy.asarray(ideal_point, dtype=float)
        intercepts = numpy.asarray(intercepts, dtype=float)
        denominator = intercepts - ideal_point
        # Under PyGAD-max, intercepts sit below ideal so denominator is
        # negative. Replace near-zero entries with a tiny negative value
        # to keep the sign correct and avoid divide-by-zero.
        safe_denominator = numpy.where(denominator > -INTERCEPT_NEAR_ZERO,
                                       -INTERCEPT_NEAR_ZERO,
                                       denominator)
        raw = (fitness - ideal_point) / safe_denominator
        return numpy.clip(raw, 0.0, 1.0)

    def associate_to_reference_points(self, normalised, reference_points):
        """
        For every normalised solution, find the reference line it is
        closest to and the perpendicular distance to that line.

        The reference line for reference point z is the ray from the
        origin through z. The perpendicular distance from a point x to
        that line is:

            d(x, z) = || x - (x . z_hat) * z_hat ||

        where z_hat = z / || z ||.

        Ties on the minimum distance go to the lower reference index
        because numpy.argmin returns the first occurrence.

        Parameters
        ----------
        normalised : numpy.ndarray
            Normalised fitness, one row per solution.
        reference_points : numpy.ndarray
            The structured reference grid, one row per point.

        Returns
        -------
        nearest : numpy.ndarray
            A 1D array of length n_solutions. Each entry is the index of
            the nearest reference point for that solution.
        nearest_distance : numpy.ndarray
            A 1D array of length n_solutions with the perpendicular
            distance to the nearest reference line.
        """
        normalised = numpy.asarray(normalised, dtype=float)
        reference_points = numpy.asarray(reference_points, dtype=float)
        # Turn every reference point into a unit direction vector once.
        unit_directions = reference_points / numpy.linalg.norm(reference_points,
                                                               axis=1,
                                                               keepdims=True)
        # Dot products of every solution with every reference direction.
        # Shape: (n_solutions, n_references).
        dot_products = normalised @ unit_directions.T
        # Project each solution onto each reference line.
        # Shape: (n_solutions, n_references, n_objectives).
        projections = dot_products[:, :, None] * unit_directions[None, :, :]
        # Perpendicular component is what is left after subtracting the
        # projection from the original solution.
        perpendicular = normalised[:, None, :] - projections
        distances = numpy.linalg.norm(perpendicular, axis=2)
        nearest = numpy.argmin(distances, axis=1)
        nearest_distance = distances[numpy.arange(len(normalised)), nearest]
        return nearest, nearest_distance

    def niching_select(self, fl_indices, fl_assoc, fl_dist, accepted_assoc,
                       num_reference_points, K):
        """
        Pick K survivors from the critical front Fl using the niching
        rules. The result preserves diversity across reference points.

        The niche count rho_j is the number of already accepted solutions
        associated with reference point j. The procedure repeats K times:
          1. Pick the reference point with the smallest niche count that
             still has at least one Fl candidate attached.
          2. If that reference point has rho_j = 0, pick the Fl candidate
             closest to its reference line.
          3. If rho_j > 0, pick one of its Fl candidates at random.
          4. Add the selected candidate to the survivor list, increase
             rho_j by 1, and remove the candidate from Fl.

        Ties on minimum rho_j go to the lower reference index.

        Parameters
        ----------
        fl_indices : list[int]
            Population indices of the candidates in the critical front.
        fl_assoc : numpy.ndarray
            Reference index each Fl candidate is associated with.
        fl_dist : numpy.ndarray
            Perpendicular distance from each Fl candidate to its
            reference line.
        accepted_assoc : numpy.ndarray
            Reference index each already-accepted solution is associated
            with. Used to seed the niche counts.
        num_reference_points : int
            Total number of reference points.
        K : int
            Number of survivors to pick from Fl.

        Returns
        -------
        picked : list[int]
            Population indices of the selected survivors, in selection
            order. Length is at most K.
        """
        rho = numpy.zeros(num_reference_points, dtype=int)
        for ref in accepted_assoc:
            rho[ref] += 1
        remaining_positions = list(range(len(fl_indices)))
        picked = []
        while len(picked) < K and remaining_positions:
            target_ref = _pick_target_reference_point(rho, fl_assoc, remaining_positions)
            if target_ref is None:
                break
            candidates_at_target = [pos for pos in remaining_positions
                                    if fl_assoc[pos] == target_ref]
            chosen_pos = _pick_candidate_at_reference(candidates_at_target,
                                                     fl_dist,
                                                     rho[target_ref])
            picked.append(fl_indices[chosen_pos])
            rho[target_ref] += 1
            remaining_positions.remove(chosen_pos)
        return picked

    def nsga3_selection(self, fitness, num_parents):
        """
        Select num_parents parents from the current population using
        NSGA-III. Solutions are first sorted into Pareto fronts. Whole
        fronts are accepted in order until the next front would overflow
        the requested parent count; that front becomes the critical front
        Fl. Survivors from Fl are picked by niching against the structured
        reference points stored on the GA instance.

        Parameters
        ----------
        fitness : numpy.ndarray
            Fitness values for the entire population. Must be
            multi-objective (each row is a vector of M values).
        num_parents : int
            Number of parents to select.

        Returns
        -------
        parents : numpy.ndarray
            Selected parent solutions copied from self.population.
        parents_indices : numpy.ndarray
            Indices of the selected parents inside self.population.
        """
        _validate_multi_objective_fitness(fitness, self.supported_int_float_types,
                                          'nsga3_selection')
        pareto_fronts, _ = self.non_dominated_sorting(fitness)
        self.pareto_fronts = pareto_fronts.copy()

        accepted_indices, fl_indices = _accumulate_fronts(pareto_fronts, num_parents)
        if fl_indices:
            # Need to pick K survivors from the critical front Fl.
            picked = self._pick_critical_front_survivors(accepted_indices,
                                                         fl_indices,
                                                         fitness,
                                                         num_parents - len(accepted_indices))
            final_indices = accepted_indices + picked
        else:
            # The accepted fronts already fit exactly; no niching needed.
            final_indices = accepted_indices

        return self._build_parents(final_indices, num_parents)

    def _pick_critical_front_survivors(self, accepted_indices, fl_indices,
                                       fitness, K):
        """
        Run the NSGA-III normalization and niching steps on the pool
        P_next U Fl, then ask niching_select for K survivors from Fl.

        The ideal point, extreme points, intercepts and normalized values
        are all computed on the combined pool (accepted plus critical
        front) because that is what the NSGA-III paper specifies.
        """
        st_indices = accepted_indices + fl_indices
        st_fitness = numpy.array([fitness[i] for i in st_indices], dtype=float)
        ideal_point = self.compute_ideal_point(st_fitness)
        extremes = self.find_extreme_points(st_fitness, ideal_point)
        intercepts = self.compute_intercepts(extremes, ideal_point, st_fitness)
        normalised = self.normalise_fitness(st_fitness, ideal_point, intercepts)
        assoc, dist = self.associate_to_reference_points(normalised,
                                                         self.nsga3_reference_points)
        # The first |accepted_indices| rows of st_fitness belong to the
        # accepted set; the rest are the Fl candidates.
        split = len(accepted_indices)
        return self.niching_select(fl_indices=fl_indices,
                                   fl_assoc=assoc[split:],
                                   fl_dist=dist[split:],
                                   accepted_assoc=assoc[:split],
                                   num_reference_points=len(self.nsga3_reference_points),
                                   K=K)

    def _build_parents(self, final_indices, num_parents):
        """
        Copy the chosen solutions out of self.population into a new
        parents array of the right dtype, and return it together with the
        index array.
        """
        if self.gene_type_single:
            parents = numpy.empty((num_parents, self.population.shape[1]),
                                  dtype=self.gene_type[0])
        else:
            parents = numpy.empty((num_parents, self.population.shape[1]),
                                  dtype=object)
        for slot, idx in enumerate(final_indices):
            parents[slot, :] = self.population[idx, :].copy()
        return parents, numpy.array(final_indices)

    def tournament_selection_nsga3(self, fitness, num_parents):
        """
        Select num_parents parents using K-tournament where the within-
        front comparison is based on NSGA-III niching.

        The full population is sorted into Pareto fronts and normalised
        once at the start. For each parent slot:
          1. Pick K_tournament solutions at random.
          2. Keep only the ones in the best (lowest) Pareto front.
          3. If more than one is left, the winner is the solution whose
             reference point has the smallest niche count. Ties on niche
             count go to the smaller perpendicular distance.

        Parameters
        ----------
        fitness : numpy.ndarray
            Fitness values for the entire population. Must be
            multi-objective.
        num_parents : int
            Number of parents to select.

        Returns
        -------
        parents : numpy.ndarray
            Selected parent solutions.
        parents_indices : numpy.ndarray
            Indices of the selected parents inside self.population.
        """
        _validate_multi_objective_fitness(fitness, self.supported_int_float_types,
                                          'tournament_selection_nsga3')
        pareto_fronts, solutions_fronts_indices = self.non_dominated_sorting(fitness)
        self.pareto_fronts = pareto_fronts.copy()

        # Convert the fitness rows to a clean 2D float array.
        fitness_matrix = numpy.array([list(row) for row in fitness], dtype=float)
        ideal_point = self.compute_ideal_point(fitness_matrix)
        extremes = self.find_extreme_points(fitness_matrix, ideal_point)
        intercepts = self.compute_intercepts(extremes, ideal_point, fitness_matrix)
        normalised = self.normalise_fitness(fitness_matrix, ideal_point, intercepts)
        assoc, dist = self.associate_to_reference_points(normalised,
                                                         self.nsga3_reference_points)
        # Niche count is the number of population solutions attached to
        # each reference point.
        rho = numpy.bincount(assoc, minlength=len(self.nsga3_reference_points))

        rand_indices = numpy.random.randint(low=0,
                                            high=len(solutions_fronts_indices),
                                            size=(num_parents, self.K_tournament))
        parents_indices = [self._pick_tournament_winner(rand_indices[slot],
                                                        solutions_fronts_indices,
                                                        assoc, dist, rho)
                           for slot in range(num_parents)]
        return self._build_parents(parents_indices, num_parents)

    def _pick_tournament_winner(self, competitor_indices, fronts_indices,
                                assoc, dist, rho):
        """
        Pick the best solution among the K-tournament competitors. The
        best front index wins first; ties are broken by lower niche count,
        then by smaller perpendicular distance.
        """
        best_front = fronts_indices[competitor_indices].min()
        finalists = competitor_indices[fronts_indices[competitor_indices] == best_front]
        if len(finalists) == 1:
            return int(finalists[0])
        finalist_rho = rho[assoc[finalists]]
        finalist_dist = dist[finalists]
        # lexsort sorts by the last key first, so this orders by rho first
        # and breaks ties by distance.
        ordering = numpy.lexsort((finalist_dist, finalist_rho))
        return int(finalists[ordering[0]])

    def _bootstrap_nsga3_reference_points(self):
        """
        Build the reference-point grid once, right after the first
        fitness evaluation. The number of objectives M is read from the
        length of the first fitness vector.

        If sol_per_pop is smaller than the number of reference points,
        grow the population to match and re-evaluate fitness so the GA
        loop can carry on with a valid population.
        """
        num_objectives = len(self.last_generation_fitness[0])
        self.nsga3_reference_points = self.generate_reference_points(
            num_objectives, self.nsga3_num_divisions)
        required_size = len(self.nsga3_reference_points)
        if self.sol_per_pop < required_size:
            self._grow_population_for_nsga3(required_size, num_objectives)

    def _grow_population_for_nsga3(self, required_size, num_objectives):
        """
        Append random solutions to self.population until the population
        size equals required_size, then re-evaluate fitness. Also refresh
        num_offspring so the next generation produces the right count.
        """
        original_size = self.sol_per_pop
        if not self.suppress_warnings:
            warnings.warn(
                f"sol_per_pop ({original_size}) is smaller than the number of "
                f"NSGA-III reference points ({required_size}) for M={num_objectives} "
                f"objectives and nsga3_num_divisions={self.nsga3_num_divisions}. "
                f"Growing the population to {required_size} random solutions "
                f"and re-evaluating fitness."
            )
        extra = self._generate_extra_random_solutions(required_size - original_size)
        self.population = numpy.vstack([self.population, extra])
        self.sol_per_pop = required_size
        self.pop_size = (required_size, self.num_genes)
        # Shared helper on the Validation mixin keeps the rule in one place.
        self._refresh_num_offspring()
        self.last_generation_fitness = self.cal_pop_fitness()

    def _generate_extra_random_solutions(self, count):
        """
        Build `count` random solutions using the same gene-by-gene logic
        the GA uses for the initial population.
        """
        extra = numpy.empty((count, self.num_genes), dtype=object)
        for sol_idx in range(count):
            for gene_idx in range(self.num_genes):
                extra[sol_idx, gene_idx] = self._generate_single_random_gene(
                    gene_idx, extra[sol_idx])
        return self.change_population_dtype_and_round(extra)

    def _generate_single_random_gene(self, gene_idx, partial_solution):
        """
        Pick a single random gene value for index `gene_idx`. Uses the
        gene-space sampler when gene_space is set; otherwise samples from
        the init range for that gene.
        """
        if self.gene_space is None:
            range_min, range_max = self.get_initial_population_range(gene_index=gene_idx)
            return self.generate_gene_value_randomly(range_min=range_min,
                                                    range_max=range_max,
                                                    gene_idx=gene_idx,
                                                    mutation_by_replacement=True,
                                                    gene_value=None,
                                                    sample_size=1,
                                                    step=1)
        return self.generate_gene_value_from_space(gene_idx=gene_idx,
                                                   mutation_by_replacement=True,
                                                   gene_value=None,
                                                   solution=partial_solution,
                                                   sample_size=1)


def _pick_target_reference_point(rho, fl_assoc, remaining_positions):
    """
    Among the reference points that still have at least one Fl candidate
    attached, pick the one with the smallest niche count. Break ties by
    the lower reference index.
    """
    candidate_refs = {int(fl_assoc[pos]) for pos in remaining_positions}
    if not candidate_refs:
        return None
    min_rho = min(rho[ref] for ref in candidate_refs)
    return min(ref for ref in candidate_refs if rho[ref] == min_rho)


def _pick_candidate_at_reference(candidates_at_target, fl_dist, rho_at_target):
    """
    Choose one Fl candidate at the given reference point. If the niche
    count is 0 (empty niche), pick the closest candidate. Otherwise pick
    a candidate at random.
    """
    if rho_at_target == 0:
        return min(candidates_at_target, key=lambda pos: fl_dist[pos])
    return candidates_at_target[numpy.random.randint(len(candidates_at_target))]


def _enumerate_compositions(num_objectives, num_divisions):
    """
    Yield every non-negative integer tuple of length num_objectives that
    sums to num_divisions. Used by generate_reference_points to build the
    Das-Dennis grid.
    """
    if num_objectives == 1:
        yield [num_divisions]
        return
    # 13
    for first in range(num_divisions + 1):
        for rest in _enumerate_compositions(num_objectives - 1, num_divisions - first):
            yield [first] + rest


def _validate_multi_objective_fitness(fitness, supported_int_float_types, method_name):
    """
    Raise an error if the first fitness value is a scalar (which means
    the problem is single-objective and NSGA-III cannot be applied) or if
    it is some other unsupported type.
    """
    if type(fitness[0]) in supported_int_float_types:
        raise ValueError(
            f"{method_name} requires a multi-objective fitness function "
            f"(an iterable per solution), but the first fitness value "
            f"({fitness[0]!r}) has scalar type {type(fitness[0]).__name__}."
        )
    if type(fitness[0]) not in (list, tuple, numpy.ndarray):
        raise TypeError(
            f"{method_name} expects each fitness value to be a list, tuple, "
            f"or numpy.ndarray, but the first fitness value has type "
            f"{type(fitness[0]).__name__}."
        )


def _accumulate_fronts(pareto_fronts, num_parents):
    """
    Walk the Pareto fronts in order and add each whole front to the
    accepted list while the running total stays at or below num_parents.
    The first front that would overflow becomes the critical front Fl.

    Returns a pair (accepted_indices, fl_indices). When the accepted set
    fits exactly into num_parents, fl_indices is empty.
    """
    accepted_indices = []
    fl_indices = []
    for front in pareto_fronts:
        front_solution_indices = front[:, 0].astype(int).tolist()
        if len(accepted_indices) + len(front_solution_indices) <= num_parents:
            accepted_indices.extend(front_solution_indices)
            if len(accepted_indices) == num_parents:
                break
        else:
            fl_indices = front_solution_indices
            break
    return accepted_indices, fl_indices
