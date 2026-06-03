"""
NSGA-III algorithm primitives.

This module contains the math used by NSGA-III: reference point
generation, ideal point, extreme points, hyperplane intercepts,
normalization, association to reference lines, and niching. The
selection routines (``nsga3_selection`` and
``tournament_selection_nsga3``) live in ``parent_selection.py`` and the
engine-time bootstrap helpers live in ``engine.py``.
"""

import numpy


# Weight used to amplify the off-axis terms in the ASF score when looking
# for the extreme point of each objective. A very small weight makes any
# deviation on a non-target axis huge so it dominates the score.
NSGA3_ASF_EPSILON = 1e-6

# Numbers smaller than this are treated as zero when we check for a
# singular linear system or a collapsed axis range.
NSGA3_INTERCEPT_NEAR_ZERO = 1e-12


class NSGA3:

    def __init__(self):
        pass

    def nsga3_generate_reference_points(self, num_objectives, num_divisions):
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
        compositions = list(_nsga3_enumerate_compositions(num_objectives, num_divisions))
        as_array = numpy.array(compositions, dtype=float)
        return as_array / num_divisions

    def nsga3_compute_ideal_point(self, fitness):
        """
        Return the ideal point: the best fitness value for each objective
        across the input fitness rows. PyGAD maximizes, so the best value
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

    def nsga3_find_extreme_points(self, fitness, ideal_point,
                                  epsilon=NSGA3_ASF_EPSILON):
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

    def nsga3_compute_intercepts(self, extreme_points, ideal_point,
                                 fallback_fitness):
        """
        Fit a hyperplane through the M extreme points and return the
        intercept point on each axis. The result is the point we use to
        scale every objective to the [0, 1] range during normalization.

        The NSGA-III paper defines the intercept as the point that
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
             below NSGA3_INTERCEPT_NEAR_ZERO after clipping, replace that
             intercept with the worst observed value so the normalization
             denominator stays non-zero.

        Parameters
        ----------
        extreme_points : numpy.ndarray
            The M extreme points returned by nsga3_find_extreme_points.
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
        if numpy.any(numpy.abs(coefficients) < NSGA3_INTERCEPT_NEAR_ZERO):
            return worst_per_objective
        intercepts = ideal_point + 1.0 / coefficients
        # Under maximization a valid intercept sits strictly below the
        # ideal. If it does not, the normalization denominator would flip
        # sign and produce nonsense values.
        if numpy.any(intercepts >= ideal_point - NSGA3_INTERCEPT_NEAR_ZERO):
            return worst_per_objective
        # Cap the intercept at the worst observed value so we never
        # extrapolate the hyperplane past the real data range.
        overshoot = intercepts < worst_per_objective
        intercepts = numpy.where(overshoot, worst_per_objective, intercepts)
        # If capping leaves the gap |intercept - ideal| too small, reset
        # that axis to the worst observed value.
        collapsed = numpy.abs(intercepts - ideal_point) < NSGA3_INTERCEPT_NEAR_ZERO
        intercepts = numpy.where(collapsed, worst_per_objective, intercepts)
        return intercepts

    def nsga3_normalize_fitness(self, fitness, ideal_point, intercepts):
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
            The intercept point returned by nsga3_compute_intercepts.

        Returns
        -------
        normalized : numpy.ndarray
            Fitness scaled to the unit hypercube, same shape as the input.
        """
        fitness = numpy.asarray(fitness, dtype=float)
        ideal_point = numpy.asarray(ideal_point, dtype=float)
        intercepts = numpy.asarray(intercepts, dtype=float)
        denominator = intercepts - ideal_point
        # Under PyGAD-max, intercepts sit below ideal so denominator is
        # negative. Replace near-zero entries with a tiny negative value
        # to keep the sign correct and avoid divide-by-zero.
        safe_denominator = numpy.where(denominator > -NSGA3_INTERCEPT_NEAR_ZERO,
                                       -NSGA3_INTERCEPT_NEAR_ZERO,
                                       denominator)
        raw = (fitness - ideal_point) / safe_denominator
        return numpy.clip(raw, 0.0, 1.0)

    def nsga3_associate_to_reference_points(self, normalized, reference_points):
        """
        For every normalized solution, find the reference line it is
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
        normalized : numpy.ndarray
            Normalized fitness, one row per solution.
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
        normalized = numpy.asarray(normalized, dtype=float)
        reference_points = numpy.asarray(reference_points, dtype=float)
        # Turn every reference point into a unit direction vector once.
        unit_directions = reference_points / numpy.linalg.norm(reference_points,
                                                               axis=1,
                                                               keepdims=True)
        # Dot products of every solution with every reference direction.
        # Shape: (n_solutions, n_references).
        dot_products = normalized @ unit_directions.T
        # Project each solution onto each reference line.
        # Shape: (n_solutions, n_references, n_objectives).
        projections = dot_products[:, :, None] * unit_directions[None, :, :]
        # Perpendicular component is what is left after subtracting the
        # projection from the original solution.
        perpendicular = normalized[:, None, :] - projections
        distances = numpy.linalg.norm(perpendicular, axis=2)
        nearest = numpy.argmin(distances, axis=1)
        nearest_distance = distances[numpy.arange(len(normalized)), nearest]
        return nearest, nearest_distance

    def nsga3_niching_select(self,
                             critical_front_indices,
                             critical_front_associations,
                             critical_front_distances,
                             accepted_associations,
                             num_reference_points,
                             num_to_select):
        """
        Pick ``num_to_select`` survivors from the critical front using
        the niching rules. The result preserves diversity across
        reference points.

        The niche count of a reference point j is the number of already
        accepted solutions associated with j. The procedure repeats
        ``num_to_select`` times:
          1. Pick the reference point with the smallest niche count that
             still has at least one critical-front candidate attached.
          2. If that reference point has niche count zero, pick the
             critical-front candidate closest to its reference line.
          3. If the niche count is positive, pick one of its
             critical-front candidates at random.
          4. Add the selected candidate to the survivor list, increase
             the niche count by 1, and remove the candidate from the
             critical front.

        Ties on minimum niche count go to the lower reference index.

        Parameters
        ----------
        critical_front_indices : list[int]
            Population indices of the candidates in the critical front.
        critical_front_associations : numpy.ndarray
            Reference index each critical-front candidate is associated
            with.
        critical_front_distances : numpy.ndarray
            Perpendicular distance from each critical-front candidate to
            its reference line.
        accepted_associations : numpy.ndarray
            Reference index each already-accepted solution is associated
            with. Used to seed the niche counts.
        num_reference_points : int
            Total number of reference points.
        num_to_select : int
            Number of survivors to pick from the critical front.

        Returns
        -------
        picked : list[int]
            Population indices of the selected survivors, in selection
            order. Length is at most ``num_to_select``.
        """
        niche_counts = numpy.zeros(num_reference_points, dtype=int)
        for reference_index in accepted_associations:
            niche_counts[reference_index] += 1
        remaining_positions = list(range(len(critical_front_indices)))
        picked = []
        while len(picked) < num_to_select and remaining_positions:
            target_reference_index = _nsga3_pick_target_reference_point(
                niche_counts, critical_front_associations, remaining_positions)
            if target_reference_index is None:
                break
            candidates_at_target = [
                position for position in remaining_positions
                if critical_front_associations[position] == target_reference_index
            ]
            chosen_position = _nsga3_pick_candidate_at_reference(
                candidates_at_target,
                critical_front_distances,
                niche_counts[target_reference_index],
            )
            picked.append(critical_front_indices[chosen_position])
            niche_counts[target_reference_index] += 1
            remaining_positions.remove(chosen_position)
        return picked


def _nsga3_pick_target_reference_point(niche_counts,
                                       critical_front_associations,
                                       remaining_positions):
    """
    Among the reference points that still have at least one critical-
    front candidate attached, pick the one with the smallest niche
    count. Break ties by the lower reference index.
    """
    candidate_references = {
        int(critical_front_associations[position])
        for position in remaining_positions
    }
    if not candidate_references:
        return None
    min_niche_count = min(niche_counts[reference]
                          for reference in candidate_references)
    return min(reference for reference in candidate_references
               if niche_counts[reference] == min_niche_count)


def _nsga3_pick_candidate_at_reference(candidates_at_target,
                                       critical_front_distances,
                                       niche_count_at_target):
    """
    Choose one critical-front candidate at the given reference point. If
    the niche count is 0 (empty niche), pick the closest candidate.
    Otherwise pick a candidate at random.
    """
    if niche_count_at_target == 0:
        return min(candidates_at_target,
                   key=lambda position: critical_front_distances[position])
    return candidates_at_target[
        numpy.random.randint(len(candidates_at_target))
    ]


def _nsga3_enumerate_compositions(num_objectives, num_divisions):
    """
    Yield every non-negative integer tuple of length num_objectives that
    sums to num_divisions. Used by nsga3_generate_reference_points to
    build the Das-Dennis grid.
    """
    if num_objectives == 1:
        yield [num_divisions]
        return
    for first in range(num_divisions + 1):
        for rest in _nsga3_enumerate_compositions(num_objectives - 1,
                                                  num_divisions - first):
            yield [first] + rest
