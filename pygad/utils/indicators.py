"""
Quality indicators for multi-objective optimization problems.

The module has four common indicators that the user can use to check
the quality of a Pareto front built by PyGAD:

1. hypervolume: volume of the objective space dominated by the front.
2. inverted_generational_distance: mean distance from each reference
   point to its nearest approximation point.
3. generational_distance: mean distance from each approximation point
   to its nearest reference point.
4. spacing: how evenly the approximation points are spread.

All functions expect fitness values in PyGAD's maximization format
(higher means better). The reference point passed to hypervolume must
be worse than every solution on every axis.
"""

import numpy


def _to_min_fitness(fitness):
    """
    Negate the fitness so the rest of the module can use the
    minimization form that the standard hypervolume algorithm uses.
    """
    return -numpy.asarray(fitness, dtype=float)


def _drop_dominated_under_min(points):
    """
    Return the rows of points that are not dominated by any other row
    under minimization. The order of the input rows is kept.
    """
    n = points.shape[0]
    keep = numpy.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if numpy.all(points[j] <= points[i]) and numpy.any(points[j] < points[i]):
                keep[i] = False
                break
    return points[keep]


def _inclusive_hv(point, reference_point):
    """
    Volume of the box whose lower corner is the point and whose upper
    corner is the reference point. Returns 0 if the point does not sit
    below the reference on every axis.
    """
    diff = reference_point - point
    if numpy.any(diff <= 0):
        return 0.0
    return float(numpy.prod(diff))


def _wfg_exclusive_hv(point, others, reference_point):
    """
    Volume dominated only by the given point and not by any of the
    other points. Uses the WFG recurrence:

        exclusive(p, others) = inclusive(p) - hv(limit(p, others))

    where limit(p, q) pushes every other point up to the corner of p.
    """
    base = _inclusive_hv(point, reference_point)
    if base == 0.0 or len(others) == 0:
        return base
    limited = numpy.maximum(others, point)
    limited = _drop_dominated_under_min(limited)
    return base - _hv_under_min(limited, reference_point)


def _hv_under_min(points, reference_point):
    """
    Hypervolume of points (under minimization) bounded by the
    reference point. Uses the WFG recurrence which is fast enough
    for the population sizes PyGAD usually runs.
    """
    if len(points) == 0:
        return 0.0
    # Sort by the last objective so the recursion stays shallow.
    order = numpy.argsort(points[:, -1])
    sorted_points = points[order]
    total = 0.0
    for i, point in enumerate(sorted_points):
        others = sorted_points[i + 1:]
        total += _wfg_exclusive_hv(point, others, reference_point)
    return total


def hypervolume(fitness, reference_point):
    """
    Return the hypervolume of the Pareto front made from every row in
    fitness. The reference point is the worst case on every objective.
    Under PyGAD's maximization format the reference values must be
    smaller than every fitness value.

    The function negates the fitness and the reference internally so
    it can use a standard hypervolume algorithm written for
    minimization. The returned volume is positive.

    Parameters
    ----------
    fitness : numpy.ndarray
        A 2D array of fitness values of shape
        (num_solutions, num_objectives).
    reference_point : array-like
        A 1D array of length num_objectives. Every entry must be
        smaller than the matching column min of fitness.

    Returns
    -------
    hv : float
        The hypervolume value. A bigger value means better coverage
        of the objective space.

    Raises
    ------
    ValueError
        If fitness is not 2D, if the reference point has the wrong
        shape, or if the reference point is not worse than every
        solution on every axis.
    """
    fitness_arr = numpy.asarray(fitness, dtype=float)
    reference_arr = numpy.asarray(reference_point, dtype=float)
    if fitness_arr.ndim != 2:
        raise ValueError(
            f"fitness must be a 2D array, but got shape {fitness_arr.shape}.")
    if reference_arr.shape != (fitness_arr.shape[1],):
        raise ValueError(
            f"reference_point must have shape ({fitness_arr.shape[1]},), "
            f"but got {reference_arr.shape}.")
    # Under PyGAD max, every solution must beat the reference on every axis.
    if numpy.any(fitness_arr.min(axis=0) <= reference_arr):
        raise ValueError(
            "reference_point must be smaller than every solution on every "
            "objective. Pick a reference smaller than the column-wise "
            "minimum of the fitness matrix.")
    min_fitness = _to_min_fitness(fitness_arr)
    min_reference = -reference_arr
    front = _drop_dominated_under_min(min_fitness)
    return _hv_under_min(front, min_reference)


def _euclidean_distance_matrix(a, b):
    """
    Return a matrix d of shape (len(a), len(b)) where d[i, j] is the
    Euclidean distance between row i of a and row j of b.
    """
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return numpy.sqrt((diff * diff).sum(axis=2))


def inverted_generational_distance(fitness, reference_front):
    """
    Return the Inverted Generational Distance (IGD): the mean
    Euclidean distance from each reference-front point to its nearest
    point in the approximation front.

    A smaller value is better. IGD reports both convergence (how far
    the approximation is from the true front) and diversity (how much
    of the true front is covered).

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front of shape (num_solutions, num_objectives)
        in PyGAD's maximization format.
    reference_front : numpy.ndarray
        Reference front of shape
        (num_reference_points, num_objectives) in the same format.

    Returns
    -------
    igd : float
        The IGD value.
    """
    distance_matrix = _euclidean_distance_matrix(reference_front, fitness)
    return float(distance_matrix.min(axis=1).mean())


def generational_distance(fitness, reference_front):
    """
    Return the Generational Distance (GD): the mean Euclidean distance
    from each approximation point to its nearest reference point.

    A smaller value is better. GD only measures convergence and not
    diversity.

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front of shape (num_solutions, num_objectives)
        in PyGAD's maximization format.
    reference_front : numpy.ndarray
        Reference front of shape
        (num_reference_points, num_objectives) in the same format.

    Returns
    -------
    gd : float
        The GD value.
    """
    distance_matrix = _euclidean_distance_matrix(fitness, reference_front)
    return float(distance_matrix.min(axis=1).mean())


def spacing(fitness):
    """
    Return the spacing metric: the standard deviation of the distance
    from each solution to its nearest neighbour. A smaller value means
    the solutions are spread more evenly.

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front of shape (num_solutions, num_objectives)
        in PyGAD's maximization format.

    Returns
    -------
    spacing_value : float
        The spacing metric. Returns 0.0 when fewer than two solutions
        are given.
    """
    fitness_arr = numpy.asarray(fitness, dtype=float)
    if fitness_arr.shape[0] < 2:
        return 0.0
    distance_matrix = _euclidean_distance_matrix(fitness_arr, fitness_arr)
    # Skip the zero distance from each point to itself.
    numpy.fill_diagonal(distance_matrix, numpy.inf)
    nearest_neighbour_distance = distance_matrix.min(axis=1)
    mean_distance = nearest_neighbour_distance.mean()
    return float(numpy.sqrt(
        ((nearest_neighbour_distance - mean_distance) ** 2).mean()))
