"""
Quality indicators for multi-objective optimisation.

Four indicators to measure the quality of a Pareto front built by
PyGAD:

1. hypervolume: volume of the objective space dominated by the front.
2. inverted_generational_distance: mean distance from each reference
   point to its nearest approximation point.
3. generational_distance: mean distance from each approximation
   point to its nearest reference point.
4. spacing: how evenly the approximation points are spread.

All functions take fitness values in PyGAD's maximisation format
(higher is better). The reference point for hypervolume must be
worse than every solution on every axis.
"""

import numpy


def _to_min_fitness(fitness):
    """Negate the fitness to switch to minimisation."""
    return -numpy.asarray(fitness, dtype=float)


def _drop_dominated_under_min(points):
    """Drop dominated rows under minimisation. Keeps input order."""
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
    """Volume of the box between point and reference. 0 if point does not sit below the reference on every axis."""
    diff = reference_point - point
    if numpy.any(diff <= 0):
        return 0.0
    return float(numpy.prod(diff))


def _wfg_exclusive_hv(point, others, reference_point):
    """
    WFG recurrence:
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
    """WFG hypervolume under minimisation. Fast enough for typical PyGAD populations."""
    if len(points) == 0:
        return 0.0
    # Sort by the last objective to keep recursion shallow.
    order = numpy.argsort(points[:, -1])
    sorted_points = points[order]
    total = 0.0
    for i, point in enumerate(sorted_points):
        others = sorted_points[i + 1:]
        total += _wfg_exclusive_hv(point, others, reference_point)
    return total


def hypervolume(fitness, reference_point):
    """
    Hypervolume of the Pareto front built from every row in fitness.

    The reference is the worst case on every axis. Under PyGAD-max it
    must be smaller than every fitness value. The function flips the
    sign internally so the WFG algorithm (written for minimisation)
    can be reused. The returned value is positive; bigger is better.

    Parameters
    ----------
    fitness : numpy.ndarray
        2D array of shape (num_solutions, num_objectives).
    reference_point : array-like
        1D array of length num_objectives, smaller than every entry
        in the matching fitness column.

    Returns
    -------
    hv : float

    Raises
    ------
    ValueError
        If fitness is not 2D, if reference_point has the wrong shape,
        or if reference_point is not smaller than every solution on
        every axis.
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
    """Pairwise Euclidean distances; d[i, j] = ||a[i] - b[j]||."""
    a = numpy.asarray(a, dtype=float)
    b = numpy.asarray(b, dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return numpy.sqrt((diff * diff).sum(axis=2))


def inverted_generational_distance(fitness, reference_front):
    """
    Inverted Generational Distance (IGD): mean distance from each
    reference point to its nearest approximation point. Smaller is
    better; reports both convergence and diversity.

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front, shape (num_solutions, num_objectives),
        in PyGAD's maximisation format.
    reference_front : numpy.ndarray
        Reference front, shape (num_reference_points, num_objectives),
        in the same format.

    Returns
    -------
    igd : float
    """
    distance_matrix = _euclidean_distance_matrix(reference_front, fitness)
    return float(distance_matrix.min(axis=1).mean())


def generational_distance(fitness, reference_front):
    """
    Generational Distance (GD): mean distance from each approximation
    point to its nearest reference point. Smaller is better; measures
    convergence only.

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front, shape (num_solutions, num_objectives),
        in PyGAD's maximisation format.
    reference_front : numpy.ndarray
        Reference front, shape (num_reference_points, num_objectives),
        in the same format.

    Returns
    -------
    gd : float
    """
    distance_matrix = _euclidean_distance_matrix(fitness, reference_front)
    return float(distance_matrix.min(axis=1).mean())


def spacing(fitness):
    """
    Spacing: standard deviation of each solution's nearest-neighbour
    distance. Smaller means the solutions are more evenly spread.

    Returns 0.0 when fewer than two solutions are given so the caller
    does not have to special-case it.

    Parameters
    ----------
    fitness : numpy.ndarray
        Approximation front, shape (num_solutions, num_objectives),
        in PyGAD's maximisation format.

    Returns
    -------
    spacing_value : float
    """
    fitness_arr = numpy.asarray(fitness, dtype=float)
    if fitness_arr.shape[0] < 2:
        return 0.0
    distance_matrix = _euclidean_distance_matrix(fitness_arr, fitness_arr)
    numpy.fill_diagonal(distance_matrix, numpy.inf)
    nearest_neighbour_distance = distance_matrix.min(axis=1)
    mean_distance = nearest_neighbour_distance.mean()
    return float(numpy.sqrt(
        ((nearest_neighbour_distance - mean_distance) ** 2).mean()))
