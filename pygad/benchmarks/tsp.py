"""
Travelling Salesman Problem (TSP) benchmark.

A solution is a permutation of city indices. Fitness is the
negative tour length so PyGAD can maximize it.

Build the problem from a 2D array of coordinates or from a
precomputed distance matrix. The class exposes gene_space,
gene_type, and allow_duplicate_genes so it plugs into PyGAD as is.
"""

import numpy


class TSP:
    """0/1 closed-tour TSP."""
    num_objectives = 1
    gene_type = int
    allow_duplicate_genes = False

    def __init__(self, coordinates=None, distance_matrix=None):
        if (coordinates is None) == (distance_matrix is None):
            raise ValueError("Pass exactly one of coordinates or distance_matrix.")
        if coordinates is not None:
            coordinates = numpy.asarray(coordinates, dtype=float)
            if coordinates.ndim != 2:
                raise ValueError(
                    f"coordinates must be a 2D array, but got shape {coordinates.shape}.")
            if coordinates.shape[0] < 2:
                raise ValueError(
                    f"coordinates must have at least 2 rows, but got {coordinates.shape[0]}.")
            self.coordinates = coordinates
            self.distance_matrix = self._build_distance_matrix(coordinates)
        else:
            distance_matrix = numpy.asarray(distance_matrix, dtype=float)
            if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
                raise ValueError(
                    f"distance_matrix must be square, but got shape {distance_matrix.shape}.")
            if distance_matrix.shape[0] < 2:
                raise ValueError(
                    f"distance_matrix must have at least 2 rows, but got {distance_matrix.shape[0]}.")
            if numpy.any(distance_matrix < 0):
                raise ValueError("distance_matrix entries must be non-negative.")
            self.coordinates = None
            self.distance_matrix = distance_matrix
        self.num_genes = int(self.distance_matrix.shape[0])
        self.gene_space = list(range(self.num_genes))

    @staticmethod
    def _build_distance_matrix(coordinates):
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        return numpy.sqrt((diff * diff).sum(axis=2))

    def tour_length(self, tour):
        """Length of the closed tour. Last leg goes from the last city back to the first."""
        tour = numpy.asarray(tour, dtype=int)
        next_city = numpy.roll(tour, -1)
        return float(self.distance_matrix[tour, next_city].sum())

    def __call__(self, ga, solution, sol_idx):
        tour = numpy.asarray(solution, dtype=int)
        if tour.shape[0] != self.num_genes:
            return -float(self.distance_matrix.sum())
        unique_count = numpy.unique(tour).shape[0]
        if unique_count != self.num_genes or tour.min() < 0 or tour.max() >= self.num_genes:
            missing = self.num_genes - unique_count
            return -float(self.distance_matrix.sum()) * (1.0 + missing)
        return -self.tour_length(tour)
