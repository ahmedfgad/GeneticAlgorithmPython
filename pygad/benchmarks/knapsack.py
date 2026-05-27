"""
0/1 Knapsack benchmark problem.

Each item has a weight and a value. The goal is to pick a subset of
items so the total value is the maximum possible while keeping the
total weight at or below the capacity.

A solution is a binary vector of length num_items. A 1 means the
item is picked and a 0 means it is not. To plug into PyGAD use the
class attributes `gene_space` and `gene_type` directly:

    problem = Knapsack(weights=[...], values=[...], capacity=...)
    ga = pygad.GA(
        ...,
        num_genes=problem.num_genes,
        gene_space=problem.gene_space,
        gene_type=problem.gene_type,
        fitness_func=problem,
    )
"""

import numpy


class Knapsack:
    """
    0/1 knapsack problem.

    If a candidate exceeds the capacity, the fitness is negative and
    scaled by how much the solution is over the limit. So infeasible
    solutions still carry useful gradient information toward the
    feasible region.
    """
    num_objectives = 1
    gene_space = [0, 1]
    gene_type = int

    def __init__(self, weights, values, capacity):
        weights = numpy.asarray(weights, dtype=float)
        values = numpy.asarray(values, dtype=float)
        if weights.ndim != 1:
            raise ValueError(
                f"weights must be a 1D array, but got shape {weights.shape}.")
        if values.ndim != 1:
            raise ValueError(
                f"values must be a 1D array, but got shape {values.shape}.")
        if weights.shape != values.shape:
            raise ValueError(
                f"weights and values must have the same length, but got "
                f"{weights.shape[0]} weights and {values.shape[0]} values.")
        if not numpy.all(weights >= 0):
            raise ValueError(
                "weights must be non-negative.")
        if not numpy.all(values >= 0):
            raise ValueError(
                "values must be non-negative.")
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, but got {capacity}.")
        self.weights = weights
        self.values = values
        self.capacity = float(capacity)
        self.num_genes = int(weights.shape[0])

    def __call__(self, ga, solution, sol_idx):
        choice = numpy.asarray(solution, dtype=int)
        total_weight = float(numpy.sum(choice * self.weights))
        total_value = float(numpy.sum(choice * self.values))
        if total_weight > self.capacity:
            return -(total_weight - self.capacity)
        return total_value
