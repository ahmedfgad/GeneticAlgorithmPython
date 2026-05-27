"""
DTLZ multi-objective benchmark problems.

Each problem takes M (objectives) and k (distance variables). The
number of decision variables is M + k - 1. Defaults: k = 5 for
DTLZ1, k = 10 for DTLZ2, DTLZ3, and DTLZ4.

Fitness values are negated so PyGAD can maximize them.
"""

import math

import numpy


class _DtlzProblem:
    """Base class with attributes shared by every DTLZ problem."""
    bounds = (0.0, 1.0)

    def __init__(self, num_objectives, num_distance_vars):
        if num_objectives < 2:
            raise ValueError(
                f"num_objectives must be at least 2 for the DTLZ suite, "
                f"but got {num_objectives}.")
        self.num_objectives = int(num_objectives)
        self.num_distance_vars = int(num_distance_vars)
        self.num_genes = self.num_objectives + self.num_distance_vars - 1


class DTLZ1(_DtlzProblem):
    """
    DTLZ1. Pareto front is the linear hyperplane sum(f_i) = 0.5.
    The g-function has many local minima.
    """

    def __init__(self, num_objectives=3, num_distance_vars=5):
        super().__init__(num_objectives, num_distance_vars)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        position_vars = x[:self.num_objectives - 1]
        distance_vars = x[self.num_objectives - 1:]
        g_value = 100.0 * (
            self.num_distance_vars
            + numpy.sum((distance_vars - 0.5) ** 2
                        - numpy.cos(20.0 * math.pi * (distance_vars - 0.5)))
        )
        radius = 0.5 * (1.0 + g_value)
        objectives = []
        for objective_index in range(self.num_objectives):
            value = radius
            for cos_index in range(self.num_objectives - 1 - objective_index):
                value *= position_vars[cos_index]
            if objective_index > 0:
                value *= (1.0 - position_vars[self.num_objectives - 1 - objective_index])
            objectives.append(-float(value))
        return objectives


class DTLZ2(_DtlzProblem):
    """
    DTLZ2. Pareto front is the first orthant of the unit sphere
    (sum(f_i ** 2) = 1). g is simple, so the challenge is diversity.
    """

    def __init__(self, num_objectives=3, num_distance_vars=10):
        super().__init__(num_objectives, num_distance_vars)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        position_vars = x[:self.num_objectives - 1]
        distance_vars = x[self.num_objectives - 1:]
        g_value = numpy.sum((distance_vars - 0.5) ** 2)
        radius = 1.0 + g_value
        angles = position_vars * (math.pi / 2.0)
        objectives = []
        for objective_index in range(self.num_objectives):
            value = radius
            for cos_index in range(self.num_objectives - 1 - objective_index):
                value *= math.cos(angles[cos_index])
            if objective_index > 0:
                value *= math.sin(angles[self.num_objectives - 1 - objective_index])
            objectives.append(-float(value))
        return objectives


class DTLZ3(_DtlzProblem):
    """
    DTLZ3. Same unit-sphere front as DTLZ2, with the hard
    multimodal g from DTLZ1. Convergence is harder.
    """

    def __init__(self, num_objectives=3, num_distance_vars=10):
        super().__init__(num_objectives, num_distance_vars)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        position_vars = x[:self.num_objectives - 1]
        distance_vars = x[self.num_objectives - 1:]
        g_value = 100.0 * (
            self.num_distance_vars
            + numpy.sum((distance_vars - 0.5) ** 2
                        - numpy.cos(20.0 * math.pi * (distance_vars - 0.5)))
        )
        radius = 1.0 + g_value
        angles = position_vars * (math.pi / 2.0)
        objectives = []
        for objective_index in range(self.num_objectives):
            value = radius
            for cos_index in range(self.num_objectives - 1 - objective_index):
                value *= math.cos(angles[cos_index])
            if objective_index > 0:
                value *= math.sin(angles[self.num_objectives - 1 - objective_index])
            objectives.append(-float(value))
        return objectives


class DTLZ4(_DtlzProblem):
    """
    DTLZ4. Same shape as DTLZ2, but position variables are raised
    to alpha (default 100). Solutions get pushed toward one corner.
    """

    def __init__(self, num_objectives=3, num_distance_vars=10, alpha=100.0):
        super().__init__(num_objectives, num_distance_vars)
        self.alpha = float(alpha)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        position_vars = x[:self.num_objectives - 1] ** self.alpha
        distance_vars = x[self.num_objectives - 1:]
        g_value = numpy.sum((distance_vars - 0.5) ** 2)
        radius = 1.0 + g_value
        angles = position_vars * (math.pi / 2.0)
        objectives = []
        for objective_index in range(self.num_objectives):
            value = radius
            for cos_index in range(self.num_objectives - 1 - objective_index):
                value *= math.cos(angles[cos_index])
            if objective_index > 0:
                value *= math.sin(angles[self.num_objectives - 1 - objective_index])
            objectives.append(-float(value))
        return objectives
