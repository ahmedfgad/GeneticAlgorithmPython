"""
DTLZ family of multi-objective benchmark problems.

Every problem supports an arbitrary number of objectives M and a
number of decision variables n. By convention n = M + k - 1 where k
is the number of "distance" variables. The defaults use k = 5 for
DTLZ1 and k = 10 for DTLZ2, DTLZ3, and DTLZ4.

All fitness values are negated so PyGAD can maximize toward the
original minimum.
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
    DTLZ1. The Pareto front is a linear hyperplane where
    sum(f_i) = 0.5. The g-function has many local minima so the
    problem is hard to converge.
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
    DTLZ2. The Pareto front is the part of the unit sphere where
    sum(f_i ** 2) = 1 in the first orthant. The g-function is
    simple, so the main challenge is keeping the population diverse.
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
    DTLZ3. Same Pareto front shape as DTLZ2 (the unit sphere). But
    the g-function is the hard multimodal one from DTLZ1, so the
    problem is harder to converge.
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
    DTLZ4. Same shape as DTLZ2 but the position variables are raised
    to a power (alpha, default 100). This makes the front strongly
    biased toward one corner.
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
