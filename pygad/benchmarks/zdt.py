"""
ZDT multi-objective benchmark problems.

Two objectives. Variables live in [0, 1] (ZDT4 uses a wider range
for some). Every class has a pareto_front() method that returns
points on the true front in PyGAD's maximization format (negated),
which you can pass to the IGD and GD indicators as reference_front.
"""

import numpy


class _ZdtProblem:
    """Base class with attributes shared by every ZDT problem."""
    num_objectives = 2
    bounds = (0.0, 1.0)

    def __init__(self, num_genes):
        self.num_genes = int(num_genes)


class ZDT1(_ZdtProblem):
    """
    ZDT1. Convex front: f2 = 1 - sqrt(f1) for f1 in [0, 1].
    Optimal solutions: x_0 in [0, 1], x_i = 0 for i >= 1.
    """

    def __init__(self, num_genes=30):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        f1 = x[0]
        g = 1.0 + 9.0 * numpy.sum(x[1:]) / (self.num_genes - 1)
        f2 = g * (1.0 - numpy.sqrt(f1 / g))
        return [-float(f1), -float(f2)]

    def pareto_front(self, num_points=100):
        f1 = numpy.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - numpy.sqrt(f1)
        return numpy.stack([-f1, -f2], axis=1)


class ZDT2(_ZdtProblem):
    """
    ZDT2. Non-convex front: f2 = 1 - f1**2 for f1 in [0, 1].
    Same variable layout as ZDT1.
    """

    def __init__(self, num_genes=30):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        f1 = x[0]
        g = 1.0 + 9.0 * numpy.sum(x[1:]) / (self.num_genes - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return [-float(f1), -float(f2)]

    def pareto_front(self, num_points=100):
        f1 = numpy.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - f1 ** 2
        return numpy.stack([-f1, -f2], axis=1)


class ZDT3(_ZdtProblem):
    """ZDT3. Front is five disconnected convex pieces."""

    def __init__(self, num_genes=30):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        f1 = x[0]
        g = 1.0 + 9.0 * numpy.sum(x[1:]) / (self.num_genes - 1)
        h = 1.0 - numpy.sqrt(f1 / g) - (f1 / g) * numpy.sin(10.0 * numpy.pi * f1)
        f2 = g * h
        return [-float(f1), -float(f2)]


class ZDT4(_ZdtProblem):
    """
    ZDT4. x_0 in [0, 1], rest in [-5, 5]. Same convex front as ZDT1
    (f2 = 1 - sqrt(f1)), but the search space has many local minima.
    """
    bounds = (-5.0, 5.0)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        x_first = numpy.clip(x[0], 0.0, 1.0)
        rest = numpy.clip(x[1:], -5.0, 5.0)
        f1 = x_first
        g = (1.0 + 10.0 * (self.num_genes - 1)
             + numpy.sum(rest ** 2 - 10.0 * numpy.cos(4.0 * numpy.pi * rest)))
        f2 = g * (1.0 - numpy.sqrt(f1 / g))
        return [-float(f1), -float(f2)]

    def pareto_front(self, num_points=100):
        f1 = numpy.linspace(0.0, 1.0, num_points)
        f2 = 1.0 - numpy.sqrt(f1)
        return numpy.stack([-f1, -f2], axis=1)


class ZDT6(_ZdtProblem):
    """
    ZDT6. Non-uniform front; solutions cluster at one end.
    """

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
        f1 = 1.0 - numpy.exp(-4.0 * x[0]) * numpy.sin(6.0 * numpy.pi * x[0]) ** 6
        g = 1.0 + 9.0 * (numpy.sum(x[1:]) / (self.num_genes - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return [-float(f1), -float(f2)]

    def pareto_front(self, num_points=100):
        f1 = numpy.linspace(0.281, 1.0, num_points)
        f2 = 1.0 - f1 ** 2
        return numpy.stack([-f1, -f2], axis=1)
