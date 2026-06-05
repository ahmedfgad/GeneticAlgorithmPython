"""
Classic single-objective benchmark problems.

Each class is callable with the (ga, solution, sol_idx) signature
and returns a single fitness value. Minimization values are negated
so PyGAD can maximize them.
"""

import math

import numpy


class _SingleObjectiveProblem:
    """Base class with attributes shared by every classic problem."""
    num_objectives = 1

    def __init__(self, num_genes):
        self.num_genes = int(num_genes)


class Sphere(_SingleObjectiveProblem):
    """Sphere. Global minimum at the origin, f(x) = 0."""
    bounds = (-5.12, 5.12)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        return -float(numpy.sum(x ** 2))


class Rastrigin(_SingleObjectiveProblem):
    """Rastrigin. Many regularly-spaced local minima. Global minimum at the origin, f(x) = 0."""
    bounds = (-5.12, 5.12)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        value = 10.0 * self.num_genes + numpy.sum(x ** 2 - 10.0 * numpy.cos(2.0 * math.pi * x))
        return -float(value)


class Rosenbrock(_SingleObjectiveProblem):
    """
    Rosenbrock. Global minimum at x = (1, ..., 1), f(x) = 0.
    The minimum sits in a long narrow banana-shaped valley.
    """
    bounds = (-5.0, 10.0)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        value = numpy.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        return -float(value)


class Griewank(_SingleObjectiveProblem):
    """Griewank. Many local minima over a wide area. Global minimum at the origin, f(x) = 0."""
    bounds = (-600.0, 600.0)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        sum_term = numpy.sum(x ** 2) / 4000.0
        i = numpy.arange(1, self.num_genes + 1, dtype=float)
        prod_term = numpy.prod(numpy.cos(x / numpy.sqrt(i)))
        value = 1.0 + sum_term - prod_term
        return -float(value)


class Schwefel(_SingleObjectiveProblem):
    """
    Schwefel. Global minimum at x = (420.9687, ..., 420.9687), f(x) ~ 0.
    It sits far from the next-best local minimum, which trips up
    many algorithms.
    """
    bounds = (-500.0, 500.0)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        value = 418.9829 * self.num_genes - numpy.sum(x * numpy.sin(numpy.sqrt(numpy.abs(x))))
        return -float(value)


class Ackley(_SingleObjectiveProblem):
    """Ackley. Near-flat outer region with a deep narrow basin at the origin, f(x) = 0."""
    bounds = (-32.768, 32.768)

    def __init__(self, num_genes=10):
        super().__init__(num_genes)

    def __call__(self, ga, solution, sol_idx):
        x = numpy.asarray(solution, dtype=float)
        a = 20.0
        b = 0.2
        c = 2.0 * math.pi
        term1 = -a * math.exp(-b * math.sqrt(numpy.mean(x ** 2)))
        term2 = -math.exp(numpy.mean(numpy.cos(c * x)))
        value = term1 + term2 + a + math.e
        return -float(value)


class Himmelblau(_SingleObjectiveProblem):
    """
    Himmelblau. 2D problem with four equal global minima at f = 0:
        (3.0, 2.0),
        (-2.805, 3.131),
        (-3.779, -3.283),
        (3.584, -1.848).
    """
    bounds = (-5.0, 5.0)

    def __init__(self):
        super().__init__(num_genes=2)

    def __call__(self, ga, solution, sol_idx):
        x, y = float(solution[0]), float(solution[1])
        value = (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2
        return -float(value)
