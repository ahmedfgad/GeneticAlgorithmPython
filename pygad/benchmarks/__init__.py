"""
Benchmark problems for PyGAD.

Each problem class can be called with the fitness signature
(ga, solution, sol_idx) and returns a fitness in PyGAD's
maximization format. For problems that are normally written as
minimisation, the values are negated.

Each class also exposes num_genes, num_objectives, and bounds so
you can plug it into pygad.GA directly.
"""

from pygad.benchmarks import classic
from pygad.benchmarks import zdt
from pygad.benchmarks import dtlz
from pygad.benchmarks import knapsack
from pygad.benchmarks import tsp

__all__ = ["classic", "zdt", "dtlz", "knapsack", "tsp"]
