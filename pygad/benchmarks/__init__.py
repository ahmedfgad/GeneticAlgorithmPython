"""
Standard benchmark problems for PyGAD.

Every problem class can be called with the standard PyGAD fitness
function signature (ga, solution, sol_idx) and returns a fitness in
PyGAD's maximization format. The original minimization values are
negated so the user can plug the problem directly into PyGAD without
extra wrapping. Each class also has the attributes num_genes,
num_objectives, and bounds.
"""

from pygad.benchmarks import classic
from pygad.benchmarks import zdt
from pygad.benchmarks import dtlz
from pygad.benchmarks import knapsack

__all__ = ["classic", "zdt", "dtlz", "knapsack"]
