"""
PyGAD Example: Using the gene_space Parameter
==============================================

The `gene_space` parameter controls the set of valid values that each gene
can take. This example demonstrates all major forms of `gene_space`.

Documentation:
  https://pygad.readthedocs.io/en/latest/pygad_more.html#more-about-the-gene-space-parameter

Dependencies:
  pip install pygad numpy
"""

import numpy
import pygad


def fitness_func(ga_instance, solution, solution_idx):
    """Return the sum of all gene values as the fitness score."""
    return numpy.sum(solution)


# ===========================================================================
# 1. Single flat list or NumPy array (shared across all genes)
# ===========================================================================
# Every gene independently draws its value from the same list.
# Using None in the list means that gene value is unconstrained (random float).

# gene_space = [1, 2, 3, 4, 5]
# gene_space = [1, 2, None]
# gene_space = numpy.array([10, 20, 30, 40, 50])

# ===========================================================================
# 2. A range or NumPy sequence (shared across all genes)
# ===========================================================================

# gene_space = range(1, 11)                        # integers 1 through 10
# gene_space = numpy.arange(0.0, 1.1, 0.1)         # [0.0, 0.1, ..., 1.0]
# gene_space = numpy.linspace(1, 5, num=9)          # 9 evenly-spaced floats in [1, 5]

# ===========================================================================
# 3. Per-gene list (each gene has its own independent space)
# ===========================================================================
# When gene_space is a list with length equal to num_genes, each element
# defines the space for the corresponding gene. Elements can be a list,
# a NumPy array, a range, or None (unconstrained).

# gene_space = [range(1, 5), range(5, 11), range(10, 21)]
# gene_space = [[1, 2, 3], None, range(10, 21)]
# gene_space = [[1, 2, 3], numpy.linspace(5, 10, num=6), [10, 12, 14, 16, 18, 20]]

# ===========================================================================
# 4. Dictionary – continuous range (shared across all genes)
# ===========================================================================
# A dict with "low" and "high" makes every gene sample from a continuous
# uniform distribution. An optional "step" key discretises the range.

# gene_space = {"low": 0.0, "high": 10.0}
# gene_space = {"low": 0.0, "high": 10.0, "step": 0.5}

# ===========================================================================
# 5. Per-gene list of dictionaries (each gene has its own continuous range)
# ===========================================================================
# Each gene gets its own dict-based range. This is the most expressive form
# and is useful when each gene lives in a different numerical domain.

gene_space = [
    [1, 2, 3, 4, 5], # gene 0
    {"low": 5,  "high": 10}, # gene 1
    range(10, 15), # gene 2
    None, # gene 3
]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=4,
    sol_per_pop=8,
    num_genes=4,
    fitness_func=fitness_func,
    gene_space=gene_space,
)

ga_instance.run()

best_solution, best_fitness, _ = ga_instance.best_solution()
print(f"Best solution : {best_solution}")
print(f"Best fitness  : {best_fitness:.4f}")