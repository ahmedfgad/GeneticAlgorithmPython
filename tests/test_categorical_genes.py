"""
Tests for the gene_categorical parameter.

A categorical gene holds a choice from a list where the order has no
meaning, such as an activation function stored as 0 for relu, 1 for tanh
and 2 for sigmoid. SBX crossover and polynomial mutation normally treat
the gene value as a point on a number line. For a categorical gene, they
must treat it as a choice instead:

1. Polynomial mutation picks one of the other allowed values at random.
2. SBX crossover gives the child the value of one of the two parents.
"""

import numpy
import pytest

import pygad


# Number of offspring rows used to measure each distribution.
NUM_TRIALS = 3000


def _zero_fitness(ga, solution, solution_idx):
    return 0.0


def _make_ga(gene_space, gene_categorical, gene_type=int, num_genes=1, **other_settings):
    """Build a small GA whose operators can be called directly."""
    settings = dict(num_generations=1,
                    num_parents_mating=2,
                    fitness_func=_zero_fitness,
                    sol_per_pop=4,
                    num_genes=num_genes,
                    gene_space=gene_space,
                    gene_type=gene_type,
                    gene_categorical=gene_categorical,
                    mutation_type="polynomial",
                    polynomial_mutation_eta=5,
                    mutation_num_genes=1,
                    random_seed=1,
                    suppress_warnings=True)
    settings.update(other_settings)
    return pygad.GA(**settings)


def _shares(values, allowed_values):
    """Return the share of each allowed value among the given values."""
    values = numpy.asarray(values, dtype=float)
    return {allowed: float(numpy.mean(values == allowed)) for allowed in allowed_values}


# Validation of the parameter.

def test_default_marks_every_gene_as_ordered():
    ga = _make_ga(gene_space=[[0, 1, 2], [0, 1, 2]], gene_categorical=None, num_genes=2)
    assert ga.gene_categorical == [False, False]


def test_single_boolean_applies_to_every_gene():
    ga = _make_ga(gene_space=[[0, 1, 2], [5, 6]], gene_categorical=True, num_genes=2)
    assert ga.gene_categorical == [True, True]


def test_list_marks_each_gene_separately():
    ga = _make_ga(gene_space=[{"low": 0.0, "high": 1.0}, [0, 1, 2]],
                  gene_categorical=[False, True],
                  gene_type=[float, int],
                  num_genes=2)
    assert ga.gene_categorical == [False, True]


def test_rejects_a_value_that_is_not_a_boolean_or_a_list():
    with pytest.raises(TypeError, match="gene_categorical"):
        _make_ga(gene_space=[[0, 1, 2]], gene_categorical="yes")


def test_rejects_a_list_with_a_non_boolean_entry():
    with pytest.raises(TypeError, match="gene_categorical"):
        _make_ga(gene_space=[[0, 1, 2]], gene_categorical=[1])


def test_rejects_a_list_with_the_wrong_length():
    with pytest.raises(ValueError, match="gene_categorical"):
        _make_ga(gene_space=[[0, 1, 2], [0, 1, 2]], gene_categorical=[True], num_genes=2)


def test_rejects_a_categorical_gene_without_a_gene_space():
    with pytest.raises(ValueError, match="gene_categorical"):
        _make_ga(gene_space=None, gene_categorical=True)


def test_rejects_a_categorical_gene_with_a_continuous_range():
    with pytest.raises(ValueError, match="gene_categorical"):
        _make_ga(gene_space=[{"low": 0, "high": 5}], gene_categorical=True)


def test_rejects_a_categorical_gene_whose_list_contains_none():
    # None in a gene space stands for any value in the initialization
    # range, so the allowed values are not a fixed list.
    with pytest.raises(ValueError, match="gene_categorical"):
        _make_ga(gene_space=[[0, 1, None]], gene_categorical=True)


def test_accepts_a_categorical_gene_with_a_stepped_range():
    ga = _make_ga(gene_space=[{"low": 0, "high": 5, "step": 1}], gene_categorical=True)
    assert ga.gene_categorical == [True]


# Polynomial mutation.

def test_polynomial_mutation_picks_every_other_category_equally_often():
    ga = _make_ga(gene_space=[[0, 1, 2, 3, 4]], gene_categorical=True)
    offspring = numpy.full((NUM_TRIALS, 1), 0)
    values = ga.polynomial_mutation(offspring)[:, 0]
    shares = _shares(values, [0, 1, 2, 3, 4])
    # The mutated gene must change, and each other category is equally likely.
    assert shares[0] == 0.0
    for category in (1, 2, 3, 4):
        assert 0.2 < shares[category] < 0.3, shares


def test_polynomial_mutation_uses_only_allowed_values_of_an_uneven_list():
    ga = _make_ga(gene_space=[[16, 32, 64]], gene_categorical=True)
    offspring = numpy.full((NUM_TRIALS, 1), 32)
    values = ga.polynomial_mutation(offspring)[:, 0]
    shares = _shares(values, [16, 32, 64])
    assert shares[32] == 0.0
    assert 0.45 < shares[16] < 0.55
    assert 0.45 < shares[64] < 0.55


def test_polynomial_mutation_keeps_the_only_allowed_value():
    ga = _make_ga(gene_space=[[7]], gene_categorical=True)
    offspring = numpy.full((10, 1), 7)
    values = ga.polynomial_mutation(offspring)[:, 0]
    assert numpy.all(values == 7)


def test_polynomial_mutation_still_moves_ordered_genes_on_the_number_line():
    # The float gene is ordered and must still get small decimal steps,
    # while the categorical gene only takes values from its list.
    ga = _make_ga(gene_space=[{"low": 0.0, "high": 10.0}, [0, 1, 2]],
                  gene_categorical=[False, True],
                  gene_type=[float, int],
                  num_genes=2,
                  mutation_num_genes=2)
    offspring = numpy.array([[5.0, 0]] * NUM_TRIALS, dtype=object)
    result = ga.polynomial_mutation(offspring)
    float_gene = result[:, 0].astype(float)
    categorical_gene = result[:, 1].astype(float)
    assert numpy.mean(float_gene != numpy.round(float_gene)) > 0.9
    assert set(categorical_gene) == {1.0, 2.0}


# SBX crossover.

def test_sbx_crossover_copies_one_parent_value_for_a_categorical_gene():
    ga = _make_ga(gene_space=[[0, 1, 2, 3, 4]],
                  gene_categorical=True,
                  crossover_type="sbx",
                  sbx_crossover_eta=5)
    parents = numpy.array([[1], [4]])
    children = ga.sbx_crossover(parents, (NUM_TRIALS, 1))[:, 0]
    shares = _shares(children, [0, 1, 2, 3, 4])
    assert shares[0] == shares[2] == shares[3] == 0.0
    assert 0.45 < shares[1] < 0.55
    assert 0.45 < shares[4] < 0.55


# Interaction with other gene rules and with a full run.

def test_categorical_mutation_respects_a_gene_constraint():
    # The constraint forbids the value 2.
    ga = _make_ga(gene_space=[[0, 1, 2, 3]],
                  gene_categorical=True,
                  gene_constraint=[lambda solution, values: [value for value in values if value != 2]])
    offspring = numpy.full((NUM_TRIALS, 1), 0)
    values = ga.polynomial_mutation(offspring)[:, 0]
    assert 2 not in set(values.astype(int))


def test_categorical_mutation_respects_allow_duplicate_genes():
    ga = _make_ga(gene_space=[0, 1, 2, 3, 4],
                  gene_categorical=True,
                  num_genes=2,
                  allow_duplicate_genes=False)
    offspring = numpy.array([[0, 1]] * NUM_TRIALS)
    result = ga.polynomial_mutation(offspring)
    assert numpy.all(result[:, 0] != result[:, 1])


def test_full_run_with_mixed_genes_keeps_every_value_allowed():
    def fitness(ga, solution, solution_idx):
        return float(solution[0]) - float(solution[1])

    ga = pygad.GA(num_generations=15,
                  num_parents_mating=4,
                  fitness_func=fitness,
                  sol_per_pop=10,
                  num_genes=3,
                  gene_space=[{"low": 0.0, "high": 1.0},
                              {"low": 16, "high": 129, "step": 1},
                              [0, 1, 2]],
                  gene_type=[float, int, int],
                  gene_categorical=[False, False, True],
                  crossover_type="sbx",
                  mutation_type="polynomial",
                  random_seed=3,
                  suppress_warnings=True)
    ga.run()
    population = numpy.asarray(ga.population, dtype=float)
    assert numpy.all((population[:, 0] >= 0.0) & (population[:, 0] <= 1.0))
    assert numpy.all((population[:, 1] >= 16) & (population[:, 1] <= 128))
    assert set(population[:, 2]) <= {0.0, 1.0, 2.0}
