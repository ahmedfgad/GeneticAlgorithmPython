"""
Tests for roulette wheel and stochastic universal selection when some
fitness values are zero or negative.

Both selectors give each solution a share of a wheel equal to its fitness
divided by the total fitness. With negative values this division flips the
order: for the fitness values -1 and -9, the worse solution (-9) would get
90% of the wheel. The selectors must shift negative fitness values so the
worst solution starts at zero, which keeps the better solutions favored.
"""

import numpy
import pytest

import pygad


# Number of selection rounds used to measure each share.
NUM_ROUNDS = 2500


def _zero_fitness(ga, solution, solution_idx):
    return 0.0


def _make_ga(selection):
    return pygad.GA(num_generations=1,
                    num_parents_mating=4,
                    fitness_func=_zero_fitness,
                    initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                    parent_selection_type=selection,
                    mutation_type=None,
                    random_seed=5,
                    suppress_warnings=True)


def _selection_shares(selection, fitness):
    """Select parents many times and return how often each solution was picked."""
    ga = _make_ga(selection)
    counts = numpy.zeros(4)
    for _ in range(NUM_ROUNDS):
        _, indices = ga.select_parents(numpy.asarray(fitness), 4)
        counts += numpy.bincount(indices, minlength=4)
    return counts / counts.sum()


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_negative_fitness_favors_the_better_solutions(selection):
    # Shifting by the smallest value gives the weights 8, 0, 4, 6.
    shares = _selection_shares(selection, [-1.0, -9.0, -5.0, -3.0])
    expected = numpy.array([8, 0, 4, 6]) / 18
    numpy.testing.assert_allclose(shares, expected, atol=0.02)
    # The worst solution has no share of the wheel.
    assert shares[1] == 0.0


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_mixed_sign_fitness_favors_the_better_solutions(selection):
    # Shifting by -3 gives the weights 5, 2, 3.5, 0.
    shares = _selection_shares(selection, [2.0, -1.0, 0.5, -3.0])
    expected = numpy.array([5, 2, 3.5, 0]) / 10.5
    numpy.testing.assert_allclose(shares, expected, atol=0.02)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_mixed_sign_fitness_with_a_zero_sum_does_not_fail(selection):
    # The sum is zero before shifting, but the values differ, so the
    # selection is still well defined.
    shares = _selection_shares(selection, [1.0, -1.0, 1.0, -1.0])
    numpy.testing.assert_allclose(shares, [0.5, 0.0, 0.5, 0.0], atol=0.02)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_equal_negative_fitness_gives_every_solution_the_same_chance(selection):
    shares = _selection_shares(selection, [-5.0, -5.0, -5.0, -5.0])
    numpy.testing.assert_allclose(shares, [0.25, 0.25, 0.25, 0.25], atol=0.03)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_non_negative_fitness_keeps_its_original_shares(selection):
    # No value is below zero, so the shares stay fitness / total.
    shares = _selection_shares(selection, [1.0, 2.0, 3.0, 4.0])
    numpy.testing.assert_allclose(shares, [0.1, 0.2, 0.3, 0.4], atol=0.02)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_negative_multi_objective_fitness_favors_the_better_solutions(selection):
    # Each row is summed first: -2, -18, -10, -6. Shifting gives 16, 0, 8, 12.
    fitness = [[-1.0, -1.0], [-9.0, -9.0], [-5.0, -5.0], [-3.0, -3.0]]
    shares = _selection_shares(selection, fitness)
    expected = numpy.array([16, 0, 8, 12]) / 36
    numpy.testing.assert_allclose(shares, expected, atol=0.02)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_negative_python_integer_fitness_stays_exact(selection):
    # Python integers too large for a float must still work after shifting.
    big = 2**1500
    fitness = numpy.array([-big, -3 * big, -2 * big, -big], dtype=object)
    ga = _make_ga(selection)
    _, indices = ga.select_parents(fitness, 4)
    assert indices.shape == (4,)
    # The worst solution (index 1) gets no share of the wheel.
    assert 1 not in set(indices.tolist())


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_minimization_with_probability_selection_improves(selection):
    # Minimize x**2 by maximizing -x**2. The best value must move toward 0.
    def fitness(ga, solution, solution_idx):
        return -float(solution[0]) ** 2

    ga = pygad.GA(num_generations=30,
                  num_parents_mating=6,
                  fitness_func=fitness,
                  sol_per_pop=12,
                  num_genes=1,
                  init_range_low=5.0,
                  init_range_high=10.0,
                  parent_selection_type=selection,
                  keep_elitism=0,
                  random_seed=2,
                  suppress_warnings=True)
    ga.run()
    first_best = ga.best_solutions_fitness[0]
    last_best = ga.best_solutions_fitness[-1]
    assert last_best > first_best
