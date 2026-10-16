"""
Tests for roulette wheel and stochastic universal selection with extreme
fitness values.

Libraries built on PyGAD often score an invalid solution with the most
negative float, -sys.float_info.max, so it can never win. Shifting by that
value, or adding fitness values close to the float limits, overflows to
infinity, and every selection chance became zero or undefined. These tests
check that the chances stay finite and meaningful at the float limits and
with infinite or missing values.
"""

import sys

import numpy
import pytest

import pygad

FLOAT_MAX = sys.float_info.max


def _zero_fitness(ga, solution, solution_idx):
    return 0.0


def _make_ga(selection="rws"):
    return pygad.GA(num_generations=1,
                    num_parents_mating=4,
                    fitness_func=_zero_fitness,
                    initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                    parent_selection_type=selection,
                    mutation_type=None,
                    random_seed=5,
                    suppress_warnings=True)


def _probabilities(fitness):
    return _make_ga().fitness_selection_probabilities(numpy.asarray(fitness, dtype=float))


def test_the_most_negative_float_gets_no_chance_and_the_rest_stay_proportional():
    # Shifted by the most negative value, the other three are almost
    # exactly equal, so each gets about a third.
    probabilities = _probabilities([-FLOAT_MAX, 1.0, 2.0, 3.0])
    assert numpy.all(numpy.isfinite(probabilities))
    assert probabilities[0] == 0.0
    numpy.testing.assert_allclose(probabilities[1:], [1 / 3] * 3, rtol=1e-12)


def test_huge_values_of_both_signs_do_not_overflow():
    probabilities = _probabilities([1e308, -1e308, 0.0, 5.0])
    assert numpy.all(numpy.isfinite(probabilities))
    assert probabilities.sum() == pytest.approx(1.0)
    # Shifted: 2e308, 0, 1e308, about 1e308, so the shares are 1/2, 0, 1/4, 1/4.
    numpy.testing.assert_allclose(probabilities, [0.5, 0.0, 0.25, 0.25], rtol=1e-12)


def test_huge_positive_values_keep_their_proportions():
    # Their sum, 5.2e308, is beyond the float range, so the expected shares
    # are written out: 1, 1.5, 1, and 1.7 parts out of 5.2.
    probabilities = _probabilities([1e308, 1.5e308, 1e308, 1.7e308])
    numpy.testing.assert_allclose(probabilities, numpy.array([1, 1.5, 1, 1.7]) / 5.2, rtol=1e-12)


def test_minus_infinity_and_nan_get_no_chance():
    probabilities = _probabilities([-numpy.inf, 1.0, numpy.nan, 3.0])
    assert numpy.all(numpy.isfinite(probabilities))
    assert probabilities[0] == 0.0
    assert probabilities[2] == 0.0
    numpy.testing.assert_allclose(probabilities[[1, 3]], [0.25, 0.75], rtol=1e-12)


def test_plus_infinity_takes_every_chance_split_evenly():
    probabilities = _probabilities([numpy.inf, 1.0, numpy.inf, 3.0])
    numpy.testing.assert_allclose(probabilities, [0.5, 0.0, 0.5, 0.0])


def test_ordinary_non_negative_fitness_keeps_its_exact_previous_values():
    fitness = numpy.array([0.3, 1.7, 2.2, 0.8])
    probabilities = _probabilities(fitness)
    numpy.testing.assert_array_equal(probabilities, fitness / numpy.sum(fitness))


def test_no_usable_fitness_raises_a_clear_error():
    with pytest.raises(ValueError, match="no fitness value is finite"):
        _probabilities([-numpy.inf, numpy.nan, -numpy.inf, numpy.nan])


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_a_run_with_the_most_negative_float_completes(selection):
    # Every solution whose first gene is above 5 is invalid and gets the
    # most negative float, the way libraries mark rejected solutions.
    def fitness(ga, solution, solution_idx):
        if solution[0] > 5:
            return -FLOAT_MAX
        return float(solution[0])

    ga = pygad.GA(num_generations=5,
                  num_parents_mating=4,
                  fitness_func=fitness,
                  sol_per_pop=10,
                  num_genes=2,
                  init_range_low=0,
                  init_range_high=10,
                  parent_selection_type=selection,
                  random_seed=4,
                  suppress_warnings=True)
    ga.run()
    assert ga.generations_completed == 5
