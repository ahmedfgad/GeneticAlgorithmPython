"""Sampling probabilities, requested parent counts, and population row mappings."""

from unittest.mock import Mock

import copy

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


@pytest.mark.parametrize("fitness,expected_counts", [
    ([1., 2., 3., 4.], [100, 200, 300, 400]),
    ([4., 1., 3., 2.], [400, 100, 300, 200]),
    ([-1., -4., -2., -3.], [400, 100, 300, 200]),
    ([[2., 2.], [0., 0.], [3., 3.], [1., 1.]], [300, 100, 400, 200]),
])
@pytest.mark.parametrize("gene_type", [int, float, [int, float]])
def test_rank_selection_favors_better_fitness_and_maps_population_rows(
        monkeypatch, fitness, expected_counts, gene_type):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           gene_type=copy.deepcopy(gene_type),
                           mutation_type=None,
                           suppress_warnings=True)
    # Sample every interval uniformly without statistical sampling noise.
    random_pointers = iter((numpy.arange(1000) + 0.5) / 1000)
    monkeypatch.setattr(ga_instance, 'numpy_random_generator', Mock(wraps=ga_instance.numpy_random_generator))
    monkeypatch.setattr(ga_instance.numpy_random_generator, "rand", lambda: next(random_pointers))

    parents, parents_indices = ga_instance.rank_selection(
        fitness=numpy.array(fitness), num_parents=1000)

    numpy.testing.assert_array_equal(
        numpy.bincount(parents_indices, minlength=4), expected_counts)
    numpy.testing.assert_array_equal(parents, ga_instance.population[parents_indices])
    assert parents.dtype == ga_instance.population.dtype


@pytest.mark.parametrize("fitness", [[2., 2., 1., 1.],
                                       [[0., 3.], [3., 0.], [1., 2.], [2., 1.]]])
def test_rank_selection_favors_best_tied_group_or_crowding_boundaries(monkeypatch, fitness):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           mutation_type=None,
                           suppress_warnings=True)
    random_pointers = iter((numpy.arange(1000) + 0.5) / 1000)
    monkeypatch.setattr(ga_instance, 'numpy_random_generator', Mock(wraps=ga_instance.numpy_random_generator))
    monkeypatch.setattr(ga_instance.numpy_random_generator, "rand", lambda: next(random_pointers))

    parents, parents_indices = ga_instance.rank_selection(
        fitness=numpy.array(fitness), num_parents=1000)

    # The two better-fitness rows or Pareto-front boundaries receive 70%.
    assert numpy.count_nonzero(parents_indices < 2) == 700
    numpy.testing.assert_array_equal(parents, ga_instance.population[parents_indices])


@pytest.mark.parametrize("num_parents", [1, 2, 7])
@pytest.mark.parametrize("gene_type", [int, float, [int, float]])
def test_rank_selection_returns_requested_parent_copies(num_parents, gene_type):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           gene_type=copy.deepcopy(gene_type),
                           mutation_type=None,
                           random_seed=17,
                           suppress_warnings=True)
    original_population = ga_instance.population.copy()

    parents, parents_indices = ga_instance.rank_selection(
        fitness=numpy.zeros(4), num_parents=num_parents)

    assert parents.shape == (num_parents, ga_instance.num_genes)
    assert parents_indices.shape == (num_parents,)
    assert parents.dtype == original_population.dtype
    numpy.testing.assert_array_equal(parents, original_population[parents_indices])
    parents[:] = -1
    numpy.testing.assert_array_equal(ga_instance.population, original_population)


def test_rank_selection_with_one_solution():
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=1,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2]],
                           mutation_type=None,
                           suppress_warnings=True)

    parents, parents_indices = ga_instance.rank_selection(
        fitness=numpy.array([0.]), num_parents=3)

    numpy.testing.assert_array_equal(parents_indices, [0, 0, 0])
    numpy.testing.assert_array_equal(parents, [[1, 2], [1, 2], [1, 2]])


@pytest.mark.parametrize("num_parents", [1, 2, 4, 8])
@pytest.mark.parametrize("multi_objective", [False, True])
@pytest.mark.parametrize("gene_type", [int, float, [int, float]])
@pytest.mark.parametrize("offset_fraction", [0.0, 0.5, 0.999999])
def test_sus_respects_requested_count_and_balances_equal_fitness(
        monkeypatch, num_parents, multi_objective, gene_type, offset_fraction):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           gene_type=copy.deepcopy(gene_type),
                           mutation_type=None,
                           random_seed=17,
                           suppress_warnings=True)
    fitness = (numpy.ones((4, 2)) if multi_objective else numpy.ones(4))
    monkeypatch.setattr(ga_instance, 'numpy_random_generator', Mock(wraps=ga_instance.numpy_random_generator))
    monkeypatch.setattr(ga_instance.numpy_random_generator, "uniform",
                        lambda **options: numpy.array([options["high"] * offset_fraction]))

    parents, parents_indices = ga_instance.stochastic_universal_selection(
        fitness=fitness, num_parents=num_parents)

    assert parents.shape == (num_parents, ga_instance.num_genes)
    assert parents.dtype == ga_instance.population.dtype
    assert parents_indices.shape == (num_parents,)
    assert numpy.all((parents_indices >= 0) & (parents_indices < ga_instance.sol_per_pop))
    numpy.testing.assert_array_equal(parents, ga_instance.population[parents_indices])
    selection_counts = numpy.bincount(parents_indices, minlength=ga_instance.sol_per_pop)
    assert selection_counts.max() - selection_counts.min() <= 1


def test_sus_selected_parents_do_not_alias_the_population():
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           mutation_type=None,
                           random_seed=17,
                           suppress_warnings=True)
    original = ga_instance.population.copy()

    parents, _ = ga_instance.stochastic_universal_selection(
        fitness=numpy.arange(1, 5, dtype=float), num_parents=4)
    parents[:] = -1

    numpy.testing.assert_array_equal(ga_instance.population, original)


@pytest.mark.parametrize("selection", ["rws", "sus"])
@pytest.mark.parametrize("dtype,value", [
    (numpy.int64, 2**62),
    (numpy.uint64, 2**63),
    (numpy.float16, 40000),
    (numpy.float32, 2e38),
])
@pytest.mark.parametrize("multi_objective", [False, True])
def test_fitness_sum_does_not_overflow_during_selection(
        selection, dtype, value, multi_objective):
    def large_fitness(ga_instance, solution, solution_idx):
        fitness = dtype(value)
        return [fitness, fitness] if multi_objective else fitness

    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=large_fitness,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           parent_selection_type=selection,
                           mutation_type=None,
                           random_seed=17,
                           suppress_warnings=True)
    ga_instance.run()

    assert ga_instance.generations_completed == 1
    assert ga_instance.last_generation_parents_indices.shape == (2,)
    numpy.testing.assert_array_equal(
        ga_instance.last_generation_parents,
        ga_instance.population[ga_instance.last_generation_parents_indices])
    assert numpy.all(ga_instance.last_generation_fitness > 0)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_probability_selection_still_rejects_zero_sum_fitness(selection):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           parent_selection_type=selection,
                           mutation_type=None,
                           suppress_warnings=True)
    with pytest.raises(ZeroDivisionError, match="sum of fitness values is zero"):
        ga_instance.select_parents(numpy.zeros(4), 2)


@pytest.mark.parametrize("selection", ["rws", "sus"])
def test_probability_selection_preserves_python_integer_fitness(selection):
    ga_instance = pygad.GA(num_generations=1,
                           num_parents_mating=2,
                           fitness_func=fitness_func,
                           initial_population=[[1, 2], [3, 4], [5, 6], [7, 8]],
                           parent_selection_type=selection,
                           mutation_type=None,
                           random_seed=17,
                           suppress_warnings=True)
    fitness = numpy.array([2**1500] * 4, dtype=object)
    parents, indices = ga_instance.select_parents(fitness, 2)

    assert indices.shape == (2,)
    numpy.testing.assert_array_equal(parents, ga_instance.population[indices])
