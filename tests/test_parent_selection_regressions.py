"""Requested parent counts and row mappings for stochastic universal selection."""

import copy

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


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
    monkeypatch.setattr(numpy.random, "uniform",
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
