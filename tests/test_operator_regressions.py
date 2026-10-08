"""Edge cases and invariants for the crossover and swap fixes."""

import itertools

import numpy
import pytest

import pygad


def _fitness(ga, solution, index):
    return float(numpy.sum(solution))


def _make_ga(num_genes=5, **options):
    parameters = dict(num_generations=5, num_parents_mating=2,
                      fitness_func=_fitness, sol_per_pop=6,
                      num_genes=num_genes, mutation_type=None,
                      random_seed=17, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


@pytest.mark.parametrize("num_genes", [1, 2, 3, 5])
@pytest.mark.parametrize("gene_type", [int, [int, float]])
def test_two_points_crossover_keeps_input_and_gene_types(num_genes, gene_type):
    if isinstance(gene_type, list):
        gene_type = [gene_type[index % 2] for index in range(num_genes)]
    ga = _make_ga(num_genes, gene_type=gene_type, crossover_type="two_points")
    parents = ga.population[:2].copy()
    before = parents.copy()
    children = ga.two_points_crossover(parents, (20, num_genes))

    numpy.testing.assert_array_equal(parents, before)
    assert children.shape == (20, num_genes)
    assert children.dtype == parents.dtype
    for index in range(num_genes):
        assert numpy.all(numpy.isin(children[:, index], parents[:, index]))
        if not ga.gene_type_single:
            assert all(type(value) is type(parents[0, index])
                       for value in children[:, index])
    if num_genes == 1:
        numpy.testing.assert_array_equal(children[:, 0],
                                         parents[(numpy.arange(20) + 1) % 2, 0])


@pytest.mark.parametrize("crossover_type", ["two_points", "sbx"])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_crossover_probability_keeps_or_crosses_parents(crossover_type, probability):
    ga = _make_ga(3, crossover_type=crossover_type,
                  crossover_probability=probability,
                  init_range_low=0.0, init_range_high=1.0)
    parents = numpy.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.8]])
    before = parents.copy()
    children = ga.crossover(parents, (100, 3))

    numpy.testing.assert_array_equal(parents, before)
    if probability == 0.0:
        numpy.testing.assert_array_equal(children, parents[numpy.arange(100) % 2])
    else:
        assert numpy.any(children != parents[numpy.arange(100) % 2])
    assert numpy.all((children >= 0.0) & (children <= 1.0))


def test_two_points_crossover_preserves_permutations_with_duplicate_repair():
    ga = _make_ga(5, crossover_type="two_points", gene_type=int,
                  gene_space=range(5), allow_duplicate_genes=False)
    parents = numpy.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    children = ga.two_points_crossover(parents, (100, 5))

    numpy.testing.assert_array_equal(numpy.sort(children, axis=1),
                                     numpy.tile(numpy.arange(5), (100, 1)))


def test_two_points_crossover_reaches_every_cut_pair():
    ga = _make_ga(5, gene_type=int, crossover_type="two_points")
    parents = numpy.array([[0] * 5, [1] * 5])
    children = ga.two_points_crossover(parents, (1000, 5))
    # Mark the segment inherited from the second parent, for either mating order.
    segments = children.copy()
    segments[1::2] = 1 - segments[1::2]
    observed = set()
    for segment in segments:
        indices = numpy.flatnonzero(segment)
        assert len(indices) > 0
        assert numpy.all(numpy.diff(indices) == 1)
        observed.add((indices[0], indices[-1] + 1))
    assert observed == set(itertools.combinations(range(6), 2))


@pytest.mark.parametrize("num_genes", [2, 3, 5, 6])
def test_swap_mutation_reaches_every_pair_and_preserves_permutations(num_genes):
    ga = _make_ga(num_genes, gene_type=int, mutation_type="swap",
                  gene_space=range(num_genes), allow_duplicate_genes=False)
    original = numpy.tile(numpy.arange(num_genes), (1000, 1))
    children = original.copy()
    result = ga.swap_mutation(children)

    assert result is children
    numpy.testing.assert_array_equal(numpy.sort(children, axis=1), original)
    changed = children != original
    assert numpy.all(numpy.count_nonzero(changed, axis=1) == 2)
    pairs = {tuple(numpy.flatnonzero(row)) for row in changed}
    assert pairs == set(itertools.combinations(range(num_genes), 2))


@pytest.mark.parametrize("num_offspring", [0, 1, 4])
def test_single_gene_swap_is_a_noop(num_offspring):
    ga = _make_ga(1, gene_type=int, mutation_type="swap")
    children = numpy.full((num_offspring, 1), 7, dtype=int)
    before = children.copy()

    assert ga.swap_mutation(children) is children
    numpy.testing.assert_array_equal(children, before)


@pytest.mark.parametrize("crossover_type", ["two_points", "sbx"])
def test_crossover_accepts_empty_offspring(crossover_type):
    ga = _make_ga(crossover_type=crossover_type)
    parents = ga.population[:2].copy()
    assert ga.crossover(parents, (0, 5)).shape == (0, 5)


@pytest.mark.parametrize("parents", [(0.2, 0.8), (0.0, 0.7),
                                    (0.3, 1.0), (0.0, 1.0)])
@pytest.mark.parametrize("quantile", [0.1, 0.9])
def test_sbx_can_select_both_symmetric_children_at_boundaries(monkeypatch, parents,
                                                            quantile):
    ga = _make_ga(1, crossover_type="sbx", init_range_low=0.0,
                  init_range_high=1.0)
    parents = numpy.array(parents).reshape(2, 1)
    # The same spread draw with opposite child choices must straddle the mean.
    draws = iter([quantile, 0.0, quantile, 0.99])
    monkeypatch.setattr(numpy.random, "random", lambda: next(draws))
    children = ga.sbx_crossover(parents, (2, 1))[:, 0]

    assert children[0] < parents.mean() < children[1]
    assert children.sum() == pytest.approx(parents.sum(), abs=1e-12)
    assert numpy.all((children >= 0.0) & (children <= 1.0))


def test_sbx_equal_parents_do_not_draw_random_values(monkeypatch):
    ga = _make_ga(2, crossover_type="sbx")
    parents = numpy.array([[0.25, 0.75], [0.25, 0.75]])

    def unexpected_draw(*args, **kwargs):
        pytest.fail("Equal parents should be copied without a random draw")

    monkeypatch.setattr(numpy.random, "random", unexpected_draw)
    numpy.testing.assert_array_equal(ga.sbx_crossover(parents, (3, 2)),
                                     numpy.tile(parents[0], (3, 1)))


def test_sbx_respects_distinct_bounds_for_each_gene():
    low = numpy.array([-5.0, 10.0, 100.0])
    high = numpy.array([-1.0, 12.0, 200.0])
    ga = _make_ga(3, crossover_type="sbx",
                  init_range_low=low.tolist(), init_range_high=high.tolist())
    parents = numpy.array([low + 0.2 * (high - low),
                           low + 0.8 * (high - low)])
    children = ga.sbx_crossover(parents, (1000, 3))

    assert numpy.all(children >= low)
    assert numpy.all(children <= high)
    share_above = numpy.mean(children > parents.mean(axis=0), axis=0)
    assert numpy.all((share_above > 0.4) & (share_above < 0.6))


@pytest.mark.parametrize("crossover_type,mutation_type", [
    ("two_points", "swap"), ("sbx", "polynomial")])
def test_seeded_runs_are_reproducible_with_corrected_operators(crossover_type,
                                                              mutation_type):
    populations = []
    for _ in range(2):
        ga = _make_ga(crossover_type=crossover_type, mutation_type=mutation_type)
        ga.run()
        assert numpy.isfinite(ga.population).all()
        populations.append(ga.population.copy())
    numpy.testing.assert_array_equal(*populations)
