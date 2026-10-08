"""Regression coverage for random and adaptive permutation mutation."""

import copy

import numpy
import pytest

import pygad


SPACE_MUTATIONS = ["mutation_by_space", "mutation_probs_by_space",
                   "adaptive_mutation_by_space", "adaptive_mutation_probs_by_space"]


def _fitness(ga, solution, index):
    return float(numpy.sum(solution))


def _make_ga(method="mutation_by_space", num_genes=2, **options):
    adaptive = method.startswith("adaptive")
    parameters = dict(num_generations=3, num_parents_mating=2,
                      fitness_func=_fitness, sol_per_pop=4,
                      num_genes=num_genes, gene_type=int,
                      gene_space=list(range(num_genes)),
                      allow_duplicate_genes=False, crossover_type=None,
                      mutation_type="adaptive" if adaptive else "random",
                      random_seed=1, suppress_warnings=True)
    if "probs" in method:
        parameters["mutation_probability"] = [1.0, 1.0] if adaptive else 1.0
    else:
        parameters["mutation_num_genes"] = [num_genes, num_genes] if adaptive else num_genes
    parameters.update(copy.deepcopy(options))
    return pygad.GA(**parameters)


def _mutate(ga, method, offspring, monkeypatch, multi_objective=False):
    if method.startswith("adaptive"):
        fitness = numpy.resize([0.0, 2.0], len(offspring))
        average = 1.0
        if multi_objective:
            fitness = numpy.column_stack([fitness, fitness])
            average = numpy.array([1.0, 1.0])
        monkeypatch.setattr(ga, "adaptive_mutation_population_fitness",
                            lambda population: (average, fitness))
    return getattr(ga, method)(offspring)


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
@pytest.mark.parametrize("multi_objective", [False, True])
def test_two_gene_permutation_mutation_is_not_undone(method, multi_objective,
                                                    monkeypatch):
    ga = _make_ga(method)
    offspring = numpy.array([[0, 1], [1, 0]])
    before = offspring.copy()
    result = _mutate(ga, method, offspring, monkeypatch, multi_objective)

    assert result is offspring
    numpy.testing.assert_array_equal(result, before[:, ::-1])


@pytest.mark.parametrize("gene_types", [[int, float], [float, int],
                                       [numpy.int16, numpy.float32]])
def test_swap_fallback_preserves_destination_gene_types(gene_types):
    ga = _make_ga(gene_type=gene_types)
    solution = numpy.array([gene_types[0](0), gene_types[1](1)], dtype=object)
    result = ga.swap_gene_by_space(solution, 0)

    assert result is solution
    assert result.tolist() == [1, 0]
    for value, dtype in zip(result, gene_types):
        assert numpy.asarray(value).dtype == numpy.dtype(dtype)


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
def test_mutation_preserves_mixed_gene_types_and_uniqueness(method, monkeypatch):
    ga = _make_ga(method, gene_type=[numpy.int16, numpy.float32])
    offspring = numpy.array([[numpy.int16(0), numpy.float32(1)],
                             [numpy.int16(1), numpy.float32(0)]], dtype=object)
    before = offspring.copy()
    result = _mutate(ga, method, offspring, monkeypatch)

    numpy.testing.assert_array_equal(result, before[:, ::-1])
    for row in result:
        assert isinstance(row[0], numpy.int16)
        assert isinstance(row[1], numpy.float32)
        assert len(set(row)) == len(row)


@pytest.mark.parametrize("num_genes", [3, 8])
@pytest.mark.parametrize("method", SPACE_MUTATIONS)
def test_full_mutation_changes_permutations_in_flat_and_nested_spaces(method,
                                                                     num_genes,
                                                                     monkeypatch):
    space = list(range(num_genes))
    ga = _make_ga(method, num_genes, gene_space=[space] * num_genes)
    offspring = numpy.array([numpy.random.permutation(num_genes) for _ in range(20)])
    before = offspring.copy()
    result = _mutate(ga, method, offspring, monkeypatch)

    numpy.testing.assert_array_equal(numpy.sort(result, axis=1),
                                     numpy.tile(space, (len(result), 1)))
    assert numpy.all(numpy.any(result != before, axis=1))


def test_swap_fallback_requires_both_destination_spaces_to_allow_the_swap():
    ga = _make_ga(gene_space=[[0, 1], [1]])
    solution = numpy.array([0, 1])
    numpy.testing.assert_array_equal(ga.swap_gene_by_space(solution, 0), [0, 1])


def test_swap_fallback_skips_casts_that_would_create_duplicates():
    ga = _make_ga(num_genes=3, gene_type=[int, float, int],
                  gene_space=[[0, 1], [0.0, 1.9], [1]])
    solution = numpy.array([0, 1.9, 1], dtype=object)
    before = solution.copy()
    ga.swap_gene_by_space(solution, 0)
    numpy.testing.assert_array_equal(solution, before)
    assert len(set(solution)) == len(solution)


def test_swap_fallback_skips_casts_that_would_change_permutation_values():
    ga = _make_ga(gene_type=[int, float], gene_space=[[0, 1], [0.0, 1.9]])
    solution = numpy.array([0, 1.9], dtype=object)
    before = solution.copy()
    ga.swap_gene_by_space(solution, 0)
    numpy.testing.assert_array_equal(solution, before)


def test_swap_fallback_skips_rounding_that_would_change_values():
    ga = _make_ga(gene_type=[[float, 1], [float, 2]],
                  gene_space=[[0.0, 0.1], [0.0, 0.14]])
    solution = numpy.array([0.0, 0.14], dtype=object)
    before = solution.copy()
    ga.swap_gene_by_space(solution, 0)
    numpy.testing.assert_array_equal(solution, before)


@pytest.mark.parametrize("constrained_gene", [0, 1])
def test_swap_fallback_does_not_swap_constrained_genes(constrained_gene):
    constraints = [None, None]
    constraints[constrained_gene] = lambda solution, values: values
    ga = _make_ga(gene_constraint=constraints)
    solution = numpy.array([0, 1])
    numpy.testing.assert_array_equal(ga.swap_gene_by_space(solution, 0), [0, 1])


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
def test_swap_fallback_preserves_constraints_depending_on_other_genes(method,
                                                                     monkeypatch):
    constraints = [None, None,
                   lambda solution, values: [value for value in values
                                             if value > solution[0]]]
    ga = _make_ga(method, 3, gene_constraint=constraints)
    # The only possible swap for gene 0 would invalidate gene 2's constraint.
    solution = numpy.array([[0, 2, 1], [0, 2, 1]])
    before = solution.copy()
    numpy.testing.assert_array_equal(_mutate(ga, method, solution, monkeypatch), before)


def test_swap_fallback_allows_swaps_that_preserve_other_genes_constraints():
    ga = _make_ga(num_genes=3, gene_constraint=[None, None,
                   lambda solution, values: [value for value in values
                                             if value > solution[0]]])
    solution = numpy.array([0, 1, 2])
    numpy.testing.assert_array_equal(ga.swap_gene_by_space(solution, 0), [1, 0, 2])


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
@pytest.mark.parametrize("allow_duplicates", [False, True])
def test_available_values_and_duplicate_allowed_mutations_do_not_use_fallback(
        method, allow_duplicates, monkeypatch):
    ga = _make_ga(method, gene_space=[0, 1, 2],
                  allow_duplicate_genes=allow_duplicates)

    def unexpected_swap(*args, **kwargs):
        pytest.fail("Fallback should not run when a replacement can be chosen")

    monkeypatch.setattr(ga, "swap_gene_by_space", unexpected_swap)
    solution = numpy.array([[0, 1], [1, 0]])
    result = _mutate(ga, method, solution, monkeypatch)
    assert numpy.all(numpy.isin(result, [0, 1, 2]))
    if not allow_duplicates:
        assert numpy.all(result[:, 0] != result[:, 1])


def test_swap_fallback_tracks_swaps_within_one_pass_only():
    ga = _make_ga()
    solution = numpy.array([0, 1])
    swapped_genes = set()
    ga.swap_gene_by_space(solution, 0, swapped_genes=swapped_genes)
    assert swapped_genes == {0, 1}
    ga.swap_gene_by_space(solution, 1, swapped_genes=swapped_genes)
    numpy.testing.assert_array_equal(solution, [1, 0])
    ga.swap_gene_by_space(solution, 1, swapped_genes=set())
    numpy.testing.assert_array_equal(solution, [0, 1])


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
def test_single_gene_permutation_stays_unchanged(method, monkeypatch):
    ga = _make_ga(method, 1)
    solution = numpy.zeros((2, 1), dtype=int)
    numpy.testing.assert_array_equal(_mutate(ga, method, solution, monkeypatch),
                                     numpy.zeros((2, 1), dtype=int))


@pytest.mark.parametrize("method", ["mutation_probs_by_space",
                                    "adaptive_mutation_probs_by_space"])
def test_zero_probability_does_not_trigger_swaps(method, monkeypatch):
    probability = [0.0, 0.0] if method.startswith("adaptive") else 0.0
    ga = _make_ga(method, mutation_probability=probability)
    solution = numpy.array([[0, 1], [1, 0]])
    before = solution.copy()
    numpy.testing.assert_array_equal(_mutate(ga, method, solution, monkeypatch), before)


@pytest.mark.parametrize("method", SPACE_MUTATIONS)
def test_full_ga_run_keeps_permutations_with_correct_gene_types(method):
    populations = []
    for _ in range(2):
        ga = _make_ga(method, gene_type=[numpy.int16, numpy.float32],
                      keep_elitism=0, keep_parents=0)
        ga.run()
        for solution in ga.population:
            assert sorted(solution) == [0, 1]
            assert isinstance(solution[0], numpy.int16)
            assert isinstance(solution[1], numpy.float32)
        populations.append(ga.population.copy())
    numpy.testing.assert_array_equal(*populations)
