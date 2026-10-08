"""Regression tests for duplicate repair across the GA lifecycle."""

import copy
import itertools
import random

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_idx):
    return float(numpy.sum(solution))


def make_ga(**options):
    parameters = dict(num_generations=3, num_parents_mating=2,
                      fitness_func=fitness_func, sol_per_pop=4, num_genes=3,
                      gene_type=int, init_range_low=0, init_range_high=6,
                      random_mutation_min_val=0, random_mutation_max_val=6,
                      mutation_by_replacement=True, allow_duplicate_genes=False,
                      random_seed=1, suppress_warnings=True)
    parameters.update(copy.deepcopy(options))
    return pygad.GA(**parameters)


@pytest.mark.parametrize("gene_space", [None, [0, 1, 2], (0, 1, 2), range(3),
                                       numpy.arange(3), {'low': 0, 'high': 3},
                                       {'low': 0, 'high': 3, 'step': 1}])
def test_manual_population_duplicates_are_repaired_without_changing_input(gene_space):
    population = [[0, 0, 1], [0, 0, 1]]
    before = copy.deepcopy(population)
    ga_instance = make_ga(initial_population=population, gene_space=gene_space)
    assert population == before
    for solution in ga_instance.population:
        assert len(set(solution)) == 3


@pytest.mark.parametrize("manual", [False, True])
def test_fixed_later_gene_can_move_an_earlier_duplicate(manual):
    options = dict(gene_space=[[0, 1], [0], [2]])
    if manual:
        options['initial_population'] = [[0, 0, 2], [0, 0, 2]]
    ga_instance = make_ga(**options)
    numpy.testing.assert_array_equal(ga_instance.population,
                                     numpy.tile([1, 0, 2], (ga_instance.sol_per_pop, 1)))


def test_long_replacement_chain_does_not_use_python_recursion():
    num_genes = 1100
    spaces = [[index, index + 1] for index in range(num_genes - 1)] + [[0]]
    solution = numpy.array(list(range(num_genes - 1)) + [0])
    ga_instance = make_ga(num_genes=num_genes, gene_space=spaces, sample_size=1,
                          initial_population=[solution.tolist(), solution.tolist()])
    numpy.testing.assert_array_equal(ga_instance.population[0],
                                     list(range(1, num_genes)) + [0])


def test_finite_repair_matches_exhaustive_search_for_small_spaces():
    # Compare with all assignments, including spaces that cannot make
    # every gene unique. The repair should maximize distinct values.
    rng = random.Random(42)
    for _ in range(100):
        spaces = [rng.sample(range(4), rng.randint(1, 3)) for _ in range(4)]
        solution = [rng.choice(space) for space in spaces]
        expected = max(len(set(values)) for values in itertools.product(*spaces))
        ga_instance = make_ga(num_genes=4, gene_space=spaces,
                              initial_population=[solution, solution], sample_size=1)
        for repaired_solution in ga_instance.population:
            assert len(set(repaired_solution)) == expected
            assert all(value in space for value, space in zip(repaired_solution, spaces))


@pytest.mark.parametrize("gene_type", [[numpy.int16, numpy.float32, int],
                                      [int, [float, 0], numpy.int32]])
def test_mixed_gene_types_are_preserved_during_range_repair(gene_type):
    ga_instance = make_ga(gene_type=gene_type,
                          initial_population=[[0, 0, 1], [0, 0, 1]])
    for solution in ga_instance.population:
        assert len(set(solution)) == 3
        for index, value in enumerate(solution):
            assert isinstance(value, ga_instance.get_gene_dtype(index)[0])


@pytest.mark.parametrize("method", ['mutation_randomly', 'mutation_probs_randomly',
                                    'adaptive_mutation_randomly',
                                    'adaptive_mutation_probs_randomly'])
def test_mutation_repairs_each_gene_using_its_own_range(method, monkeypatch):
    adaptive = method.startswith('adaptive')
    probability = [1.0, 1.0] if adaptive else 1.0
    ga_instance = make_ga(initial_population=[[0, 2], [0, 2]],
                          mutation_type='adaptive' if adaptive else 'random',
                          mutation_probability=probability if 'probs' in method else None,
                          mutation_num_genes=[1, 1] if adaptive else 1,
                          random_mutation_min_val=[1, 10],
                          random_mutation_max_val=[3, 12])
    monkeypatch.setattr(random, 'sample', lambda values, count: [0])
    monkeypatch.setattr(ga_instance, 'mutation_process_gene_value',
                        lambda solution, gene_idx, **kwargs: 2 if gene_idx == 0 else solution[gene_idx])
    if adaptive:
        monkeypatch.setattr(ga_instance, 'adaptive_mutation_population_fitness',
                            lambda offspring: (1.0, numpy.ones(len(offspring))))
    result = getattr(ga_instance, method)(numpy.array([[0, 2]]))
    assert len(set(result[0])) == 2
    assert 10 <= result[0, 1] < 12


@pytest.mark.parametrize("gene_space, gene_type", [(None, float),
                                                   ([0, 1, 2], int),
                                                   ({'low': 0, 'high': 3, 'step': 1}, int)])
def test_sample_size_one_accepts_scalar_candidates(gene_space, gene_type):
    ga_instance = make_ga(gene_space=gene_space, gene_type=gene_type,
                          sample_size=1, initial_population=[[0, 0, 1], [0, 0, 1]])
    assert all(len(set(solution)) == 3 for solution in ga_instance.population)


@pytest.mark.parametrize("space", [None, [None], [None, 100], (None, 100),
                                   numpy.array([None, 100], dtype=object)])
def test_none_entries_generate_fresh_values_from_per_gene_ranges(space):
    ga_instance = make_ga(gene_space=[space, [100], [200]], gene_type=float,
                          init_range_low=[0, 100, 200], init_range_high=[1, 101, 201],
                          random_mutation_min_val=[10, 100, 200],
                          random_mutation_max_val=[11, 101, 201])
    candidates = ga_instance.get_gene_space_values(0, gene_value=0.5, sample_size=10)
    random_candidates = candidates[candidates != 100]
    assert len(random_candidates) > 1
    assert numpy.all((random_candidates >= 10) & (random_candidates < 11))


def test_flat_none_space_with_mixed_types_and_per_gene_ranges():
    ga_instance = make_ga(gene_space=[None, 100], gene_type=[int, float, numpy.int16],
                          init_range_low=[0, 10, 20], init_range_high=[3, 13, 23])
    for solution in ga_instance.population:
        assert len(set(solution)) == 3
    assert set(ga_instance.get_gene_space_values(2)) == {20, 21, 22, 100}


def test_repair_does_not_invalidate_a_constraint_on_another_gene():
    constraint = lambda solution, values: [value for value in values if value >= solution[1]]
    ga_instance = make_ga(gene_space=[[0], [0, 1], [2]],
                          gene_constraint=[constraint, None, None])
    before = numpy.array([0, 0, 2])
    repaired, duplicates, count = ga_instance.solve_duplicate_genes(before)
    numpy.testing.assert_array_equal(repaired, before)
    assert ga_instance.solution_satisfies_gene_constraints(repaired)
    assert duplicates == {1}
    assert count == 1


def test_constraint_can_require_changing_another_nonduplicated_gene():
    constraint = lambda solution, values: [value for value in values if value <= solution[2]]
    ga_instance = make_ga(gene_space=[[0], [0, 1], [0, 2]],
                          gene_constraint=[None, constraint, None])
    repaired, duplicates, count = ga_instance.solve_duplicate_genes(numpy.array([0, 0, 0]))
    numpy.testing.assert_array_equal(repaired, [0, 1, 2])
    assert duplicates == set()
    assert count == 0


def test_impossible_initial_space_warns_and_returns_remaining_duplicates():
    with pytest.warns(UserWarning, match='Failed to find a unique value'):
        ga_instance = make_ga(gene_space=[0], suppress_warnings=False)
    repaired, duplicates, count = ga_instance.solve_duplicate_genes(
        [0, 0, 0], build_initial_pop=True, warn=False)
    assert repaired.tolist() == [0, 0, 0]
    assert duplicates == {1, 2}
    assert count == 2


@pytest.mark.parametrize("crossover_type", ['single_point', 'two_points', 'uniform',
                                           'scattered'])
def test_crossover_can_repair_a_chain_without_mutation(crossover_type):
    ga_instance = make_ga(num_genes=4, gene_space=[[0, 1], [1, 2], [2, 3], [0]],
                          crossover_type=crossover_type, mutation_type=None)
    offspring = ga_instance.crossover(numpy.array([[0, 1, 2, 0], [0, 1, 2, 0]]), (2, 4))
    numpy.testing.assert_array_equal(offspring, [[1, 2, 3, 0], [1, 2, 3, 0]])


@pytest.mark.parametrize("stage", ['crossover', 'mutation', 'on_crossover', 'on_mutation'])
def test_user_operator_and_callback_outputs_are_repaired(stage):
    def crossover(parents, offspring_size, ga_instance):
        return numpy.zeros(offspring_size)

    def mutation(offspring, ga_instance):
        return numpy.zeros_like(offspring)

    def callback(ga_instance, offspring):
        offspring[:] = 0

    options = dict(gene_space=[0, 1, 2], crossover_type=None, mutation_type=None)
    if stage == 'crossover':
        options['crossover_type'] = crossover
    elif stage == 'mutation':
        options['mutation_type'] = mutation
    else:
        options[stage] = callback
    ga_instance = make_ga(**options)
    ga_instance.run()
    assert all(len(set(solution)) == 3 for solution in ga_instance.population)


def test_rounding_is_applied_before_duplicate_repair():
    ga_instance = make_ga(gene_type=[float, 0], gene_space=[0, 1, 2])
    result = ga_instance.solve_duplicate_genes_in_population(numpy.array([[0.1, 0.2, 1.1]]))
    assert len(set(result[0])) == 3
    assert set(result[0]) == {0, 1, 2}


@pytest.mark.parametrize("method", ['swap_mutation', 'inversion_mutation',
                                    'scramble_mutation'])
def test_permutation_mutation_repairs_duplicates_created_by_destination_casts(method,
                                                                             monkeypatch):
    ga_instance = make_ga(num_genes=4, gene_type=[float, int, int, int], mutation_type=method.split('_')[0],
                          initial_population=[[0.5, 1, 0, 2], [0.5, 1, 0, 2]])
    if method == 'swap_mutation':
        monkeypatch.setattr(numpy.random, 'choice', lambda *args, **kwargs: numpy.array([0, 1]))
    else:
        monkeypatch.setattr(numpy.random, 'randint', lambda *args, **kwargs: numpy.array([0]))
        if method == 'scramble_mutation':
            monkeypatch.setattr(numpy.random, 'shuffle', lambda values: values.__setitem__(slice(None), values[::-1].copy()))
    result = getattr(ga_instance, method)(ga_instance.population.copy())
    for solution in result:
        assert len(set(solution)) == 4
        for gene_index, value in enumerate(solution):
            assert isinstance(value, ga_instance.get_gene_dtype(gene_index)[0])


@pytest.mark.parametrize("operator", ['sbx', 'polynomial'])
def test_bounded_operators_round_values_and_repair_using_their_own_bounds(operator):
    ga_instance = make_ga(gene_type=[float, 0], crossover_type='sbx',
                          mutation_type='polynomial', mutation_probability=1.0,
                          init_range_low=0, init_range_high=3,
                          random_mutation_min_val=100, random_mutation_max_val=200,
                          initial_population=[[0, 1, 2], [2, 0, 1]])
    if operator == 'sbx':
        result = ga_instance.sbx_crossover(ga_instance.population, (20, 3))
    else:
        result = ga_instance.polynomial_mutation(numpy.tile([0, 1, 2], (20, 1)).astype(float))
    for solution in result:
        assert len(set(solution)) == 3
        assert numpy.all((solution >= 0) & (solution <= 3))
        numpy.testing.assert_array_equal(solution, numpy.round(solution))


def test_constraint_sample_size_one_does_not_crash():
    constraints = [lambda solution, values: values] * 3
    ga_instance = make_ga(gene_constraint=constraints, sample_size=1,
                          initial_population=[[0, 0, 1], [0, 0, 1]])
    assert all(len(set(solution)) == 3 for solution in ga_instance.population)


@pytest.mark.parametrize("gene_type", [int, float, numpy.int16, [float, 1]])
def test_equal_random_bounds_keep_values_and_report_impossible_duplicates(gene_type):
    ga_instance = make_ga(gene_type=gene_type, init_range_low=0, init_range_high=0,
                          random_mutation_min_val=0, random_mutation_max_val=0)
    repaired, duplicates, count = ga_instance.solve_duplicate_genes([0, 0, 0])
    assert repaired.tolist() == [0, 0, 0]
    assert duplicates == {1, 2}
    assert count == 2


def test_reversed_integer_bounds_remain_usable_for_initialization_and_repair():
    ga_instance = make_ga(init_range_low=3, init_range_high=0,
                          random_mutation_min_val=3, random_mutation_max_val=0)
    assert all(set(solution) == {0, 1, 2} for solution in ga_instance.population)
    repaired, duplicates, count = ga_instance.solve_duplicate_genes([0, 0, 0])
    assert set(repaired) == {0, 1, 2}
    assert duplicates == set()
    assert count == 0


@pytest.mark.parametrize("gene_type", [numpy.float16, numpy.float32, [numpy.float16, 1]])
def test_none_candidates_support_small_float_types_and_precision(gene_type):
    ga_instance = make_ga(gene_type=gene_type, gene_space=[[None], [None], [None]])
    assert all(len(set(solution)) == 3 for solution in ga_instance.population)


@pytest.mark.parametrize("space", [None, [None], {'low': 0, 'high': 2}])
def test_swap_fallback_uses_current_none_ranges_and_continuous_bounds(space):
    ga_instance = make_ga(num_genes=2, gene_space=[space, [0, 1]],
                          init_range_low=10, init_range_high=12,
                          random_mutation_min_val=0, random_mutation_max_val=2,
                          gene_type=float, initial_population=[[0, 1], [0, 1]])
    solution = numpy.array([0.0, 1.0])
    numpy.testing.assert_array_equal(ga_instance.swap_gene_by_space(solution, 0), [1, 0])


def test_integer_none_mutation_adds_the_offset_before_casting(monkeypatch):
    ga_instance = make_ga(gene_space=[None, [100], [200]],
                          initial_population=[[-2, 100, 200], [-2, 100, 200]],
                          mutation_by_replacement=False,
                          random_mutation_min_val=-1, random_mutation_max_val=1)
    monkeypatch.setattr(numpy.random, 'uniform', lambda *args, **kwargs: 0.75)
    value = ga_instance.generate_gene_value_from_space(
        0, False, ga_instance.population[0], gene_value=-2, sample_size=1)
    assert value == -1


def test_integer_dictionary_with_fractional_bounds_includes_all_converted_values():
    ga_instance = make_ga(num_genes=5, gene_space={'low': 0.8, 'high': 4.2},
                          initial_population=[[0, 0, 1, 2, 3], [0, 0, 1, 2, 3]],
                          sample_size=1)
    assert set(ga_instance.get_gene_space_values(0)) == {0, 1, 2, 3, 4}
    assert all(set(solution) == {0, 1, 2, 3, 4} for solution in ga_instance.population)


def test_mixed_float_precision_compares_exact_numeric_values():
    ga_instance = make_ga(num_genes=2, gene_type=[numpy.float32, float],
                          gene_space=[[0.1, 1], [0.1, 1]],
                          initial_population=[[0.1, 0.1], [0.1, 0.1]])
    # float32(0.1) and Python's float(0.1) have different stored values.
    # NumPy scalar comparison can promote the Python float to float32.
    assert ga_instance.get_duplicate_gene_indices(ga_instance.population[0]) == set()
    assert ga_instance.get_duplicate_gene_indices([numpy.float32(1), 1.0]) == {1}
    assert ga_instance.find_two_duplicates(ga_instance.population[0],
                                           ga_instance.gene_space_unpacked) == (None, None)
    value = ga_instance.select_unique_value([numpy.float32(0.1)], [1.0, 0.1], 0)
    assert isinstance(value, numpy.float32)
    assert float(value) != 0.1


def test_repeated_nans_can_be_repaired_from_a_finite_space():
    ga_instance = make_ga(gene_type=float, gene_space=[0, 1, 2],
                          initial_population=[[numpy.nan, numpy.nan, 0],
                                              [numpy.nan, numpy.nan, 0]])
    for solution in ga_instance.population:
        assert ga_instance.get_duplicate_gene_indices(solution) == set()
        assert numpy.count_nonzero(numpy.isnan(solution)) == 1


def test_repair_and_full_runs_are_reproducible():
    populations = []
    for _ in range(2):
        ga_instance = make_ga(gene_space=[[0, 1], [1, 2], [0, 2]],
                              initial_population=[[0, 1, 0], [1, 1, 2]])
        ga_instance.run()
        populations.append(ga_instance.population)
    numpy.testing.assert_array_equal(*populations)
