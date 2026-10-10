"""Tests for generating and accepting initial populations."""

import copy

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return float(numpy.sum(solution))


def make_ga(**options):
    parameters = dict(num_generations=1, num_parents_mating=2,
                      fitness_func=fitness_func, sol_per_pop=50, num_genes=3,
                      mutation_type=None, crossover_type=None,
                      random_seed=7, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


@pytest.mark.parametrize("container", [list, tuple, numpy.array])
def test_supplied_population_determines_dimensions_before_per_gene_validation(container):
    population = container([[1, 2, 3], [4, 5, 6]])
    original_population = copy.deepcopy(population)
    ga_instance = make_ga(initial_population=population, sol_per_pop=-1,
                          num_genes=99, gene_type=(int, float, numpy.int8),
                          gene_space=[[1, 4], [2, 5], [3, 6]],
                          init_range_low=[0, 0, 0], init_range_high=[10, 10, 10],
                          gene_constraint=[None, None, None])
    assert ga_instance.sol_per_pop == 2
    assert ga_instance.num_genes == 3
    assert ga_instance.pop_size == (2, 3)
    numpy.testing.assert_array_equal(population, original_population)
    numpy.testing.assert_array_equal(ga_instance.population, original_population)
    assert not numpy.shares_memory(ga_instance.population, ga_instance.initial_population)


@pytest.mark.parametrize("population", [[], [[]], [[1], [2, 3]], [1, 2],
                                        numpy.empty((0, 3)), numpy.empty((3, 0)),
                                        numpy.zeros((2, 2, 2))])
def test_invalid_population_shape_has_a_descriptive_error(population):
    with pytest.raises(ValueError, match="non-empty rectangular 2D"):
        make_ga(initial_population=population, gene_type=(int, float, int))


@pytest.mark.parametrize("population", ["population", 5, [[None]], [[True]],
                                        [["1"]], [[1 + 2j]]])
def test_invalid_population_type_is_rejected(population):
    with pytest.raises(TypeError, match="initial population|initial_population"):
        make_ga(initial_population=population)


@pytest.mark.parametrize("gene_type", [[int, [numpy.float32, 2], numpy.int8],
                                       (int, (numpy.float32, 2), numpy.int8),
                                       numpy.array([int, [numpy.float32, 2], numpy.int8], dtype=object)])
def test_gene_type_specifications_are_not_modified(gene_type):
    original_specification = copy.deepcopy(gene_type)
    ga_instance = make_ga(initial_population=((2**53 + 1, 1.236, 7), (2**53 + 3, 2.341, 8)),
                          gene_type=gene_type)
    assert ga_instance.population[0, 0] == 2**53 + 1
    assert ga_instance.population[1, 0] == 2**53 + 3
    assert type(ga_instance.population[0, 1]) is numpy.float32
    assert type(ga_instance.population[0, 2]) is numpy.int8
    assert ga_instance.population[0, 1] == numpy.float32(1.24)
    for original, actual in zip(original_specification, gene_type):
        if isinstance(original, (list, tuple, numpy.ndarray)):
            assert list(original) == list(actual)
        else:
            assert original is actual


@pytest.mark.parametrize("gene_type", [(float, None, ), ((float, None), int, float)])
def test_float_types_accept_an_explicit_none_precision(gene_type):
    ga_instance = make_ga(gene_type=gene_type)
    assert ga_instance.population.shape == (50, 3)


@pytest.mark.parametrize("gene_type", [int, numpy.int8, numpy.uint8, float,
                                       [float, 1], [numpy.float16, 1],
                                       [numpy.float16, 5], numpy.float32])
@pytest.mark.parametrize("bounds", [(0.2, 3.8), (-3.8, -0.2), (3.8, 0.2)])
@pytest.mark.parametrize("use_gene_space", [False, True])
def test_range_values_remain_inside_bounds_after_conversion(gene_type, bounds, use_gene_space):
    lower, upper = bounds
    if gene_type is numpy.uint8 and max(bounds) < 0:
        with pytest.raises(ValueError, match="representable"):
            make_ga(gene_type=gene_type, init_range_low=lower, init_range_high=upper)
        return
    options = dict(gene_type=gene_type, init_range_low=lower, init_range_high=upper)
    if use_gene_space:
        options['gene_space'] = {'low': lower, 'high': upper}
    ga_instance = make_ga(**options)
    assert numpy.all(numpy.asarray(ga_instance.population, dtype=float) >= min(bounds))
    assert numpy.all(numpy.asarray(ga_instance.population, dtype=float) < max(bounds))


@pytest.mark.parametrize("gene_space", [None, [None, None, None],
                                        [[None], [None], [None]]])
def test_none_entries_follow_their_own_ranges(gene_space):
    ga_instance = make_ga(gene_space=gene_space, gene_type=(int, (float, 1), numpy.float32),
                          init_range_low=[2.2, -1.3, 20.0],
                          init_range_high=[6.1, -0.2, 21.0])
    for gene_index, (lower, upper) in enumerate(zip([2.2, -1.3, 20.0], [6.1, -0.2, 21.0])):
        column = ga_instance.population[:, gene_index]
        assert numpy.all(column >= lower) and numpy.all(column < upper)
        assert len(numpy.unique(column)) > 1


@pytest.mark.parametrize("gene_space", [[10, 20], (10, 20), range(10, 21, 10),
                                        numpy.array([10, 20]),
                                        {'low': 10, 'high': 21, 'step': 10},
                                        {'low': 20, 'high': 9, 'step': -10}])
def test_finite_spaces_override_initialization_ranges(gene_space):
    ga_instance = make_ga(gene_space=gene_space, gene_type=int,
                          init_range_low=-5, init_range_high=-1)
    assert set(ga_instance.population.flat) == {10, 20}


def test_nested_spaces_support_fixed_values_dictionaries_and_none_choices():
    space = [5, {'low': 10, 'high': 13, 'step': 1}, [None, 100]]
    original_space = copy.deepcopy(space)
    ga_instance = make_ga(gene_space=space, gene_type=int,
                          init_range_low=[0, 0, 20], init_range_high=[1, 1, 30])
    assert numpy.all(ga_instance.population[:, 0] == 5)
    assert set(ga_instance.population[:, 1]).issubset({10, 11, 12})
    values = set(ga_instance.population[:, 2])
    assert 100 in values and len(values) > 2
    assert values.issubset(set(range(20, 30)) | {100})
    assert space == original_space


@pytest.mark.parametrize("gene_space", [None, {'low': 0, 'high': 10**9},
                                        [[None], [None], [None]]])
def test_large_integer_ranges_do_not_allocate_the_whole_domain(monkeypatch, gene_space):
    def fail_if_enumerated(*args, **kwargs):
        raise AssertionError("The initialization range must be sampled directly.")
    monkeypatch.setattr(numpy, 'arange', fail_if_enumerated)
    ga_instance = make_ga(gene_type=int, gene_space=gene_space,
                          init_range_low=0, init_range_high=10**9)
    assert ga_instance.population.shape == (50, 3)


def test_large_integer_bounds_preserve_exact_values():
    ga_instance = make_ga(gene_type=int, init_range_low=2**53 + 1,
                          init_range_high=2**53 + 4)
    assert set(ga_instance.population.flat) == {2**53 + 1, 2**53 + 2, 2**53 + 3}


def test_large_integer_constraint_search_does_not_allocate_the_whole_domain(monkeypatch):
    def fail_if_enumerated(*args, **kwargs):
        raise AssertionError("Large initialization intervals must be sampled for constraints.")
    monkeypatch.setattr(numpy, 'arange', fail_if_enumerated)
    ga_instance = make_ga(gene_type=int, init_range_low=0, init_range_high=10**9,
                          gene_constraint=[lambda solution, values: [], None, None])
    assert ga_instance.population.shape == (50, 3)


@pytest.mark.parametrize("gene_type", [numpy.int64, numpy.uint64])
@pytest.mark.parametrize("use_gene_space", [False, True])
def test_integer_type_limits_do_not_overflow_during_sampling(gene_type, use_gene_space):
    upper = int(numpy.iinfo(gene_type).max) + 1
    space = {'low': upper - 3, 'high': upper} if use_gene_space else None
    ga_instance = make_ga(gene_type=gene_type, init_range_low=upper - 3,
                          init_range_high=upper, gene_space=space)
    assert set(int(value) for value in ga_instance.population.flat) == {upper - 3, upper - 2, upper - 1}
    if use_gene_space:
        assert set(int(value) for value in ga_instance.gene_space_unpacked) == {upper - 3, upper - 2, upper - 1}


@pytest.mark.parametrize("gene_type,lower,upper", [(int, 0.1, 0.9),
                                                  (numpy.int8, 128, 130),
                                                  ([float, 1], 0.01, 0.09),
                                                  (int, 1.5, 1.5)])
def test_ranges_without_representable_values_fail_clearly(gene_type, lower, upper):
    with pytest.raises(ValueError, match="representable.*gene"):
        make_ga(gene_type=gene_type, init_range_low=lower, init_range_high=upper)


@pytest.mark.parametrize("gene_type", [int, float, [float, 1]])
def test_equal_bounds_generate_a_fixed_representable_value(gene_type):
    ga_instance = make_ga(gene_type=gene_type, init_range_low=2, init_range_high=2)
    assert numpy.all(ga_instance.population == 2)


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.float32, numpy.float64])
def test_rounding_uses_the_stored_numpy_value_at_decimal_bounds(dtype):
    ga_instance = make_ga(gene_type=[dtype, 1], init_range_low=0.1, init_range_high=0.2)
    assert numpy.all(numpy.asarray(ga_instance.population, dtype=float) >= 0.1)
    assert numpy.all(numpy.asarray(ga_instance.population, dtype=float) < 0.2)


def test_supplied_values_outside_the_generation_domain_are_preserved():
    ga_instance = make_ga(initial_population=((100, -100, 30), (20, 30, 40)),
                          gene_space=[0, 1, 2], gene_type=int,
                          init_range_low=0, init_range_high=3)
    numpy.testing.assert_array_equal(ga_instance.population, [[100, -100, 30], [20, 30, 40]])


def test_supplied_population_constraints_use_initialization_candidates():
    population = [[1.236, 1, 1], [1.236, 1, 1]]
    ga_instance = make_ga(initial_population=population, gene_type=([float, 2], int, int),
                          gene_space=[[1.236], [2, 3], [4, 5]], sample_size=1,
                          gene_constraint=[lambda solution, values: [value for value in values if value == 1.24],
                                           lambda solution, values: [value for value in values if value > 2],
                                           lambda solution, values: [value for value in values if value > solution[1]]])
    assert numpy.all(ga_instance.population[:, 0] == 1.24)
    assert numpy.all(ga_instance.population[:, 1] == 3)
    assert numpy.all(ga_instance.population[:, 2] > 3)
    assert population == [[1.236, 1, 1], [1.236, 1, 1]]


def test_failed_constraints_warn_without_modifying_the_supplied_input():
    with pytest.warns(UserWarning, match="No value satisfied"):
        ga_instance = make_ga(initial_population=[[1, 2, 3], [1, 2, 3]],
                              gene_space=[1, 2, 3], suppress_warnings=False,
                              gene_constraint=[lambda solution, values: [], None, None])
    assert numpy.all(ga_instance.population[:, 0] == 1)


def test_constraints_receive_an_independent_complete_solution():
    def constraint(solution, values):
        assert len(solution) == 3 and all(value is not None for value in solution)
        solution[1] = -100
        return values
    ga_instance = make_ga(initial_population=[[1, 2, 3], [1, 2, 3]],
                          gene_constraint=[constraint, None, None])
    assert numpy.all(ga_instance.population[:, 1] == 2)


def test_growth_and_initialization_use_the_same_rules():
    ga_instance = make_ga(num_genes=4, gene_type=(int, int, int, int),
                          gene_space=[[0, 1], [1, 2], [2, 3], [0]],
                          allow_duplicate_genes=False,
                          gene_constraint=[None, None, lambda solution, values: [value for value in values if value >= 2], None])
    extra = ga_instance._nsga3_generate_extra_random_solutions(12)
    assert extra.shape == (12, 4)
    for solution in numpy.vstack([ga_instance.population, extra]):
        assert len(set(solution)) == 4
        assert solution[3] == 0 and solution[2] >= 2


def test_reinitialization_uses_its_constraint_and_duplicate_settings():
    ga_instance = make_ga(gene_space=[1, 2, 3], gene_type=int)
    constraints = [lambda solution, values: [value for value in values if value == 3], None, None]
    ga_instance.initialize_population(False, ga_instance.gene_type, constraints)
    assert numpy.all(ga_instance.population[:, 0] == 3)
    assert all(len(set(solution)) == 3 for solution in ga_instance.population)
    numpy.testing.assert_array_equal(ga_instance.population, ga_instance.initial_population)
    assert not numpy.shares_memory(ga_instance.population, ga_instance.initial_population)


def test_seed_reproduces_generated_population_with_none_entries():
    options = dict(gene_space=[[1, None], {'low': -2, 'high': 3}, range(5)],
                   gene_type=(int, [float, 2], int))
    first = make_ga(**options)
    second = make_ga(**options)
    numpy.testing.assert_array_equal(first.population, second.population)


@pytest.mark.parametrize("gene_space", [None, {'low': 0.0, 'high': 1.0}])
def test_continuous_population_keeps_solution_then_gene_draw_order(gene_space):
    numpy.random.seed(7)
    expected = numpy.random.uniform(0, 1, size=(50, 3))
    ga_instance = make_ga(gene_space=gene_space, init_range_low=0, init_range_high=1)
    numpy.testing.assert_array_equal(ga_instance.population, expected)


@pytest.mark.parametrize("result", [lambda values: [values[0], values[0]],
                                    lambda values: [1000], lambda values: None])
def test_invalid_constraint_outputs_are_rejected(result):
    with pytest.raises(Exception, match="constraint"):
        make_ga(gene_constraint=[lambda solution, values: result(values), None, None])


@pytest.mark.parametrize("options,error", [({'init_range_low': [0, 1], 'init_range_high': [2, 3]}, ValueError),
                                           ({'init_range_low': [[0], [1], [2]], 'init_range_high': [2, 3, 4]}, ValueError),
                                           ({'init_range_low': numpy.nan}, ValueError),
                                           ({'gene_space': {'low': '0', 'high': 3}}, TypeError),
                                           ({'gene_space': {'low': 0, 'high': numpy.inf}}, ValueError),
                                           ({'gene_space': {'low': 0, 'high': 3, 'step': 0}}, ValueError),
                                           ({'gene_space': [{'low': 0, 'high': 3, 'step': -1}]}, ValueError)])
def test_invalid_generation_settings_are_rejected_early(options, error):
    with pytest.raises(error):
        make_ga(**options)
