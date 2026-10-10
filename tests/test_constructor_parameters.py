"""Constructor validation, parameter ownership, and operator compatibility."""

import functools
import logging
import random
import warnings
from unittest.mock import Mock

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return float(sum(solution))


def make_ga(**options):
    parameters = dict(num_generations=2, num_parents_mating=2, fitness_func=fitness_func,
                      sol_per_pop=4, num_genes=8, random_seed=7, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


@pytest.mark.parametrize("name", ['num_parents_mating', 'num_generations', 'sol_per_pop', 'num_genes',
                                  'sample_size', 'keep_elitism', 'fitness_batch_size', 'random_seed'])
@pytest.mark.parametrize("value", [True, numpy.bool_(False), 1.5, numpy.nan, '2'])
def test_integer_parameters_reject_non_integer_values(name, value):
    with pytest.raises(TypeError, match=name):
        make_ga(**{name: value})


@pytest.mark.parametrize("dtype", [numpy.int8, numpy.uint8, numpy.int64, numpy.uint64])
def test_numpy_counts_are_python_integers_before_arithmetic(dtype):
    ga_instance = make_ga(num_generations=dtype(200) if dtype is not numpy.int8 else dtype(100),
                          sol_per_pop=dtype(4), num_genes=dtype(3), num_parents_mating=dtype(2),
                          mutation_type=None, crossover_type=None, fitness_batch_size=dtype(2),
                          fitness_func=lambda ga, solutions, indices: [1.0] * len(solutions))
    count = int(ga_instance.num_generations)
    ga_instance.run()
    ga_instance.run()
    assert ga_instance.generations_completed == 2 * count
    for name in ['num_generations', 'sol_per_pop', 'num_genes', 'num_parents_mating', 'fitness_batch_size']:
        assert type(getattr(ga_instance, name)) is int


@pytest.mark.parametrize("adaptive", [False, True])
def test_numpy_percentages_do_not_overflow(adaptive):
    ga_instance = make_ga(num_genes=3, mutation_type='adaptive' if adaptive else 'random',
                          mutation_percent_genes=[numpy.uint8(100), numpy.uint8(100)] if adaptive else numpy.uint8(100))
    assert ga_instance.mutation_num_genes == ([3, 3] if adaptive else 3)


@pytest.mark.parametrize("mutation_type,probability", [('random', 0.5), ('adaptive', [0.8, 0.2])])
def test_probability_takes_precedence_over_inactive_counts_and_percentages(mutation_type, probability):
    ga_instance = make_ga(mutation_type=mutation_type, mutation_probability=probability,
                          mutation_num_genes='ignored', mutation_percent_genes=numpy.array([0, numpy.nan]))
    ga_instance.run()
    assert ga_instance.mutation_num_genes is None


def test_gene_count_takes_precedence_over_inactive_percentage():
    ga_instance = make_ga(mutation_num_genes=2, mutation_percent_genes=numpy.array([0, 0]))
    assert ga_instance.mutation_num_genes == 2


@pytest.mark.parametrize("operator", ['random', 'adaptive', 'swap', 'inversion', 'scramble', 'polynomial'])
def test_zero_probability_keeps_offspring_unchanged(operator):
    ga_instance = make_ga(mutation_type=operator, mutation_probability=[0.0, 0.0] if operator == 'adaptive' else 0.0)
    original = ga_instance.population.copy()
    result = ga_instance.mutation(original.copy())
    numpy.testing.assert_array_equal(result, original)


@pytest.mark.parametrize("operator", ['swap', 'inversion', 'scramble', 'polynomial'])
def test_mutation_preserves_fixed_destination_spaces(operator):
    population = numpy.tile(numpy.arange(8), (4, 1))
    ga_instance = make_ga(initial_population=population, mutation_type=operator, mutation_probability=1.0,
                          gene_space=[[value] for value in range(8)], allow_duplicate_genes=False)
    ga_instance.run()
    numpy.testing.assert_array_equal(ga_instance.population, population)


def fixed_constraint(value):
    def constraint(solution, values):
        return [candidate for candidate in values if candidate == value]
    return constraint


@pytest.mark.parametrize("operator", ['swap', 'inversion', 'scramble', 'polynomial'])
def test_mutation_preserves_fixed_constraints(operator):
    population = numpy.tile(numpy.arange(8), (4, 1))
    ga_instance = make_ga(initial_population=population, mutation_type=operator, mutation_probability=1.0,
                          gene_constraint=[fixed_constraint(value) for value in range(8)], allow_duplicate_genes=False)
    ga_instance.run()
    numpy.testing.assert_array_equal(ga_instance.population, population)


def test_swap_searches_other_pairs_when_the_initial_pair_is_incompatible():
    ga_instance = make_ga(num_genes=3, initial_population=[[0, 1, 2]] * 4,
                          gene_space=[[0], [1, 2], [1, 2]], mutation_type='swap', allow_duplicate_genes=False)
    ga_instance.numpy_random_generator = Mock(wraps=ga_instance.numpy_random_generator)
    ga_instance.numpy_random_generator.choice = lambda *args, **kwargs: numpy.array([0, 1])
    result = ga_instance.swap_mutation(ga_instance.population.copy())
    numpy.testing.assert_array_equal(result, [[0, 2, 1]] * 4)


@pytest.mark.parametrize("operator", ['swap', 'inversion', 'scramble', 'polynomial'])
def test_explicit_gene_counts_limit_mutation_to_eligible_positions(operator):
    ga_instance = make_ga(mutation_type=operator, mutation_num_genes=2, initial_population=[list(range(8))] * 4,
                          init_range_low=0, init_range_high=8)
    ga_instance.python_random_generator.sample = lambda values, count: [1, 2]
    result = ga_instance.mutation(ga_instance.population.copy())
    numpy.testing.assert_array_equal(result[:, [0, 3, 4, 5, 6, 7]], ga_instance.population[:, [0, 3, 4, 5, 6, 7]])


@pytest.mark.parametrize("operator", ['sbx', 'polynomial'])
def test_bounded_operators_use_gene_spaces_outside_default_initialization_bounds(operator):
    ga_instance = make_ga(num_genes=3, initial_population=[[10, 20, 30], [11, 21, 31]] * 2,
                          gene_space=[[10, 11], [20, 21], [30, 31]],
                          crossover_type='sbx' if operator == 'sbx' else None,
                          mutation_type='polynomial' if operator == 'polynomial' else None,
                          mutation_probability=1.0)
    ga_instance.run()
    for solution in ga_instance.population:
        assert all(value in ga_instance.gene_space[index] for index, value in enumerate(solution))


@pytest.mark.parametrize("operator", ['sbx', 'polynomial'])
def test_bounded_operators_handle_reversed_bounds_and_outside_supplied_values(operator):
    ga_instance = make_ga(initial_population=[list(range(10, 18)), list(range(20, 28))] * 2,
                          init_range_low=4, init_range_high=-4, sbx_crossover_eta=1.5,
                          crossover_type='sbx' if operator == 'sbx' else None,
                          mutation_type='polynomial' if operator == 'polynomial' else None,
                          mutation_probability=1.0, keep_elitism=0, keep_parents=0)
    ga_instance.run()
    assert numpy.all(numpy.isfinite(ga_instance.population))
    assert numpy.all((ga_instance.population >= -4) & (ga_instance.population <= 4))


@pytest.mark.parametrize("selection", ['tournament', 'tournament_nsga2', 'tournament_nsga3'])
@pytest.mark.parametrize("value,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)])
def test_all_tournament_operators_validate_their_size(selection, value, error):
    with pytest.raises(error, match='K_tournament'):
        make_ga(parent_selection_type=selection, K_tournament=value, nsga3_num_divisions=1)


@pytest.mark.parametrize("criterion", ['saturate_0', 'saturate_-1', 'saturate_1.5',
                                       'evaluations_0', 'evaluations_-1', 'evaluations_1.5',
                                       'time_-1', 'time_nan', 'reach_inf', 'reach_1..2'])
def test_stopping_criteria_reject_invalid_thresholds(criterion):
    with pytest.raises(ValueError):
        make_ga(stop_criteria=criterion)


def test_stopping_criteria_support_scientific_notation_exact_counts_and_order():
    ga_instance = make_ga(stop_criteria=numpy.array(['reach_1e3', 'saturate_3', 'reach_1e3', 'evaluations_9007199254740993']))
    assert ga_instance.stop_criteria == [['reach', 1000.0], ['saturate', 3], ['evaluations', 9007199254740993]]


@pytest.mark.parametrize("name", ['random_mutation_min_val', 'random_mutation_max_val', 'sbx_crossover_eta', 'polynomial_mutation_eta'])
@pytest.mark.parametrize("value", [numpy.nan, numpy.inf, -numpy.inf])
def test_ranges_and_distribution_indices_must_be_finite(name, value):
    with pytest.raises(ValueError, match=name):
        make_ga(**{name: value})


@pytest.mark.parametrize("name", ['init_range_low', 'random_mutation_min_val', 'gene_space'])
def test_zero_dimensional_parameter_arrays_fail_descriptively(name):
    with pytest.raises(ValueError, match='1D|0D'):
        make_ga(**{name: numpy.array(1)})


class Constraint:
    def method(self, solution, values):
        return values

    def __call__(self, solution, values):
        return values


def constraint_with_option(solution, values, threshold=0):
    return values


@pytest.mark.parametrize("constraint", [Constraint(), Constraint().method,
                                        functools.partial(constraint_with_option, threshold=1)])
def test_constraints_support_bound_methods_callable_objects_and_partials(constraint):
    ga_instance = make_ga(gene_constraint=[constraint] * 8)
    ga_instance.run()


@pytest.mark.parametrize("value", [False, 0, [], ()])
def test_constraint_parameter_does_not_bypass_validation_when_falsey(value):
    with pytest.raises((TypeError, ValueError), match='constraint'):
        make_ga(gene_constraint=value)


def test_callable_validation_checks_positional_arguments_without_executing_functions():
    def keyword_fitness(ga, solution, *, index):
        pytest.fail('Validation must not call fitness functions.')
    def keyword_callback(*, ga):
        pytest.fail('Validation must not call callbacks.')
    with pytest.raises(ValueError, match='fitness_func'):
        make_ga(fitness_func=keyword_fitness)
    with pytest.raises(ValueError, match='on_start'):
        make_ga(on_start=keyword_callback)


def test_functions_can_have_additional_optional_parameters():
    def fitness(ga, solution, index, offset=1):
        return float(sum(solution)) + offset
    def on_generation(ga, unused=None):
        pass
    make_ga(fitness_func=fitness, on_generation=on_generation).run()


def test_invalid_logger_does_not_mask_the_validation_error():
    with pytest.raises(TypeError, match='logger'):
        make_ga(logger='invalid')


def test_adaptive_replacement_is_supported_without_an_incorrect_warning():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        ga_instance = make_ga(mutation_type='adaptive', mutation_probability=[1.0, 1.0],
                              mutation_by_replacement=True, random_mutation_min_val=7,
                              random_mutation_max_val=7, suppress_warnings=False, keep_elitism=0, keep_parents=0)
        ga_instance.run()
    assert not any('replacement' in str(warning.message) for warning in captured)
    numpy.testing.assert_array_equal(ga_instance.population, 7)


@pytest.mark.parametrize("setting", [0, numpy.int64(0), ['thread', 0], ('process', numpy.int64(0))])
def test_zero_workers_consistently_disable_parallel_processing(setting):
    assert make_ga(parallel_processing=setting).parallel_processing is None


@pytest.mark.parametrize("setting", [False, 0.0, ['thread', 0.0], ['process', False]])
def test_worker_counts_require_integers(setting):
    with pytest.raises(TypeError, match='parallel_processing'):
        make_ga(parallel_processing=setting)


def test_mutable_parameter_containers_are_owned_by_the_ga():
    space = [[0, 1], {'low': 2, 'high': 4}] + [None] * 6
    minimum, maximum = [-1] * 8, [1] * 8
    constraints = [None] * 8
    rates = [0.8, 0.2]
    ga_instance = make_ga(gene_space=space, gene_constraint=constraints, mutation_type='adaptive',
                          mutation_probability=rates, random_mutation_min_val=minimum, random_mutation_max_val=maximum)
    space[0][:] = [100]
    space[1]['low'] = 100
    minimum[:] = [100] * 8
    maximum[:] = [100] * 8
    constraints[0] = False
    rates[:] = [0, 0]
    assert ga_instance.gene_space[:2] == [[0, 1], {'low': 2, 'high': 4}]
    assert ga_instance.random_mutation_min_val == [-1] * 8
    assert ga_instance.random_mutation_max_val == [1] * 8
    assert ga_instance.gene_constraint == [None] * 8
    assert ga_instance.mutation_probability == [0.8, 0.2]


def test_rejected_constructors_do_not_sample_or_execute_constraints():
    numpy_state, python_state = numpy.random.get_state(), random.getstate()
    def constraint(solution, values):
        pytest.fail('Invalid parameters must be rejected before constraints execute.')
    with pytest.raises(TypeError, match='fitness_func'):
        make_ga(fitness_func=None, gene_constraint=[constraint] * 8)
    numpy.testing.assert_equal(numpy.random.get_state(), numpy_state)
    assert random.getstate() == python_state


def test_seeded_instances_and_global_generators_do_not_interfere():
    first = make_ga(random_seed=numpy.int64(7))
    second = make_ga(random_seed=7)
    first.run()
    unrelated = make_ga(random_seed=99)
    unrelated.run()
    numpy.random.seed(123)
    random.seed(123)
    second.run()
    numpy.testing.assert_array_equal(first.population, second.population)


def test_generator_states_survive_checkpoint_continuation(tmp_path):
    ga_instance = make_ga()
    ga_instance.run()
    filename = str(tmp_path / 'generator-state')
    ga_instance.save(filename)
    loaded = pygad.load(filename)
    ga_instance.run()
    loaded.run()
    numpy.testing.assert_array_equal(ga_instance.population, loaded.population)


@pytest.mark.parametrize("space", [range(10**12), {'low': 0, 'high': 10**12, 'step': 1}])
def test_large_finite_spaces_are_sampled_and_inspected_without_materialization(space, monkeypatch):
    original_arange = numpy.arange
    def checked_arange(*args, **kwargs):
        if args and max(args) > 100:
            pytest.fail('A large domain must not be allocated for ordinary sampling.')
        return original_arange(*args, **kwargs)
    monkeypatch.setattr(numpy, 'arange', checked_arange)
    ga_instance = make_ga(num_genes=3, gene_type=int, gene_space=space, mutation_probability=1.0)
    assert len(ga_instance.gene_space_unpacked) <= 100
    ga_instance.run()
    assert all(0 <= int(value) < 10**12 for value in ga_instance.population.flat)
    candidates = ga_instance.get_initial_population_gene_candidates(0, 5, all_integer_values=False)
    assert len(candidates) <= 5
    assert ga_instance.is_gene_value_in_space(0, 500_000_000_000, 1)


def test_large_integer_mutation_ranges_sample_small_candidate_arrays():
    ga_instance = make_ga(gene_type=int, random_mutation_min_val=-10**12, random_mutation_max_val=10**12,
                          mutation_probability=1.0)
    values = ga_instance.generate_gene_value_randomly(-10**12, 10**12, 1, 0, False, sample_size=5)
    assert len(values) == 5
    assert all(-10**12 + 1 <= int(value) < 10**12 + 1 for value in values)


def test_default_logger_does_not_remove_existing_handlers():
    logger = logging.getLogger('pygad.utils.validation')
    handler = logging.NullHandler()
    logger.addHandler(handler)
    try:
        make_ga()
        assert handler in logger.handlers
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("space", [range(10**12), {'low': 0, 'high': 10**12},
                                   {'low': 0, 'high': 10**12, 'step': 2}])
def test_large_nested_spaces_and_constraints_sample_bounded_candidates(space, monkeypatch):
    original_arange = numpy.arange
    def checked_arange(*args, **kwargs):
        if args and max(args) > 100:
            pytest.fail('Constraint sampling must not materialize a large domain.')
        return original_arange(*args, **kwargs)
    monkeypatch.setattr(numpy, 'arange', checked_arange)
    def constraint(solution, values):
        return [value for value in values if value >= 0]
    ga_instance = make_ga(num_genes=3, gene_type=int, gene_space=[space] * 3,
                          gene_constraint=[constraint] * 3, mutation_probability=1.0)
    ga_instance.run()
    assert all(0 <= int(value) < 10**12 for value in ga_instance.population.flat)


@pytest.mark.parametrize("lower,upper,step,expected", [(5, 1, 1, [1, 2, 3, 4]),
                                                       (5, 1, -1, [2, 3, 4, 5])])
def test_integer_range_helper_supports_reversed_bounds_and_descending_steps(lower, upper, step, expected):
    ga_instance = make_ga(gene_type=int)
    values = ga_instance.generate_gene_value_randomly(lower, upper, 0, 0, True,
                                                      sample_size=None, step=step)
    numpy.testing.assert_array_equal(values, expected)


def test_swap_explicit_count_can_change_multiple_pairs():
    ga_instance = make_ga(mutation_type='swap', mutation_num_genes=4)
    original = numpy.arange(8, dtype=float)[None, :]
    mutated = ga_instance.mutation(original.copy())
    assert numpy.count_nonzero(original != mutated) == 4
    numpy.testing.assert_array_equal(numpy.sort(mutated), original)


@pytest.mark.parametrize("dtype", [numpy.float16, numpy.float32, numpy.float64])
@pytest.mark.parametrize("precision", [None, 1])
def test_bounded_conversion_keeps_the_closest_value_below_excluded_upper_bound(dtype, precision):
    ga_instance = make_ga(gene_type=[dtype, precision], gene_space={'low': 0.0, 'high': 1.0})
    converted = ga_instance.convert_bounded_operator_gene_value(0, 1.0)
    assert isinstance(converted, dtype)
    assert float(converted) < 1.0
    expected = numpy.nextafter(dtype(1.0), dtype(-numpy.inf), dtype=dtype) if precision is None else dtype(0.9)
    assert converted == expected


def test_fixed_integer_dictionary_membership_matches_converted_candidates():
    ga_instance = make_ga(gene_type=int, gene_space={'low': 2, 'high': 2})
    assert ga_instance.is_gene_value_in_space(0, 2, 2)
    assert ga_instance.convert_bounded_operator_gene_value(0, 2.0) == 2


def test_falsey_callable_constraints_are_still_applied():
    class Constraint:
        def __bool__(self):
            return False
        def __call__(self, solution, values):
            return [value for value in values if value == 2]
    ga_instance = make_ga(gene_type=int, gene_space=[1, 2, 3], gene_constraint=[Constraint()] * 8,
                          mutation_probability=1.0)
    ga_instance.run()
    numpy.testing.assert_array_equal(ga_instance.population, 2)


def test_async_callable_objects_are_rejected_before_population_creation():
    class Fitness:
        async def __call__(self, ga_instance, solution, solution_index):
            return 1
    with pytest.raises(TypeError, match='fitness_func'):
        make_ga(fitness_func=Fitness())


@pytest.mark.parametrize("space", [{'low': 0.3, 'high': 1.0, 'step': 0.1},
                                   {'low': 1.0, 'high': 0.3, 'step': -0.1},
                                   {'low': 1.0, 'high': 2.0, 'step': 0.03}])
@pytest.mark.parametrize("gene_type", [float, int, [float, 2]])
def test_lazy_float_steps_match_numpy_arange_before_conversion(space, gene_type):
    ga_instance = make_ga(gene_type=gene_type, gene_space=space)
    expected = ga_instance.change_gene_dtype_and_round(0, numpy.arange(space['low'], space['high'], space['step']))
    numpy.testing.assert_array_equal(ga_instance.get_gene_space_values(0), numpy.unique(expected))
    assert all(ga_instance.is_gene_value_in_space(0, value, value) for value in expected)


def test_legacy_checkpoints_initialize_generators_and_normalize_counts(tmp_path):
    ga_instance = make_ga(num_generations=200, mutation_probability=0)
    ga_instance.num_generations = numpy.uint8(200)
    del ga_instance.numpy_random_generator
    del ga_instance.python_random_generator
    del ga_instance.mutation_control_explicitly_set
    filename = str(tmp_path / 'legacy-generator-state')
    ga_instance.save(filename)
    loaded = pygad.load(filename)
    assert type(loaded.num_generations) is int
    loaded.run()
    loaded.run()
    assert loaded.generations_completed == 400


@pytest.mark.parametrize("crossover_type", ['single_point', 'two_points', 'uniform', 'scattered', 'sbx'])
def test_zero_crossover_probability_preserves_parents_even_when_random_draw_is_zero(crossover_type, monkeypatch):
    ga_instance = make_ga(crossover_type=crossover_type, crossover_probability=0)
    ga_instance.numpy_random_generator = Mock(wraps=ga_instance.numpy_random_generator)
    monkeypatch.setattr(ga_instance.numpy_random_generator, 'random',
                        lambda size=None: 0.0 if size is None else numpy.zeros(size))
    parents = numpy.array([[0.0] * 8, [1.0] * 8])
    children = ga_instance.crossover(parents, (4, 8))
    numpy.testing.assert_array_equal(children, numpy.tile(parents, (2, 1)))


@pytest.mark.parametrize("dtype", [numpy.int64, numpy.uint64])
@pytest.mark.parametrize("operator", ['sbx', 'polynomial'])
@pytest.mark.parametrize("use_gene_space", [False, True])
def test_bounded_operators_do_not_overflow_at_large_integer_type_limits(dtype, operator, use_gene_space):
    maximum = int(numpy.iinfo(dtype).max)
    def fitness(ga_instance, solution, solution_index):
        return float(int(solution[0]) - maximum)
    ga_instance = make_ga(num_genes=3, gene_type=dtype, init_range_low=maximum - 10,
                          init_range_high=maximum + 1, keep_elitism=0, keep_parents=0,
                          fitness_func=fitness,
                          gene_space=range(maximum - 10, maximum + 1) if use_gene_space else None,
                          crossover_type='sbx' if operator == 'sbx' else None,
                          mutation_type='polynomial' if operator == 'polynomial' else None,
                          mutation_probability=1)
    ga_instance.run()
    assert ga_instance.population.dtype == numpy.dtype(dtype)
    assert all(maximum - 10 <= int(value) <= maximum for value in ga_instance.population.flat)
