"""Python int array storage is consistent across NumPy versions and platforms."""

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return float(sum(int(value) for value in solution))


def multi_objective_fitness_func(ga_instance, solution, solution_index):
    return [float(solution[0]), float(solution[1])]


def make_ga(**options):
    parameters = dict(num_generations=2, num_parents_mating=2,
                      fitness_func=fitness_func, sol_per_pop=4, num_genes=3,
                      gene_type=int, mutation_type=None, crossover_type=None,
                      random_seed=7, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


@pytest.mark.parametrize("gene_type", [int, [int, None]])
def test_python_integer_conversion_uses_int64_arrays_and_keeps_scalar_type(gene_type):
    ga_instance = make_ga(gene_type=gene_type)
    values = numpy.array([[-2**63, -(2**53 + 1), 2**53 + 1, 2**63 - 1]], dtype=object)
    converted = ga_instance.change_gene_dtype_and_round(0, values)
    assert converted.dtype == numpy.dtype(numpy.int64)
    numpy.testing.assert_array_equal(converted, values)
    assert not numpy.shares_memory(converted, values)
    for value in values.flat:
        scalar = ga_instance.change_gene_dtype_and_round(0, value)
        assert type(scalar) is int
        assert scalar == value
    assert ga_instance.change_gene_dtype_and_round(0, numpy.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("gene_space", [None, [None, None, None], [[None], [None], [None]],
                                        {'low': 2**53 + 1, 'high': 2**53 + 4},
                                        {'low': 2**53 + 1, 'high': 2**53 + 4, 'step': 1},
                                        range(2**53 + 1, 2**53 + 4),
                                        [2**53 + 1, 2**53 + 3], [None, 2**53 + 1]])
def test_generated_python_integer_populations_and_candidates_keep_large_values(gene_space):
    ga_instance = make_ga(gene_space=gene_space, init_range_low=2**53 + 1,
                          init_range_high=2**53 + 4)
    assert ga_instance.population.dtype == numpy.dtype(numpy.int64)
    assert all(2**53 + 1 <= int(value) < 2**53 + 4 for value in ga_instance.population.flat)
    candidates = ga_instance.get_initial_population_gene_candidates(0, 3)
    assert candidates.dtype == numpy.dtype(numpy.int64)
    assert all(2**53 + 1 <= int(value) < 2**53 + 4 for value in candidates)


@pytest.mark.parametrize("lower,upper", [(-2**63, -2**63 + 3), (2**63 - 3, 2**63),
                                        (2**63 - 1, 2**63 - 1)])
def test_python_integer_initialization_uses_signed_64_bit_bounds(lower, upper):
    ga_instance = make_ga(init_range_low=lower, init_range_high=upper)
    expected = [lower] if lower == upper else list(range(lower, upper))
    candidates = ga_instance.get_initial_population_gene_candidates(0, 3)
    numpy.testing.assert_array_equal(candidates, expected)
    assert all(int(value) in expected for value in ga_instance.population.flat)


@pytest.mark.parametrize("gene_type", [numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                                       numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64])
def test_explicit_numpy_integer_types_keep_their_storage_width(gene_type):
    ga_instance = make_ga(gene_type=gene_type, init_range_low=0, init_range_high=4)
    assert ga_instance.population.dtype == numpy.dtype(gene_type)
    assert ga_instance.initialize_parents_array((2, 3)).dtype == numpy.dtype(gene_type)
    assert ga_instance.change_gene_dtype_and_round(0, [1, 2]).dtype == numpy.dtype(gene_type)
    assert type(ga_instance.change_gene_dtype_and_round(0, 1)) is gene_type


@pytest.mark.parametrize("value", [-2**63 - 1, 2**63])
@pytest.mark.parametrize("gene_type", [int, [int, float, numpy.int32]])
def test_population_values_outside_python_integer_array_limits_raise(gene_type, value):
    # Supplied values are converted, not clipped to a different integer.
    with pytest.raises(OverflowError):
        make_ga(gene_type=gene_type, initial_population=[[value, 1, 2]] * 4)


def test_object_storage_keeps_integers_beyond_int64_without_losing_precision():
    values = [[2**80 + 1, -(2**80 + 1), 0]] * 4
    ga_instance = make_ga(gene_type=object, initial_population=values)
    ga_instance.run()
    numpy.testing.assert_array_equal(ga_instance.population, values)
    assert all(type(value) is int for value in ga_instance.population.flat)


@pytest.mark.parametrize("value,expected", [(2**53, 2**53 + 1), (2**53 + 2, 2**53 + 2),
                                            (2**63, 2**53 + 4)])
def test_bounded_operators_use_python_integer_storage_limits(value, expected):
    ga_instance = make_ga(init_range_low=2**53 + 1, init_range_high=2**53 + 4)
    converted = ga_instance.convert_bounded_operator_gene_value(0, value)
    assert type(converted) is int
    assert converted == expected


@pytest.mark.parametrize("mutation_type", ['random', 'adaptive'])
@pytest.mark.parametrize("use_probability", [False, True])
def test_random_mutation_keeps_large_python_integer_addition_exact(mutation_type, use_probability):
    value = 2**53 + 1
    parameters = dict(initial_population=[[value] * 3] * 4,
                      mutation_type=mutation_type, random_mutation_min_val=1,
                      random_mutation_max_val=2, keep_elitism=0, keep_parents=0)
    if use_probability:
        parameters['mutation_probability'] = [1.0, 1.0] if mutation_type == 'adaptive' else 1.0
    else:
        parameters['mutation_num_genes'] = [3, 3] if mutation_type == 'adaptive' else 3
    ga_instance = make_ga(**parameters)
    ga_instance.run()
    assert ga_instance.population.dtype == numpy.dtype(numpy.int64)
    numpy.testing.assert_array_equal(ga_instance.population, [[value + 2] * 3] * 4)


@pytest.mark.parametrize("parent_selection_type", ['sss', 'rws', 'sus', 'rank', 'random',
                                                   'tournament', 'nsga2', 'tournament_nsga2', 'nsga3'])
@pytest.mark.parametrize("crossover_type", ['single_point', 'two_points', 'uniform', 'scattered', 'sbx'])
def test_parent_selection_and_crossover_keep_python_integer_storage(parent_selection_type, crossover_type):
    allowed_values = [2**53 + 1, 2**53 + 3]
    population = [[allowed_values[0], allowed_values[1], allowed_values[0]],
                  [allowed_values[1], allowed_values[0], allowed_values[1]]] * 2
    ga_instance = make_ga(initial_population=population, gene_space=allowed_values,
                          fitness_func=multi_objective_fitness_func,
                          parent_selection_type=parent_selection_type,
                          crossover_type=crossover_type,
                          # SBX uses real arithmetic; zero probability checks
                          # its allocation without changing the parent values.
                          crossover_probability=0.0 if crossover_type == 'sbx' else 1.0,
                          nsga3_num_divisions=1, keep_elitism=0, keep_parents=0,
                          save_best_solutions=True)
    ga_instance.run()
    for values in [ga_instance.population, ga_instance.last_generation_parents,
                   ga_instance.last_generation_offspring_crossover, ga_instance.best_solutions]:
        assert values.dtype == numpy.dtype(numpy.int64)
        assert set(values.flat).issubset(allowed_values)


def test_nsga3_population_growth_keeps_large_python_integers():
    ga_instance = make_ga(initial_population=[[2**53 + 1] * 3] * 2,
                          init_range_low=2**53 + 1, init_range_high=2**53 + 4,
                          parent_selection_type='nsga3', fitness_func=multi_objective_fitness_func,
                          nsga3_num_divisions=4)
    ga_instance.run()
    assert ga_instance.population.shape[0] == 5
    assert ga_instance.population.dtype == numpy.dtype(numpy.int64)
    assert all(2**53 + 1 <= int(value) < 2**53 + 4 for value in ga_instance.population.flat)
