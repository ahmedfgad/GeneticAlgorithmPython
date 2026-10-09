"""Conversion, rounding, and gene-type preservation across the GA lifecycle."""

import copy

import numpy
import pytest

import pygad


def fitness_func(ga_instance, solution, solution_index):
    return float(sum(solution))


def make_ga(**options):
    parameters = dict(num_generations=2, num_parents_mating=2,
                      fitness_func=fitness_func, sol_per_pop=4, num_genes=3,
                      mutation_type=None, crossover_type=None,
                      random_seed=7, suppress_warnings=True)
    parameters.update(options)
    return pygad.GA(**parameters)


@pytest.mark.parametrize("dtype", [float, numpy.float16, numpy.float32, numpy.float64])
@pytest.mark.parametrize("precision", [None, 0, 1, 2, 4, -1])
def test_scalar_candidate_and_population_conversions_agree(dtype, precision):
    ga_instance = make_ga(gene_type=[dtype, precision])
    values = numpy.array([1.25, -1.75, 7.12345])
    expected = values if precision is None else numpy.round(values, precision)
    expected = numpy.asarray(expected, dtype=dtype)
    candidates = ga_instance.change_gene_dtype_and_round(0, values)
    population = ga_instance.change_population_dtype_and_round([values])[0]
    numpy.testing.assert_array_equal(candidates, expected)
    numpy.testing.assert_array_equal(population, expected)
    for value, converted in zip(values, expected):
        scalar = ga_instance.change_gene_dtype_and_round(0, value)
        assert type(scalar) is dtype
        assert scalar == converted


def test_rounding_uses_nearest_even_halfway_values():
    ga_instance = make_ga(gene_type=[float, 0])
    values = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
    numpy.testing.assert_array_equal(ga_instance.change_gene_dtype_and_round(0, values),
                                     [0, 2, 2, 0, -2, -2])


@pytest.mark.parametrize("dtype", [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                                    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64])
def test_integer_conversion_truncates_fractional_values(dtype):
    ga_instance = make_ga(gene_type=dtype)
    values = [0.9, 1.9, 2.1] if numpy.issubdtype(numpy.dtype(dtype), numpy.unsignedinteger) else [-1.9, -0.9, 2.9]
    expected = [int(value) for value in values]
    numpy.testing.assert_array_equal(ga_instance.change_gene_dtype_and_round(0, values), expected)
    assert type(ga_instance.change_gene_dtype_and_round(0, values[0])) is dtype


@pytest.mark.parametrize("values", [numpy.array(1.25), [1.25, 1.75],
                                     ((1.25, 1.75), (2.25, 2.75)), numpy.empty((0, 2))])
def test_candidate_conversion_preserves_input_shape_and_values(values):
    ga_instance = make_ga(gene_type=[numpy.float32, 1])
    original_values = copy.deepcopy(values)
    converted = ga_instance.change_gene_dtype_and_round(0, values)
    assert numpy.shape(converted) == numpy.shape(values)
    numpy.testing.assert_array_equal(values, original_values)
    if isinstance(converted, numpy.ndarray) and isinstance(values, numpy.ndarray):
        assert not numpy.shares_memory(values, converted)


@pytest.mark.parametrize("gene_type", [numpy.float32, [float, 2],
                                       [int, [numpy.float32, 2], numpy.int8]])
def test_population_conversion_returns_an_independent_array(gene_type):
    ga_instance = make_ga(gene_type=gene_type)
    population = numpy.array([[1, 2.236, 3], [4, 5.678, 6]], dtype=object)
    original_population = population.copy()
    converted = ga_instance.change_population_dtype_and_round(population)
    converted[0, 0] = 10
    numpy.testing.assert_array_equal(population, original_population)
    assert not numpy.shares_memory(population, converted)


def test_grouped_conversion_preserves_declared_scalar_types_and_large_integers():
    gene_type = [int, [numpy.float32, 2], int, [numpy.float32, 2], numpy.int8, float]
    ga_instance = make_ga(num_genes=6, gene_type=gene_type)
    values = ((2**53 + 1, 1.236, 2**53 + 3, 2.341, 7, 1.5),)
    converted = ga_instance.change_population_dtype_and_round(values)
    assert converted[0, 0] == 2**53 + 1 and type(converted[0, 0]) is int
    assert converted[0, 2] == 2**53 + 3 and type(converted[0, 2]) is int
    assert converted[0, 1] == numpy.float32(1.24) and type(converted[0, 1]) is numpy.float32
    assert converted[0, 3] == numpy.float32(2.34) and type(converted[0, 3]) is numpy.float32
    assert type(converted[0, 4]) is numpy.int8 and type(converted[0, 5]) is float


def test_round_genes_rounds_before_narrow_float_casting():
    ga_instance = make_ga(gene_type=[numpy.float16, 4])
    values = numpy.array([[7.12345, 1.25, 1.75]])
    expected = numpy.asarray(numpy.round(values, 4), dtype=numpy.float16)
    rounded = ga_instance.round_genes(values)
    numpy.testing.assert_array_equal(rounded, expected)
    assert numpy.all(numpy.isfinite(rounded))


def test_round_genes_preserves_mixed_array_identity_and_destination_types():
    ga_instance = make_ga(gene_type=[int, [numpy.float32, 2], float])
    values = numpy.array([[1.9, 2.236, 3.5]], dtype=object)
    rounded = ga_instance.round_genes(values)
    assert rounded is values
    assert type(rounded[0, 0]) is int and rounded[0, 0] == 1
    assert type(rounded[0, 1]) is numpy.float32 and rounded[0, 1] == numpy.float32(2.24)


def test_additive_mutation_rounds_after_adding_in_working_precision():
    ga_instance = make_ga(gene_type=[numpy.float16, 2])
    converted = ga_instance.mutation_change_gene_dtype_and_round(
        numpy.float16(0.00501), 0, numpy.float16(1), False)
    assert converted == numpy.float16(1.01)


def test_additive_mutation_keeps_mixed_signed_and_unsigned_integer_addition_exact():
    ga_instance = make_ga(gene_type=numpy.uint64)
    value = 2**63 + 1
    converted = ga_instance.mutation_change_gene_dtype_and_round(
        numpy.int64(2), 0, numpy.uint64(value), False)
    assert int(converted) == value + 2
    candidates = ga_instance.mutation_change_gene_dtype_and_round(
        numpy.array([1, 2], dtype=numpy.int64), 0, numpy.uint64(value), False)
    assert [int(candidate) for candidate in candidates] == [value + 1, value + 2]


def test_fractional_integer_offsets_are_added_before_truncation():
    ga_instance = make_ga(gene_type=int)
    value = ga_instance.generate_gene_value_randomly(0.2, 0.3, -1, 0, False)
    assert value == 0


@pytest.mark.parametrize("gene_type", [[int, float], (int, float),
                                       numpy.array([int, float]), [[float, numpy.int32(2)], int],
                                       numpy.array([[float, 2], [int, None]], dtype=object)])
def test_validation_normalizes_specifications_without_modifying_them(gene_type):
    original = copy.deepcopy(gene_type)
    ga_instance = make_ga(num_genes=2, gene_type=gene_type)
    assert not ga_instance.gene_type_single
    numpy.testing.assert_array_equal(numpy.asarray(gene_type, dtype=object), numpy.asarray(original, dtype=object))


@pytest.mark.parametrize("specification,error", [([float, True], TypeError),
                                                  ([float, 1.5], TypeError), ([float, '2'], TypeError),
                                                  ([int, 1], ValueError), ([int, -1], ValueError),
                                                  (numpy.array(float), ValueError),
                                                  ([numpy.array(float), int, float], ValueError),
                                                  ([[float, 1, 2], int, float], ValueError),
                                                  (numpy.dtype('float32'), TypeError),
                                                  (bool, TypeError), (complex, TypeError)])
def test_invalid_type_and_precision_specifications_fail_descriptively(specification, error):
    with pytest.raises(error, match="gene_type|precision"):
        make_ga(gene_type=specification)


def custom_crossover(parents, offspring_size, ga_instance):
    return numpy.tile(numpy.array([7.9, 1.236, 2.341]), (offspring_size[0], 1))


def custom_mutation(offspring, ga_instance):
    return numpy.tile(numpy.array([8.9, 2.341, 3.456]), (offspring.shape[0], 1))


@pytest.mark.parametrize("allow_duplicate_genes", [True, False])
def test_custom_operators_are_converted_before_callbacks_and_population_use(allow_duplicate_genes):
    events = []
    def check_types(ga_instance, offspring):
        events.append(offspring.copy())
        assert type(offspring[0, 0]) is int
        assert type(offspring[0, 1]) is numpy.float32
        assert type(offspring[0, 2]) is float
        assert offspring[0, 1] in [numpy.float32(1.24), numpy.float32(2.34)]
    ga_instance = make_ga(gene_type=[int, [numpy.float32, 2], [float, 2]],
                          crossover_type=custom_crossover, mutation_type=custom_mutation,
                          on_crossover=check_types, on_mutation=check_types,
                          keep_elitism=0, keep_parents=0, allow_duplicate_genes=allow_duplicate_genes)
    ga_instance.run()
    assert len(events) == 4
    numpy.testing.assert_array_equal(ga_instance.population[0], [8, numpy.float32(2.34), 3.46])


@pytest.mark.parametrize("callback_name", ['on_crossover', 'on_mutation', 'on_parents'])
@pytest.mark.parametrize("return_values", [True, False])
def test_callback_outputs_keep_mixed_types_and_large_integer_values(callback_name, return_values):
    value = 2**53 + 1
    def callback(ga_instance, population):
        replacement = [[value, 1.236, 4] for _ in population]
        if return_values:
            return (replacement, ga_instance.last_generation_parents_indices) if callback_name == 'on_parents' else replacement
        population[:] = numpy.asarray(replacement, dtype=object)
    ga_instance = make_ga(gene_type=[int, [numpy.float32, 2], numpy.int8],
                          keep_elitism=0, keep_parents=0, **{callback_name: callback})
    ga_instance.run()
    population = ga_instance.last_generation_parents if callback_name == 'on_parents' else ga_instance.population
    assert population[0, 0] == value and type(population[0, 0]) is int
    assert population[0, 1] == numpy.float32(1.24) and type(population[0, 1]) is numpy.float32
    assert type(population[0, 2]) is numpy.int8


@pytest.mark.parametrize("mutation_type", ['random', 'adaptive', 'swap', 'inversion', 'scramble', 'polynomial'])
def test_builtin_mutations_preserve_mixed_destination_types(mutation_type):
    parameters = dict(gene_type=[int, [numpy.float32, 2], numpy.int8], mutation_type=mutation_type,
                      crossover_type='uniform', keep_elitism=0, keep_parents=0)
    parameters['mutation_probability'] = [1.0, 1.0] if mutation_type == 'adaptive' else 1.0
    ga_instance = make_ga(**parameters)
    ga_instance.run()
    for solution in ga_instance.population:
        assert type(solution[0]) is int
        assert type(solution[1]) is numpy.float32
        assert type(solution[2]) is numpy.int8


def test_saved_best_solutions_preserve_mixed_types_across_repeated_runs():
    value = 2**53 + 1
    ga_instance = make_ga(gene_type=[int, [numpy.float32, 2], numpy.int8],
                          initial_population=[[value, 1.236, 4], [value, 1.236, 4]],
                          save_best_solutions=True)
    ga_instance.run()
    ga_instance.run()
    assert ga_instance.best_solutions.dtype == object
    for solution in ga_instance.best_solutions:
        assert solution[0] == value and type(solution[0]) is int
        assert type(solution[1]) is numpy.float32 and solution[1] == numpy.float32(1.24)
        assert type(solution[2]) is numpy.int8


@pytest.mark.parametrize("gene_type", [object, [object, 2], [int, object, float]])
def test_object_storage_keeps_numeric_values_without_integer_range_handling(gene_type):
    ga_instance = make_ga(gene_type=gene_type, mutation_type='random', mutation_probability=1.0)
    ga_instance.run()
    assert ga_instance.population.dtype == object
    assert all(isinstance(value, (int, float, numpy.number)) for value in ga_instance.population.flat)


@pytest.mark.parametrize("precision", [2, 309, 400, -309, 2**40, -(2**40)])
def test_extreme_rounding_preserves_finite_values(precision):
    values = [1.234, -1.234, 1e308, -1e308, 0.0]
    ga_instance = make_ga(gene_type=[float, precision], initial_population=[values, values])
    expected = [round(value, precision) for value in values]
    numpy.testing.assert_array_equal(ga_instance.population[0], expected)
    for value, rounded_value in zip(values, expected):
        assert ga_instance.change_gene_dtype_and_round(0, value) == rounded_value


@pytest.mark.parametrize("precision", [309, 400, -309, 2**40, -(2**40)])
def test_generated_population_handles_extreme_precision(precision):
    ga_instance = make_ga(gene_type=[float, precision])
    assert numpy.all(numpy.isfinite(ga_instance.population))
    assert numpy.all(ga_instance.population >= -4)
    assert numpy.all(ga_instance.population < 4)
    if precision < 0:
        numpy.testing.assert_array_equal(ga_instance.population, 0)


@pytest.mark.parametrize("gene_type", [int, [int, numpy.float32, numpy.int64]])
def test_finite_gene_spaces_convert_large_integers_before_float_inference(gene_type):
    value = 2**53 + 1
    ga_instance = make_ga(gene_type=gene_type, gene_space=[value, 1.5])
    assert value in ga_instance.get_gene_space_values(0)
    unpacked_space = ga_instance.gene_space_unpacked if ga_instance.gene_type_single else ga_instance.gene_space_unpacked[0]
    assert value in unpacked_space


@pytest.mark.parametrize("lower,upper,expected", [(2**53 + 1, 2**53 + 3, [2**53 + 1, 2**53 + 2]),
                                                 (numpy.int64(2**53 + 1), numpy.int64(2**53 + 3), [2**53 + 1, 2**53 + 2]),
                                                 (numpy.float32(1.2), numpy.float32(2.0), [1]),
                                                 (-1.9, -1.0, [-1]), (-1.9, -0.2, [-1, 0]),
                                                 (1.2, 2.0, [1])])
def test_integer_continuous_space_bounds_remain_exact(lower, upper, expected):
    ga_instance = make_ga(gene_type=int, gene_space={'low': lower, 'high': upper},
                          initial_population=[[expected[0]] * 3] * 2)
    numpy.testing.assert_array_equal(ga_instance.get_gene_space_values(0), expected)


@pytest.mark.parametrize("dtype", [numpy.int64, numpy.uint64])
def test_generated_integer_ranges_preserve_exact_numpy_bounds(dtype):
    lower = 2**53 + 1 if dtype is numpy.int64 else 2**63 + 1
    ga_instance = make_ga(gene_type=dtype, init_range_low=dtype(lower), init_range_high=dtype(lower + 3))
    assert all(lower <= int(value) < lower + 3 for value in ga_instance.population.flat)
    numpy.testing.assert_array_equal(ga_instance._initial_population_integer_bounds(0, dtype(lower), dtype(lower + 3)),
                                     [lower, lower + 2])


@pytest.mark.parametrize("dtype,precision", [(numpy.float32, 2), (float, 400), (float, 2**40)])
def test_continuous_space_fallback_respects_stored_value_bounds(dtype, precision, monkeypatch):
    ga_instance = make_ga(gene_type=[dtype, precision], gene_space={'low': 0.9, 'high': 1.0})
    monkeypatch.setattr(numpy.random, 'uniform', lambda *args, **kwargs: numpy.ones(kwargs['size']))
    values = ga_instance.get_gene_space_values(0, sample_size=1)
    assert len(values) == 1
    assert 0.9 <= float(values[0]) < 1.0


def test_custom_parent_selection_applies_types_before_its_callback():
    def selection_func(fitness, num_parents, ga_instance):
        return (numpy.tile([7.9, 1.236, 4.9], (num_parents, 1)), numpy.arange(num_parents))
    def on_parents(ga_instance, parents):
        assert type(parents[0, 0]) is int and parents[0, 0] == 7
        assert type(parents[0, 1]) is numpy.float32 and parents[0, 1] == numpy.float32(1.24)
        assert type(parents[0, 2]) is numpy.int8 and parents[0, 2] == 4
    ga_instance = make_ga(parent_selection_type=selection_func, on_parents=on_parents,
                          gene_type=[int, [numpy.float32, 2], numpy.int8],
                          keep_elitism=0, keep_parents=0)
    ga_instance.run()


def test_custom_operators_apply_types_without_callbacks():
    ga_instance = make_ga(gene_type=[int, [numpy.float32, 2], [float, 2]],
                          crossover_type=custom_crossover, mutation_type=custom_mutation,
                          keep_elitism=0, keep_parents=0)
    ga_instance.run()
    numpy.testing.assert_array_equal(ga_instance.last_generation_offspring_crossover[0],
                                     [7, numpy.float32(1.24), 2.34])
    numpy.testing.assert_array_equal(ga_instance.population[0], [8, numpy.float32(2.34), 3.46])


@pytest.mark.parametrize("mutation_type", ['random', 'adaptive'])
@pytest.mark.parametrize("use_probability", [True, False])
def test_mutation_from_finite_spaces_preserves_declared_scalar_types(mutation_type, use_probability):
    options = dict(gene_type=[int, [numpy.float32, 2], numpy.int8], gene_space=[1.236, 2.345, 3.456],
                   mutation_type=mutation_type, keep_elitism=0, keep_parents=0)
    if use_probability:
        options['mutation_probability'] = [1.0, 1.0] if mutation_type == 'adaptive' else 1.0
    else:
        options['mutation_num_genes'] = [3, 3] if mutation_type == 'adaptive' else 3
    ga_instance = make_ga(**options)
    ga_instance.run()
    for solution in ga_instance.population:
        assert type(solution[0]) is int
        assert type(solution[1]) is numpy.float32 and solution[1] in [numpy.float32(1.24), numpy.float32(2.35), numpy.float32(3.46)]
        assert type(solution[2]) is numpy.int8
