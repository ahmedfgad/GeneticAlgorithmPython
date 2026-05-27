import math

import numpy
import pytest

import pygad
from pygad.benchmarks import classic, zdt, dtlz, knapsack


# ── Classic single-objective problems ────────────────────────────────────────


def test_sphere_global_minimum_at_origin():
    # The Sphere function has its global minimum at the origin where
    # f(x) = 0. Under PyGAD's max convention the fitness is -0 = 0.
    problem = classic.Sphere(num_genes=5)
    fitness = problem(None, numpy.zeros(5), 0)
    assert fitness == pytest.approx(0.0, abs=1e-12)


def test_rastrigin_global_minimum_at_origin():
    problem = classic.Rastrigin(num_genes=4)
    fitness = problem(None, numpy.zeros(4), 0)
    assert fitness == pytest.approx(0.0, abs=1e-12)


def test_rosenbrock_global_minimum_at_ones():
    problem = classic.Rosenbrock(num_genes=4)
    fitness = problem(None, numpy.ones(4), 0)
    assert fitness == pytest.approx(0.0, abs=1e-12)


def test_griewank_global_minimum_at_origin():
    problem = classic.Griewank(num_genes=3)
    fitness = problem(None, numpy.zeros(3), 0)
    assert fitness == pytest.approx(0.0, abs=1e-12)


def test_schwefel_global_minimum_at_420():
    problem = classic.Schwefel(num_genes=3)
    fitness = problem(None, numpy.full(3, 420.9687), 0)
    # The Schwefel constant is an approximation, so the fitness is
    # near zero but not exactly zero.
    assert fitness == pytest.approx(0.0, abs=1e-3)


def test_ackley_global_minimum_at_origin():
    problem = classic.Ackley(num_genes=5)
    fitness = problem(None, numpy.zeros(5), 0)
    assert fitness == pytest.approx(0.0, abs=1e-9)


def test_himmelblau_known_global_minima_all_evaluate_to_zero():
    problem = classic.Himmelblau()
    minima = numpy.array([
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ])
    for x in minima:
        assert problem(None, x, 0) == pytest.approx(0.0, abs=1e-4)


def test_sphere_integrates_with_pygad_ga_loop():
    # End-to-end check: run a few generations to confirm the problem
    # class plugs into PyGAD without configuration tweaks.
    problem = classic.Sphere(num_genes=5)
    ga = pygad.GA(num_generations=10,
                  num_parents_mating=6,
                  fitness_func=problem,
                  sol_per_pop=10,
                  num_genes=problem.num_genes,
                  init_range_low=problem.bounds[0],
                  init_range_high=problem.bounds[1],
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    # Best fitness must be at most 0 (it is the negated sum of squares).
    best_fitness = ga.best_solution(ga.last_generation_fitness)[1]
    assert best_fitness <= 0.0


# ── ZDT family ───────────────────────────────────────────────────────────────


def test_zdt1_returns_two_objectives_and_pareto_front_is_convex():
    problem = zdt.ZDT1(num_genes=10)
    output = problem(None, numpy.zeros(10), 0)
    assert len(output) == 2
    # On the Pareto front (x_1..x_n = 0) f1 = x_0 and f2 = 1 - sqrt(f1).
    optimal_solution = numpy.array([0.5] + [0.0] * 9)
    f1_max, f2_max = problem(None, optimal_solution, 0)
    assert f1_max == pytest.approx(-0.5)
    assert f2_max == pytest.approx(-(1.0 - math.sqrt(0.5)), abs=1e-9)


def test_zdt1_pareto_front_values_satisfy_curve():
    front = zdt.ZDT1().pareto_front(num_points=11)
    # Negate back to the original min convention so the relation
    # f2 = 1 - sqrt(f1) is easy to check.
    f1 = -front[:, 0]
    f2 = -front[:, 1]
    numpy.testing.assert_allclose(f2, 1.0 - numpy.sqrt(f1), atol=1e-9)


def test_zdt2_pareto_front_satisfies_curve():
    front = zdt.ZDT2().pareto_front(num_points=11)
    f1 = -front[:, 0]
    f2 = -front[:, 1]
    numpy.testing.assert_allclose(f2, 1.0 - f1 ** 2, atol=1e-9)


def test_zdt3_returns_two_objectives():
    problem = zdt.ZDT3(num_genes=10)
    output = problem(None, numpy.zeros(10), 0)
    assert len(output) == 2


def test_zdt4_returns_two_objectives_and_uses_wider_bounds():
    problem = zdt.ZDT4(num_genes=10)
    assert problem.bounds == (-5.0, 5.0)
    output = problem(None, numpy.zeros(10), 0)
    assert len(output) == 2


def test_zdt6_returns_two_objectives_with_correct_optimal_value():
    problem = zdt.ZDT6(num_genes=10)
    # When all rest of the variables are 0 the front has g = 1, so
    # f1 = 1 - exp(-4 * x0) * sin(6 pi x0)^6 and f2 = 1 - f1^2.
    solution = numpy.array([0.1] + [0.0] * 9)
    f1_max, f2_max = problem(None, solution, 0)
    expected_f1 = 1.0 - math.exp(-0.4) * math.sin(0.6 * math.pi) ** 6
    expected_f2 = 1.0 - expected_f1 ** 2
    assert f1_max == pytest.approx(-expected_f1, abs=1e-9)
    assert f2_max == pytest.approx(-expected_f2, abs=1e-9)


# ── DTLZ family ──────────────────────────────────────────────────────────────


def test_dtlz1_returns_three_objectives_on_pareto_optimal_solution():
    problem = dtlz.DTLZ1(num_objectives=3, num_distance_vars=5)
    # Distance variables at 0.5 give g = 0, so each f_i = 0.5 *
    # product of position variables (the simplex sum = 0.5).
    solution = numpy.array([0.5, 0.5] + [0.5] * 5)
    objectives = problem(None, solution, 0)
    assert len(objectives) == 3
    assert sum(-f for f in objectives) == pytest.approx(0.5, abs=1e-12)


def test_dtlz2_returns_three_objectives_on_unit_sphere_pareto_solution():
    problem = dtlz.DTLZ2(num_objectives=3, num_distance_vars=10)
    # Distance variables at 0.5 give g = 0, so the solution lives on
    # the unit sphere in the first orthant.
    solution = numpy.array([0.3, 0.7] + [0.5] * 10)
    objectives = problem(None, solution, 0)
    radius_squared = sum(f ** 2 for f in objectives)
    assert radius_squared == pytest.approx(1.0, abs=1e-9)


def test_dtlz3_uses_hard_g_function():
    # When distance variables are at 0.5 the g-function evaluates to
    # 0 even with DTLZ3's hard multimodal definition, so the front
    # also lives on the unit sphere.
    problem = dtlz.DTLZ3(num_objectives=3, num_distance_vars=10)
    solution = numpy.array([0.4, 0.6] + [0.5] * 10)
    objectives = problem(None, solution, 0)
    radius_squared = sum(f ** 2 for f in objectives)
    assert radius_squared == pytest.approx(1.0, abs=1e-9)


def test_dtlz4_alpha_biases_objectives_toward_one_corner():
    # With alpha large, the position variables get squashed toward 0
    # unless x is very close to 1. So with x_position = 0.5 most
    # objective weight ends up on the first objective.
    problem = dtlz.DTLZ4(num_objectives=3, num_distance_vars=10, alpha=100.0)
    solution = numpy.array([0.5, 0.5] + [0.5] * 10)
    objectives = problem(None, solution, 0)
    # The first objective dominates because cos(very small angle) ≈ 1.
    abs_objectives = [abs(f) for f in objectives]
    assert abs_objectives[0] == max(abs_objectives)


def test_dtlz2_rejects_num_objectives_below_two():
    with pytest.raises(ValueError, match="num_objectives"):
        dtlz.DTLZ2(num_objectives=1)


def test_benchmarks_module_is_importable_from_top_level():
    # Sanity check that the user-facing import path works.
    assert pygad.benchmarks.classic.Sphere(num_genes=2).num_genes == 2
    assert pygad.benchmarks.zdt.ZDT1().num_objectives == 2
    assert pygad.benchmarks.dtlz.DTLZ2().num_objectives == 3
    assert pygad.benchmarks.knapsack.Knapsack(
        weights=[1.0], values=[1.0], capacity=1.0).num_genes == 1


# ── Knapsack ─────────────────────────────────────────────────────────────────


def test_knapsack_returns_sum_of_chosen_values_when_feasible():
    problem = knapsack.Knapsack(weights=[2.0, 3.0, 4.0],
                                values=[10.0, 20.0, 30.0],
                                capacity=5.0)
    # Picking items 0 and 1: weight = 5 (at capacity), value = 30.
    fitness = problem(None, numpy.array([1, 1, 0]), 0)
    assert fitness == pytest.approx(30.0)


def test_knapsack_at_exact_capacity_is_considered_feasible():
    problem = knapsack.Knapsack(weights=[5.0],
                                values=[7.0],
                                capacity=5.0)
    assert problem(None, numpy.array([1]), 0) == pytest.approx(7.0)


def test_knapsack_overweight_solution_gets_negative_fitness():
    problem = knapsack.Knapsack(weights=[2.0, 3.0, 4.0],
                                values=[10.0, 20.0, 30.0],
                                capacity=5.0)
    # Picking all three items: weight = 9, over capacity by 4.
    fitness = problem(None, numpy.array([1, 1, 1]), 0)
    assert fitness == pytest.approx(-4.0)


def test_knapsack_empty_selection_has_zero_fitness():
    problem = knapsack.Knapsack(weights=[2.0, 3.0, 4.0],
                                values=[10.0, 20.0, 30.0],
                                capacity=5.0)
    assert problem(None, numpy.array([0, 0, 0]), 0) == pytest.approx(0.0)


def test_knapsack_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        knapsack.Knapsack(weights=[1.0, 2.0], values=[1.0], capacity=1.0)


def test_knapsack_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity must be positive"):
        knapsack.Knapsack(weights=[1.0], values=[1.0], capacity=0.0)


def test_knapsack_rejects_negative_weights():
    with pytest.raises(ValueError, match="weights must be non-negative"):
        knapsack.Knapsack(weights=[-1.0], values=[1.0], capacity=1.0)


def test_knapsack_finds_optimal_solution_on_small_instance_end_to_end():
    # Classic small knapsack instance with a known optimal subset.
    # Items (weight, value): (2,3), (3,4), (4,5), (5,6).
    # Capacity = 5. Optimal subset = items 0 and 1 (weight 5, value 7).
    problem = knapsack.Knapsack(weights=[2.0, 3.0, 4.0, 5.0],
                                values=[3.0, 4.0, 5.0, 6.0],
                                capacity=5.0)
    ga = pygad.GA(num_generations=50,
                  num_parents_mating=10,
                  fitness_func=problem,
                  sol_per_pop=30,
                  num_genes=problem.num_genes,
                  gene_space=problem.gene_space,
                  gene_type=problem.gene_type,
                  random_seed=0,
                  suppress_warnings=True)
    ga.run()
    best_solution, best_fitness, _ = ga.best_solution(ga.last_generation_fitness)
    assert best_fitness == pytest.approx(7.0)
    numpy.testing.assert_array_equal(best_solution, [1, 1, 0, 0])
