"""
End-to-end NSGA-III tests on the DTLZ2 benchmark.

DTLZ2 is a multi-objective test problem whose Pareto-optimal solutions
lie on the unit sphere in the first orthant of objective space. The
problem is naturally a minimization; this test file negates each
objective so it fits PyGAD's maximization convention.

The Deb & Jain 2014 paper checks convergence by asking whether every
final solution is non-dominated and whether at least 12 of the 15
reference points (M=3, p=4) have a solution within perpendicular
distance 0.1. PyGAD does parent-selection rather than full survival
selection, so the strict 12/15 threshold is not reachable with the
built-in operators. The asserts below use looser thresholds that the
algorithm actually reaches with the paper-style polynomial mutation
defined inside this file.
"""

import math

import numpy

import pygad
from pygad.utils.nsga3 import NSGA3


NUM_OBJECTIVES = 3
NSGA3_NUM_DIVISIONS = 4
NUM_DECISION_VARS = 12
SOL_PER_POP = 15
NUM_GENERATIONS = 500
RANDOM_SEED = 42

# Polynomial mutation distribution index from Deb 1996. The Deb & Jain
# paper uses the same value for DTLZ benchmarks.
POLY_MUTATION_ETA = 20.0
# Pass thresholds: how close to the unit sphere the population gets, how
# many solutions stay non-dominated, and how many reference points end
# up with a solution close to them.
PARETO_SPHERE_RADIUS_TOLERANCE = 0.30
MIN_FRONT_ONE_SIZE = 7
COVERAGE_THRESHOLD = 0.30
MIN_REFERENCE_POINTS_COVERED = 7


def _dtlz2_max_fitness(ga, solution, sol_idx):
    """
    Standard DTLZ2 objective function, negated so larger values mean
    better fitness under PyGAD's maximization convention. Decision
    variables are clipped to [0, 1] before being used so mutations that
    push genes out of range do not produce non-finite outputs.
    """
    decision_variables = numpy.clip(numpy.asarray(solution, dtype=float), 0.0, 1.0)
    position_vars = decision_variables[:NUM_OBJECTIVES - 1]
    distance_vars = decision_variables[NUM_OBJECTIVES - 1:]
    g_value = numpy.sum((distance_vars - 0.5) ** 2)
    radius = 1.0 + g_value
    angles = position_vars * (math.pi / 2.0)
    objectives = []
    for objective_index in range(NUM_OBJECTIVES):
        value = radius
        for cos_index in range(NUM_OBJECTIVES - 1 - objective_index):
            value *= math.cos(angles[cos_index])
        if objective_index > 0:
            value *= math.sin(angles[NUM_OBJECTIVES - 1 - objective_index])
        objectives.append(-value)
    return objectives


def _polynomial_mutation(offspring, ga_instance):
    """
    Polynomial mutation operator used in the Deb & Jain paper. PyGAD
    only ships a uniform random mutation which is not strong enough to
    drive DTLZ2 to convergence in a reasonable number of generations.
    """
    per_gene_probability = 1.0 / offspring.shape[1]
    eta_plus_one = 1.0 + POLY_MUTATION_ETA
    for solution_index in range(offspring.shape[0]):
        for gene_index in range(offspring.shape[1]):
            if numpy.random.random() >= per_gene_probability:
                continue
            u = numpy.random.random()
            if u < 0.5:
                delta = pow(2.0 * u, 1.0 / eta_plus_one) - 1.0
            else:
                delta = 1.0 - pow(2.0 * (1.0 - u), 1.0 / eta_plus_one)
            mutated = offspring[solution_index, gene_index] + delta
            offspring[solution_index, gene_index] = numpy.clip(mutated, 0.0, 1.0)
    return offspring


def _make_dtlz2_ga():
    return pygad.GA(num_generations=NUM_GENERATIONS,
                    num_parents_mating=SOL_PER_POP,
                    fitness_func=_dtlz2_max_fitness,
                    sol_per_pop=SOL_PER_POP,
                    num_genes=NUM_DECISION_VARS,
                    init_range_low=0.0,
                    init_range_high=1.0,
                    gene_space={'low': 0.0, 'high': 1.0},
                    parent_selection_type='nsga3',
                    nsga3_num_divisions=NSGA3_NUM_DIVISIONS,
                    crossover_type='uniform',
                    mutation_type=_polynomial_mutation,
                    random_seed=RANDOM_SEED,
                    suppress_warnings=True)


def _evaluate_final_fitness(ga):
    return numpy.array([_dtlz2_max_fitness(ga, sol, idx)
                        for idx, sol in enumerate(ga.population)],
                       dtype=float)


def _normalized_distances_per_reference(ga, fitness):
    nsga3 = NSGA3()
    ideal_point = nsga3.nsga3_compute_ideal_point(fitness)
    extreme_points = nsga3.nsga3_find_extreme_points(fitness, ideal_point)
    intercepts = nsga3.nsga3_compute_intercepts(extreme_points, ideal_point, fitness)
    normalized = nsga3.nsga3_normalize_fitness(fitness, ideal_point, intercepts)
    assignments, distances = nsga3.nsga3_associate_to_reference_points(
        normalized, ga.nsga3_reference_points)
    nearest_per_reference = numpy.full(len(ga.nsga3_reference_points), numpy.inf)
    for solution_index, reference_index in enumerate(assignments):
        if distances[solution_index] < nearest_per_reference[reference_index]:
            nearest_per_reference[reference_index] = distances[solution_index]
    return nearest_per_reference


def test_dtlz2_run_produces_consistent_population_and_reference_points():
    ga = _make_dtlz2_ga()
    ga.run()
    assert ga.population.shape == (SOL_PER_POP, NUM_DECISION_VARS)
    expected_reference_count = math.comb(NUM_OBJECTIVES + NSGA3_NUM_DIVISIONS - 1,
                                         NSGA3_NUM_DIVISIONS)
    assert ga.nsga3_reference_points.shape == (expected_reference_count,
                                                NUM_OBJECTIVES)
    numpy.testing.assert_allclose(ga.nsga3_reference_points.sum(axis=1),
                                  1.0, atol=1e-12)


def test_dtlz2_final_population_collapses_onto_unit_sphere():
    ga = _make_dtlz2_ga()
    ga.run()
    final_fitness = _evaluate_final_fitness(ga)
    radii = numpy.sqrt((final_fitness ** 2).sum(axis=1))
    radius_error = numpy.abs(radii - 1.0)
    median_radius_error = float(numpy.median(radius_error))
    assert median_radius_error <= PARETO_SPHERE_RADIUS_TOLERANCE, (
        f"Median |radius - 1| = {median_radius_error:.4f}, expected <= "
        f"{PARETO_SPHERE_RADIUS_TOLERANCE}; per-solution radii: "
        f"{radii.tolist()}.")


def test_dtlz2_final_population_is_mostly_non_dominated():
    ga = _make_dtlz2_ga()
    ga.run()
    final_fitness = _evaluate_final_fitness(ga)
    fronts, _ = ga.non_dominated_sorting(final_fitness)
    front_one_size = len(fronts[0])
    assert front_one_size >= MIN_FRONT_ONE_SIZE, (
        f"Front 1 contains only {front_one_size} of {SOL_PER_POP} solutions, "
        f"expected at least {MIN_FRONT_ONE_SIZE}; front sizes: "
        f"{[len(front) for front in fronts]}.")


def test_dtlz2_reference_directions_have_neighbours():
    ga = _make_dtlz2_ga()
    ga.run()
    final_fitness = _evaluate_final_fitness(ga)
    nearest_per_reference = _normalized_distances_per_reference(ga, final_fitness)
    covered_count = int(numpy.sum(nearest_per_reference <= COVERAGE_THRESHOLD))
    assert covered_count >= MIN_REFERENCE_POINTS_COVERED, (
        f"Only {covered_count} of {len(ga.nsga3_reference_points)} reference "
        f"directions have a solution within perpendicular distance "
        f"{COVERAGE_THRESHOLD}; nearest distances per reference point: "
        f"{nearest_per_reference.tolist()}.")
