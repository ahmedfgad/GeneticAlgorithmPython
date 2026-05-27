import math
import warnings

import numpy
import pytest

import pygad
from pygad.utils.nsga3 import NSGA3


@pytest.fixture
def nsga3():
    return NSGA3()


# Six solutions in PyGAD maximisation form. The same numbers under the
# usual minimisation convention would be (1, 6), (2, 4.5), (3, 3), (4.5, 2),
# (6, 1), (4, 4). Solutions s1 and s5 are the two axis extremes; s6 is a
# dominated interior point. The expected NSGA-III values below were
# derived by hand from this fitness pool.
GUIDE_FITNESS_NEGATED = numpy.array([
    [-1.0, -6.0],   # s1
    [-2.0, -4.5],   # s2
    [-3.0, -3.0],   # s3
    [-4.5, -2.0],   # s4
    [-6.0, -1.0],   # s5
    [-4.0, -4.0],   # s6
])


def test_generate_reference_points_count_matches_binomial_for_M2_p3(nsga3):
    points = nsga3.generate_reference_points(num_objectives=2, num_divisions=3)
    assert points.shape == (math.comb(2 + 3 - 1, 3), 2)


def test_generate_reference_points_count_matches_binomial_for_M3_p4(nsga3):
    points = nsga3.generate_reference_points(num_objectives=3, num_divisions=4)
    assert points.shape == (math.comb(3 + 4 - 1, 4), 3)


def test_generate_reference_points_count_matches_binomial_for_M5_p4(nsga3):
    points = nsga3.generate_reference_points(num_objectives=5, num_divisions=4)
    assert points.shape == (math.comb(5 + 4 - 1, 4), 5)


def test_generate_reference_points_rows_sum_to_one(nsga3):
    points = nsga3.generate_reference_points(num_objectives=3, num_divisions=4)
    numpy.testing.assert_allclose(points.sum(axis=1), 1.0, atol=1e-12)


def test_generate_reference_points_M2_p3_matches_expected_set(nsga3):
    points = nsga3.generate_reference_points(num_objectives=2, num_divisions=3)
    expected = numpy.array([
        [3 / 3, 0 / 3],
        [2 / 3, 1 / 3],
        [1 / 3, 2 / 3],
        [0 / 3, 3 / 3],
    ])
    sorted_actual = numpy.array(sorted(points.tolist(), reverse=True))
    sorted_expected = numpy.array(sorted(expected.tolist(), reverse=True))
    numpy.testing.assert_allclose(sorted_actual, sorted_expected, atol=1e-12)


def test_compute_ideal_point_takes_column_max(nsga3):
    fitness = numpy.array([
        [1.0, 5.0],
        [3.0, 2.0],
        [0.0, 4.0],
    ])
    ideal = nsga3.compute_ideal_point(fitness)
    numpy.testing.assert_allclose(ideal, [3.0, 5.0])


def test_compute_ideal_point_on_negated_six_solution_set(nsga3):
    ideal = nsga3.compute_ideal_point(GUIDE_FITNESS_NEGATED)
    numpy.testing.assert_allclose(ideal, [-1.0, -1.0])


def test_find_extreme_points_picks_s5_for_f1_axis(nsga3):
    ideal = nsga3.compute_ideal_point(GUIDE_FITNESS_NEGATED)
    extremes = nsga3.find_extreme_points(GUIDE_FITNESS_NEGATED, ideal)
    numpy.testing.assert_allclose(extremes[0], [-6.0, -1.0])


def test_find_extreme_points_picks_s1_for_f2_axis(nsga3):
    ideal = nsga3.compute_ideal_point(GUIDE_FITNESS_NEGATED)
    extremes = nsga3.find_extreme_points(GUIDE_FITNESS_NEGATED, ideal)
    numpy.testing.assert_allclose(extremes[1], [-1.0, -6.0])


def test_compute_intercepts_six_solution_set_returns_minus_six(nsga3):
    # Intercept point sits at ideal + 1/b, where b solves
    # (extremes - ideal) @ b = 1. For this dataset both axes give -6.
    # The extreme rows then normalise to the simplex corners (1, 0) and
    # (0, 1).
    ideal = nsga3.compute_ideal_point(GUIDE_FITNESS_NEGATED)
    extremes = nsga3.find_extreme_points(GUIDE_FITNESS_NEGATED, ideal)
    intercepts = nsga3.compute_intercepts(extremes, ideal, GUIDE_FITNESS_NEGATED)
    numpy.testing.assert_allclose(intercepts, [-6.0, -6.0], atol=1e-9)


def test_compute_intercepts_falls_back_to_nadir_on_singular_extremes(nsga3):
    ideal = numpy.array([0.0, 0.0])
    duplicate_extremes = numpy.array([
        [-3.0, -2.0],
        [-3.0, -2.0],
    ])
    pool = numpy.array([
        [-3.0, -2.0],
        [-1.5, -4.0],
    ])
    intercepts = nsga3.compute_intercepts(duplicate_extremes, ideal, pool)
    numpy.testing.assert_allclose(intercepts, pool.min(axis=0))


def test_normalise_fitness_places_extremes_at_simplex_corners(nsga3):
    # With intercepts = (-6, -6) and ideal = (-1, -1) the denominator
    # (intercepts - ideal) is (-5, -5) and the formula
    # (f - ideal) / (intercepts - ideal) maps each row to a point inside
    # the unit simplex. The two axis extremes (s5 and s1) land exactly on
    # the simplex corners.
    ideal = nsga3.compute_ideal_point(GUIDE_FITNESS_NEGATED)
    extremes = nsga3.find_extreme_points(GUIDE_FITNESS_NEGATED, ideal)
    intercepts = nsga3.compute_intercepts(extremes, ideal, GUIDE_FITNESS_NEGATED)
    normalised = nsga3.normalise_fitness(GUIDE_FITNESS_NEGATED, ideal, intercepts)
    expected = numpy.array([
        [0.0, 1.0],   # s1 -> simplex corner on f2
        [0.2, 0.7],   # s2
        [0.4, 0.4],   # s3
        [0.7, 0.2],   # s4
        [1.0, 0.0],   # s5 -> simplex corner on f1
        [0.6, 0.6],   # s6 (dominated)
    ])
    numpy.testing.assert_allclose(normalised, expected, atol=1e-9)


def test_normalise_fitness_clips_above_one_and_below_zero(nsga3):
    # First row sits "above" the ideal under maximisation (raw values
    # bigger than the ideal) so the formula would produce a negative
    # ratio. Second row sits below the intercept and would produce a
    # ratio above 1. Both must be clipped back to [0, 1].
    ideal = numpy.array([0.0, 0.0])
    intercepts = numpy.array([-1.0, -1.0])
    fitness = numpy.array([
        [0.5, 0.5],
        [-2.0, -2.0],
    ])
    normalised = nsga3.normalise_fitness(fitness, ideal, intercepts)
    assert normalised.min() >= 0.0
    assert normalised.max() <= 1.0


def test_normalise_fitness_handles_near_zero_negative_denominator(nsga3):
    # Intercept sits within 1e-12 of the ideal so the denominator
    # collapses to a tiny negative. The safeguard must keep the sign
    # negative so (fitness - ideal) / denom comes out positive (and
    # then clips to 1.0). A buggy safeguard that lets the denom flip
    # to zero or positive would produce inf / nan or 0.0 instead.
    ideal = numpy.array([0.0])
    intercepts = numpy.array([-1e-15])
    fitness = numpy.array([[-1.0]])
    normalised = nsga3.normalise_fitness(fitness, ideal, intercepts)
    assert numpy.all(numpy.isfinite(normalised))
    assert normalised[0, 0] == pytest.approx(1.0)


# Reference points for M=2, p=3 in the order generate_reference_points
# emits them (stars-and-bars enumeration).
REFERENCE_POINTS_M2_P3 = numpy.array([
    [1.0,    0.0  ],   # ref 0
    [2 / 3,  1 / 3],   # ref 1
    [1 / 3,  2 / 3],   # ref 2
    [0.0,    1.0  ],   # ref 3
])


def test_associate_picks_nearest_reference_line(nsga3):
    # The point (0, 1) lies on the f2 axis and is collinear with ref 3.
    # Perpendicular distance is zero.
    point = numpy.array([[0.0, 1.0]])
    nearest, distance = nsga3.associate_to_reference_points(point, REFERENCE_POINTS_M2_P3)
    assert nearest[0] == 3
    assert distance[0] == pytest.approx(0.0, abs=1e-12)


def test_associate_breaks_ties_by_lower_reference_index(nsga3):
    # The point (0.6, 0.6) sits on the diagonal and is the same distance
    # from ref 1 and ref 2. The lower index wins.
    point = numpy.array([[0.6, 0.6]])
    nearest, _ = nsga3.associate_to_reference_points(point, REFERENCE_POINTS_M2_P3)
    assert nearest[0] == 1


def test_associate_perpendicular_distance_for_diagonal_point(nsga3):
    # Same diagonal point. Expected distance ~ 0.2683 computed by hand
    # from the formula || x - (x . z_hat) z_hat ||.
    point = numpy.array([[0.6, 0.6]])
    _, distance = nsga3.associate_to_reference_points(point, REFERENCE_POINTS_M2_P3)
    assert distance[0] == pytest.approx(0.2683, abs=1e-3)


def test_niching_with_single_fl_candidate_returns_that_candidate(nsga3):
    # Only one candidate is available in Fl, so it must be selected.
    fl_indices = [42]
    fl_assoc = numpy.array([1])
    fl_dist = numpy.array([0.224])
    accepted_assoc = numpy.array([3, 2, 1, 1, 0])
    picked = nsga3.niching_select(fl_indices=fl_indices,
                                  fl_assoc=fl_assoc,
                                  fl_dist=fl_dist,
                                  accepted_assoc=accepted_assoc,
                                  num_reference_points=4,
                                  K=1)
    assert picked == [42]


def test_niching_picks_candidate_in_lower_rho_niche(nsga3):
    # Two Fl candidates. The first is associated with ref 1 where rho=2;
    # the second is associated with ref 2 where rho=1. Niching prefers
    # the lower rho, so the second candidate wins.
    fl_indices = [60, 70]
    fl_assoc = numpy.array([1, 2])
    fl_dist = numpy.array([0.224, 0.10])
    accepted_assoc = numpy.array([3, 2, 1, 1, 0])
    picked = nsga3.niching_select(fl_indices=fl_indices,
                                  fl_assoc=fl_assoc,
                                  fl_dist=fl_dist,
                                  accepted_assoc=accepted_assoc,
                                  num_reference_points=4,
                                  K=1)
    assert picked == [70]


def test_niching_picks_smallest_distance_when_rho_is_zero(nsga3):
    # Both candidates are at ref 1 and rho_1 = 0 (empty niche). The
    # closer candidate wins (distance 0.158 < 0.224).
    fl_indices = [60, 70]
    fl_assoc = numpy.array([1, 1])
    fl_dist = numpy.array([0.224, 0.158])
    accepted_assoc = numpy.array([3, 2, 0])
    picked = nsga3.niching_select(fl_indices=fl_indices,
                                  fl_assoc=fl_assoc,
                                  fl_dist=fl_dist,
                                  accepted_assoc=accepted_assoc,
                                  num_reference_points=4,
                                  K=1)
    assert picked == [70]


def test_niching_picks_from_candidate_pool_when_rho_is_positive(nsga3):
    # Both candidates are at ref 1 and rho_1 > 0, so the pick is random.
    # Run 50 different seeds and verify that the chosen candidate always
    # comes from {60, 70} and that both candidates show up over the run.
    fl_indices = [60, 70]
    fl_assoc = numpy.array([1, 1])
    fl_dist = numpy.array([0.224, 0.158])
    accepted_assoc = numpy.array([3, 2, 1, 1, 0])
    seen = set()
    rng_state = numpy.random.get_state()
    try:
        for seed in range(50):
            numpy.random.seed(seed)
            picked = nsga3.niching_select(fl_indices=fl_indices,
                                          fl_assoc=fl_assoc,
                                          fl_dist=fl_dist,
                                          accepted_assoc=accepted_assoc,
                                          num_reference_points=4,
                                          K=1)
            assert picked[0] in {60, 70}
            seen.add(picked[0])
    finally:
        numpy.random.set_state(rng_state)
    assert seen == {60, 70}


# Fitness helpers used by the integration tests below. The scalar one
# returns a single number so we can check that NSGA-III rejects it; the
# other two return a list of objectives.

def _scalar_fitness(ga, solution, sol_idx):
    return float(numpy.sum(solution))


def _two_objective_fitness(ga, solution, sol_idx):
    return [float(numpy.sum(solution)), -float(numpy.sum(solution ** 2))]


def _three_objective_fitness(ga, solution, sol_idx):
    return [float(solution[0]), float(solution[1]), float(solution[2])]


def test_nsga3_requires_nsga3_num_divisions():
    with pytest.raises(ValueError, match="nsga3_num_divisions"):
        pygad.GA(num_generations=2,
                 num_parents_mating=3,
                 fitness_func=_two_objective_fitness,
                 sol_per_pop=8,
                 num_genes=4,
                 parent_selection_type='nsga3',
                 suppress_warnings=True)


def test_nsga3_rejects_non_positive_nsga3_num_divisions():
    with pytest.raises(ValueError, match="nsga3_num_divisions"):
        pygad.GA(num_generations=2,
                 num_parents_mating=3,
                 fitness_func=_two_objective_fitness,
                 sol_per_pop=8,
                 num_genes=4,
                 parent_selection_type='nsga3',
                 nsga3_num_divisions=0,
                 suppress_warnings=True)


def test_nsga3_rejects_single_objective_problem():
    ga = pygad.GA(num_generations=2,
                  num_parents_mating=3,
                  fitness_func=_scalar_fitness,
                  sol_per_pop=8,
                  num_genes=4,
                  parent_selection_type='nsga3',
                  nsga3_num_divisions=4,
                  suppress_warnings=True)
    with pytest.raises(TypeError, match="single-objective"):
        ga.run()


def test_tournament_nsga3_rejects_single_objective_problem():
    ga = pygad.GA(num_generations=2,
                  num_parents_mating=3,
                  fitness_func=_scalar_fitness,
                  sol_per_pop=8,
                  num_genes=4,
                  parent_selection_type='tournament_nsga3',
                  nsga3_num_divisions=4,
                  K_tournament=2,
                  suppress_warnings=True)
    with pytest.raises(TypeError, match="single-objective"):
        ga.run()


def test_nsga3_bootstrap_generates_reference_points_with_expected_shape():
    ga = pygad.GA(num_generations=2,
                  num_parents_mating=5,
                  fitness_func=_three_objective_fitness,
                  sol_per_pop=15,
                  num_genes=4,
                  parent_selection_type='nsga3',
                  nsga3_num_divisions=4,
                  random_seed=1,
                  suppress_warnings=True)
    ga.run()
    assert ga.nsga3_reference_points.shape == (15, 3)


def test_sol_per_pop_below_reference_count_triggers_warning_and_grows_population():
    # M=3, p=4 needs 15 reference points but sol_per_pop is only 8. The
    # GA should warn once, grow the population to 15, and re-evaluate
    # fitness before the generational loop starts.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ga = pygad.GA(num_generations=2,
                      num_parents_mating=3,
                      fitness_func=_three_objective_fitness,
                      sol_per_pop=8,
                      num_genes=4,
                      parent_selection_type='nsga3',
                      nsga3_num_divisions=4,
                      random_seed=1)
        ga.run()
    nsga3_warning_messages = [str(w.message) for w in caught
                              if "NSGA-III reference points" in str(w.message)]
    assert len(nsga3_warning_messages) == 1
    assert ga.sol_per_pop == 15
    assert ga.population.shape[0] == 15


def test_sol_per_pop_auto_grow_also_fires_for_tournament_nsga3():
    # Same scenario but using the tournament-based NSGA-III selection.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ga = pygad.GA(num_generations=2,
                      num_parents_mating=3,
                      fitness_func=_three_objective_fitness,
                      sol_per_pop=8,
                      num_genes=4,
                      parent_selection_type='tournament_nsga3',
                      nsga3_num_divisions=4,
                      K_tournament=2,
                      random_seed=2)
        ga.run()
    grown_warning_messages = [str(w.message) for w in caught
                              if "NSGA-III reference points" in str(w.message)]
    assert len(grown_warning_messages) == 1
    assert ga.sol_per_pop == 15
