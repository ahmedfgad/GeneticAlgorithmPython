"""
Pipeline-level tests for NSGA-III using hand-built ground-truth values.

Every test in this file pins the output of one NSGA-III step (reference
point generation, ideal point, extreme points, intercepts, normalized
fitness, association, niching, or the full pipeline) to a hardcoded
value derived from a small dataset whose answer can be checked on paper.

The main dataset is THREE_OBJECTIVE_FITNESS: seven solutions in M=3
space whose hand-derived ideal point, extreme points, intercepts, and
normalized positions are all simple round numbers. This makes the
expected outputs easy to verify without re-running the algorithm.
"""

import numpy
import pytest

from pygad.utils.nsga3 import NSGA3


# Three-objective dataset used by most tests below. Values are expressed
# in PyGAD's maximization convention, so all fitness values are <= 0 and
# the ideal point sits at the origin.
#
#   s0..s2  : axis extremes (best on one objective, ideal on the others).
#   s3..s5  : midpoints of each edge of the simplex.
#   s6      : centre of the unit simplex.
THREE_OBJECTIVE_FITNESS = numpy.array([
    [-1.0,  0.0,  0.0],   # s0 — extreme for f0
    [ 0.0, -1.0,  0.0],   # s1 — extreme for f1
    [ 0.0,  0.0, -1.0],   # s2 — extreme for f2
    [-0.5, -0.5,  0.0],   # s3
    [-0.5,  0.0, -0.5],   # s4
    [ 0.0, -0.5, -0.5],   # s5
    [-1 / 3, -1 / 3, -1 / 3],   # s6 — simplex centre
])


@pytest.fixture
def nsga3():
    return NSGA3()


@pytest.mark.parametrize("num_objectives,num_divisions,expected_count", [
    (2, 3, 4),
    (3, 4, 15),
    (3, 12, 91),
    (5, 4, 70),
    (8, 3, 120),
])
def test_reference_point_count_matches_binomial(nsga3, num_objectives,
                                                num_divisions, expected_count):
    points = nsga3.nsga3_generate_reference_points(num_objectives, num_divisions)
    assert points.shape == (expected_count, num_objectives)
    numpy.testing.assert_allclose(points.sum(axis=1), 1.0, atol=1e-12)


def test_reference_points_M3_p2_match_expected_set(nsga3):
    points = nsga3.nsga3_generate_reference_points(3, 2)
    expected = numpy.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0],
    ])
    sorted_actual = numpy.array(sorted(points.tolist(), reverse=True))
    sorted_expected = numpy.array(sorted(expected.tolist(), reverse=True))
    numpy.testing.assert_allclose(sorted_actual, sorted_expected, atol=1e-12)


def test_ideal_point_for_three_objective_set(nsga3):
    ideal = nsga3.nsga3_compute_ideal_point(THREE_OBJECTIVE_FITNESS)
    numpy.testing.assert_allclose(ideal, [0.0, 0.0, 0.0])


def test_extreme_points_for_three_objective_set(nsga3):
    ideal = nsga3.nsga3_compute_ideal_point(THREE_OBJECTIVE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(THREE_OBJECTIVE_FITNESS, ideal)
    expected = numpy.array([
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0],
    ])
    numpy.testing.assert_allclose(extremes, expected, atol=1e-12)


def test_intercepts_for_three_objective_set(nsga3):
    ideal = nsga3.nsga3_compute_ideal_point(THREE_OBJECTIVE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(THREE_OBJECTIVE_FITNESS, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, THREE_OBJECTIVE_FITNESS)
    numpy.testing.assert_allclose(intercepts, [-1.0, -1.0, -1.0], atol=1e-12)


def test_intercepts_cap_at_worst_observed_per_objective(nsga3):
    # Make the linear solve extrapolate well beyond the actual data:
    # the extreme points are packed close to the ideal so 1/b is large,
    # but the pool's worst values per axis sit much closer to the ideal.
    # The cap should pull each intercept back to the worst observed
    # value.
    ideal = numpy.array([0.0, 0.0])
    extremes = numpy.array([
        [-0.001, -0.5],
        [-0.5,  -0.001],
    ])
    pool = numpy.array([
        [-0.001, -0.5],
        [-0.5,  -0.001],
        [-0.1,  -0.1],
    ])
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, pool)
    numpy.testing.assert_allclose(intercepts, pool.min(axis=0), atol=1e-12)


def test_intercepts_fall_back_when_extremes_singular(nsga3):
    # Both extreme points are the same row, so the linear system is
    # singular. The function must fall back to the worst per objective.
    ideal = numpy.array([0.0, 0.0])
    extremes = numpy.array([
        [-1.0, -1.0],
        [-1.0, -1.0],
    ])
    pool = numpy.array([
        [-1.0, -1.0],
        [-2.0, -0.5],
    ])
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, pool)
    numpy.testing.assert_allclose(intercepts, pool.min(axis=0))


def test_normalized_fitness_for_three_objective_set(nsga3):
    ideal = nsga3.nsga3_compute_ideal_point(THREE_OBJECTIVE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(THREE_OBJECTIVE_FITNESS, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, THREE_OBJECTIVE_FITNESS)
    normalized = nsga3.nsga3_normalize_fitness(THREE_OBJECTIVE_FITNESS, ideal, intercepts)
    expected = numpy.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
    ])
    numpy.testing.assert_allclose(normalized, expected, atol=1e-12)


def test_associations_for_three_objective_set(nsga3):
    # Reference points produced by nsga3_generate_reference_points(3, 2),
    # in the order our enumeration emits them:
    #   ref[0] = (0,   0,   1  )
    #   ref[1] = (0,   0.5, 0.5)
    #   ref[2] = (0,   1,   0  )
    #   ref[3] = (0.5, 0,   0.5)
    #   ref[4] = (0.5, 0.5, 0  )
    #   ref[5] = (1,   0,   0  )
    # Each on-simplex solution sits on one reference line and has zero
    # distance. The centre solution is the same distance from ref[1], ref[3]
    # and ref[4]; the lower-index tie break picks ref[1].
    nsga3_ref_points = nsga3.nsga3_generate_reference_points(3, 2)
    ideal = nsga3.nsga3_compute_ideal_point(THREE_OBJECTIVE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(THREE_OBJECTIVE_FITNESS, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, THREE_OBJECTIVE_FITNESS)
    normalized = nsga3.nsga3_normalize_fitness(THREE_OBJECTIVE_FITNESS, ideal, intercepts)
    nearest, distance = nsga3.nsga3_associate_to_reference_points(normalized, nsga3_ref_points)
    expected_nearest = numpy.array([5, 2, 0, 4, 3, 1, 1])
    expected_distance = numpy.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 3])
    numpy.testing.assert_array_equal(nearest, expected_nearest)
    numpy.testing.assert_allclose(distance, expected_distance, atol=1e-9)


def test_niching_picks_one_candidate_per_empty_niche(nsga3):
    # Three critical-front candidates, each attached to a different
    # empty niche. We need three survivors. Every empty niche must take
    # its only candidate.
    critical_front_indices = [10, 11, 12]
    critical_front_associations = numpy.array([0, 1, 2])
    critical_front_distances = numpy.array([0.01, 0.02, 0.03])
    accepted_associations = numpy.array([3, 3, 3])
    picked = nsga3.nsga3_niching_select(
        critical_front_indices=critical_front_indices,
        critical_front_associations=critical_front_associations,
        critical_front_distances=critical_front_distances,
        accepted_associations=accepted_associations,
        num_reference_points=4,
        num_to_select=3)
    assert set(picked) == {10, 11, 12}


def test_niching_picks_closest_candidate_when_niche_count_is_zero(nsga3):
    # Two candidates at the same empty niche. The closer one wins
    # deterministically.
    critical_front_indices = [20, 21]
    critical_front_associations = numpy.array([0, 0])
    critical_front_distances = numpy.array([0.5, 0.1])
    accepted_associations = numpy.array([1, 2, 3])
    picked = nsga3.nsga3_niching_select(
        critical_front_indices=critical_front_indices,
        critical_front_associations=critical_front_associations,
        critical_front_distances=critical_front_distances,
        accepted_associations=accepted_associations,
        num_reference_points=4,
        num_to_select=1)
    assert picked == [21]


def test_niching_picks_candidate_in_lower_niche_count_with_unique_owner(nsga3):
    # ref 0 has niche count 2, ref 1 has niche count 3. We want the
    # candidate at ref 0 because its niche count is smaller.
    critical_front_indices = [30, 31]
    critical_front_associations = numpy.array([0, 1])
    critical_front_distances = numpy.array([0.4, 0.2])
    accepted_associations = numpy.array([0, 0, 1, 1, 1])
    picked = nsga3.nsga3_niching_select(
        critical_front_indices=critical_front_indices,
        critical_front_associations=critical_front_associations,
        critical_front_distances=critical_front_distances,
        accepted_associations=accepted_associations,
        num_reference_points=4,
        num_to_select=1)
    assert picked == [30]


def test_full_pipeline_recovers_simplex_corners(nsga3):
    # Run the full pipeline on the THREE_OBJECTIVE_FITNESS dataset and
    # verify that the six on-simplex solutions cover six different
    # reference points with zero perpendicular distance.
    nsga3_ref_points = nsga3.nsga3_generate_reference_points(3, 2)
    fitness = THREE_OBJECTIVE_FITNESS
    ideal = nsga3.nsga3_compute_ideal_point(fitness)
    extremes = nsga3.nsga3_find_extreme_points(fitness, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, fitness)
    normalized = nsga3.nsga3_normalize_fitness(fitness, ideal, intercepts)
    nearest, distance = nsga3.nsga3_associate_to_reference_points(normalized, nsga3_ref_points)
    covered = numpy.unique(nearest[:6])
    assert set(covered.tolist()) == {0, 1, 2, 3, 4, 5}
    assert numpy.all(distance[:6] < 1e-9)


# Three solutions used by the wide-range and narrow-range normalization
# tests below. The first row is the f0 extreme, the second is the f1
# extreme, and the third is the middle point of the front. Both versions
# of the dataset must produce the same normalized positions because the
# NSGA-III normalization is invariant to positive affine transforms of
# the fitness.
WIDE_RANGE_FITNESS = numpy.array([
    [15.0, -10.0],   # s0 — best f0, worst f1
    [-10.0,  15.0],  # s1 — worst f0, best f1
    [  0.0,   0.0],  # s2 — middle
])

NARROW_RANGE_FITNESS = numpy.array([
    [0.7, 0.3],   # s0
    [0.3, 0.7],   # s1
    [0.5, 0.5],   # s2
])


def test_normalize_fitness_for_wide_range_input(nsga3):
    # Fitness values cross zero and span 25 units per axis. The
    # algorithm must still map the two extremes onto the simplex corners
    # and the middle point to (0.6, 0.6).
    ideal = nsga3.nsga3_compute_ideal_point(WIDE_RANGE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(WIDE_RANGE_FITNESS, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, WIDE_RANGE_FITNESS)
    normalized = nsga3.nsga3_normalize_fitness(WIDE_RANGE_FITNESS, ideal, intercepts)
    numpy.testing.assert_allclose(ideal, [15.0, 15.0])
    numpy.testing.assert_allclose(intercepts, [-10.0, -10.0])
    expected_normalized = numpy.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.6, 0.6],
    ])
    numpy.testing.assert_allclose(normalized, expected_normalized, atol=1e-12)


def test_normalize_fitness_for_narrow_range_input(nsga3):
    # Fitness values are all inside [0.3, 0.7] (a 0.4-wide window).
    # Normalization must still pin the two extremes to the simplex
    # corners and place the middle point at (0.5, 0.5).
    ideal = nsga3.nsga3_compute_ideal_point(NARROW_RANGE_FITNESS)
    extremes = nsga3.nsga3_find_extreme_points(NARROW_RANGE_FITNESS, ideal)
    intercepts = nsga3.nsga3_compute_intercepts(extremes, ideal, NARROW_RANGE_FITNESS)
    normalized = nsga3.nsga3_normalize_fitness(NARROW_RANGE_FITNESS, ideal, intercepts)
    numpy.testing.assert_allclose(ideal, [0.7, 0.7])
    numpy.testing.assert_allclose(intercepts, [0.3, 0.3])
    expected_normalized = numpy.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
    ])
    numpy.testing.assert_allclose(normalized, expected_normalized, atol=1e-12)


@pytest.mark.parametrize("scale,shift", [
    (1.0,  0.0),     # identity
    (37.0, 0.0),     # pure positive scale
    (0.01, 0.0),     # pure positive scale (shrink)
    (1.0,  100.0),   # pure shift up
    (1.0, -100.0),   # pure shift down
    (5.0, -12.5),    # mixed
])
def test_normalize_fitness_is_invariant_under_positive_affine_transforms(nsga3, scale, shift):
    # NSGA-III normalization should not care about the absolute scale or
    # offset of fitness as long as the transform is a positive affine
    # one. Verify by transforming the base dataset and checking that the
    # normalized positions match the untransformed reference.
    base = NARROW_RANGE_FITNESS
    transformed = scale * base + shift

    base_ideal = nsga3.nsga3_compute_ideal_point(base)
    base_extremes = nsga3.nsga3_find_extreme_points(base, base_ideal)
    base_intercepts = nsga3.nsga3_compute_intercepts(base_extremes, base_ideal, base)
    base_normalized = nsga3.nsga3_normalize_fitness(base, base_ideal, base_intercepts)

    transformed_ideal = nsga3.nsga3_compute_ideal_point(transformed)
    transformed_extremes = nsga3.nsga3_find_extreme_points(transformed, transformed_ideal)
    transformed_intercepts = nsga3.nsga3_compute_intercepts(transformed_extremes,
                                                            transformed_ideal,
                                                            transformed)
    transformed_normalized = nsga3.nsga3_normalize_fitness(transformed,
                                                           transformed_ideal,
                                                           transformed_intercepts)
    numpy.testing.assert_allclose(transformed_normalized, base_normalized, atol=1e-9)
