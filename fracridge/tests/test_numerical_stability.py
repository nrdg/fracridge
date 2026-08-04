import warnings

import numpy as np
import pytest

from fracridge.fracridge import fracridge, vec_len

RELATIVE_TOLERANCE = 1e-6
MINIMUM_NORM_TOLERANCE = 1e-8
FRACTION_TOLERANCE = 0.01
NEAR_DUPLICATE_EPSILON = 1e-8
TRUE_COEFFICIENTS = np.array([1.0, -1.0, 0.5, -0.25, 0.125])
SECOND_TARGET_COEFFICIENTS = np.array([0.2, 0.3, -1.0, 2.0, 0.0])


@pytest.fixture
def collinear_columns():
    """Fixed base columns plus one fixed perturbation direction."""
    rng = np.random.default_rng(20260804)
    n_samples = 240
    return {
        "base": rng.normal(size=n_samples),
        "perturbation": rng.normal(size=n_samples),
        "extras": rng.normal(size=(n_samples, 3)),
    }


@pytest.fixture
def ill_conditioned_design(collinear_columns):
    """A tall design whose first two columns are nearly identical."""
    near_duplicate = (
        collinear_columns["base"]
        + NEAR_DUPLICATE_EPSILON * collinear_columns["perturbation"]
    )
    return np.column_stack(
        (collinear_columns["base"], near_duplicate, collinear_columns["extras"])
    )


@pytest.fixture
def rank_deficient_design(collinear_columns):
    """A tall design with an exactly duplicated column."""
    return np.column_stack(
        (
            collinear_columns["base"],
            collinear_columns["base"],
            collinear_columns["extras"],
        )
    )


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("jit", [False, True])
def test_ill_conditioned_tall_matches_lstsq(ill_conditioned_design, jit):
    y = ill_conditioned_design @ TRUE_COEFFICIENTS
    expected = np.linalg.lstsq(ill_conditioned_design, y, rcond=None)[0]
    coef, _ = fracridge(ill_conditioned_design, y, fracs=1.0, jit=jit)

    assert coef.shape == (ill_conditioned_design.shape[1],)
    assert relative_error(coef, expected) < RELATIVE_TOLERANCE


def test_ill_conditioned_tall_multi_target(ill_conditioned_design):
    targets = np.column_stack(
        [
            ill_conditioned_design @ TRUE_COEFFICIENTS,
            ill_conditioned_design @ SECOND_TARGET_COEFFICIENTS,
        ]
    )
    expected = np.linalg.lstsq(ill_conditioned_design, targets, rcond=None)[0]
    coef, _ = fracridge(ill_conditioned_design, targets, fracs=1.0, jit=False)

    assert coef.shape == (ill_conditioned_design.shape[1], 2)
    for target in range(targets.shape[1]):
        assert (
            relative_error(coef[:, target], expected[:, target])
            < RELATIVE_TOLERANCE
        )


@pytest.mark.parametrize("frac", [0.23, 0.5])
def test_ill_conditioned_tall_achieves_requested_fraction(
    ill_conditioned_design, frac
):
    y = ill_conditioned_design @ TRUE_COEFFICIENTS
    expected = np.linalg.lstsq(ill_conditioned_design, y, rcond=None)[0]
    coef, _ = fracridge(ill_conditioned_design, y, fracs=frac, jit=False)

    achieved = vec_len(coef) / vec_len(expected)
    assert abs(achieved - frac) < FRACTION_TOLERANCE


@pytest.mark.parametrize("jit", [False, True])
def test_rank_deficient_tall_is_finite_and_minimum_norm(
    rank_deficient_design, jit
):
    y = rank_deficient_design @ TRUE_COEFFICIENTS
    expected = np.linalg.lstsq(rank_deficient_design, y, rcond=None)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coef, alpha = fracridge(rank_deficient_design, y, fracs=1.0, jit=jit)

    assert np.all(np.isfinite(coef))
    assert np.all(np.isfinite(alpha))
    assert relative_error(coef, expected) < MINIMUM_NORM_TOLERANCE


@pytest.fixture
def zero_column_design(collinear_columns):
    """A tall design with one exactly-zero column."""
    return np.column_stack(
        (
            collinear_columns["base"],
            collinear_columns["extras"][:, 0],
            np.zeros_like(collinear_columns["base"]),
        )
    )


@pytest.mark.parametrize("jit", [False, True])
def test_zero_singular_value_is_finite(zero_column_design, jit):
    y = zero_column_design @ np.array([1.0, -0.5, 0.0])
    expected = np.linalg.lstsq(zero_column_design, y, rcond=None)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coef, alpha = fracridge(
            zero_column_design, y, fracs=np.array([0.3, 1.0]), jit=jit
        )

    assert np.all(np.isfinite(coef))
    assert np.all(np.isfinite(alpha))
    assert relative_error(coef[:, -1], expected) < MINIMUM_NORM_TOLERANCE


def test_zero_singular_value_multi_target_is_finite(zero_column_design):
    targets = np.column_stack(
        [
            zero_column_design @ np.array([1.0, -0.5, 0.0]),
            zero_column_design @ np.array([0.25, 2.0, 0.0]),
        ]
    )
    expected = np.linalg.lstsq(zero_column_design, targets, rcond=None)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        coef, alpha = fracridge(zero_column_design, targets, fracs=1.0, jit=False)

    assert coef.shape == (zero_column_design.shape[1], 2)
    assert np.all(np.isfinite(coef))
    assert np.all(np.isfinite(alpha))
    for target in range(targets.shape[1]):
        assert (
            relative_error(coef[:, target], expected[:, target])
            < MINIMUM_NORM_TOLERANCE
        )
