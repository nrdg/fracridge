import numpy as np
import pytest
from sklearn.model_selection import KFold

from fracridge.fracridge import FracRidgeRegressorCV

INTERCEPT = 3.0
SLOPE = 2.0
SECOND_INTERCEPT = -1.5
SECOND_SLOPE = 0.5


@pytest.fixture
def affine_data():
    """Noiseless affine data obeying y = 3 + 2X exactly."""
    X = np.linspace(0.0, 1.0, 101)[:, np.newaxis]
    y = INTERCEPT + SLOPE * X[:, 0]
    return X, y


@pytest.fixture
def fixed_five_fold():
    """A deterministic five-fold split, so failures are reproducible."""
    return KFold(n_splits=5, shuffle=True, random_state=20260804)


def test_cv_constructor_leaves_fracs_unset():
    """The constructor used to pass `self` as the base class's `fracs`
    argument, leaving a freshly built estimator holding itself as a parameter
    value.
    """
    assert FracRidgeRegressorCV().fracs is None


def test_cv_recovers_nonzero_intercept(affine_data, fixed_five_fold):
    """The outer estimator used to center the whole dataset before the folds
    were built, so the intercept it copied was defined in centered coordinates
    while `predict` received raw X. The slope survived; the offset did not.
    """
    X, y = affine_data
    model = FracRidgeRegressorCV(
        fit_intercept=True, cv=fixed_five_fold, jit=False
    )
    model.fit(X, y, frac_grid=np.array([1.0]))

    assert np.isclose(
        float(np.asarray(model.intercept_).squeeze()), INTERCEPT, atol=1e-10
    )
    assert np.isclose(float(np.asarray(model.coef_).squeeze()), SLOPE, atol=1e-10)
    assert np.allclose(model.predict(X), y, atol=1e-10)
    assert np.isclose(model.score(X, y), 1.0, atol=1e-12)


def test_cv_recovers_two_target_intercepts(affine_data, fixed_five_fold):
    """The same lost-offset bug with two targets, which also pins the
    per-target intercept and coefficient shapes alongside their values.
    """
    X, y = affine_data
    targets = np.column_stack([y, SECOND_INTERCEPT + SECOND_SLOPE * X[:, 0]])
    model = FracRidgeRegressorCV(
        fit_intercept=True, cv=fixed_five_fold, jit=False
    )
    model.fit(X, targets, frac_grid=np.array([1.0]))

    assert model.coef_.shape == (1, 2)
    assert np.asarray(model.intercept_).shape == (2,)
    assert np.allclose(model.coef_.ravel(), [SLOPE, SECOND_SLOPE], atol=1e-10)
    assert np.allclose(
        np.asarray(model.intercept_), [INTERCEPT, SECOND_INTERCEPT], atol=1e-10
    )
    assert np.allclose(model.predict(X), targets, atol=1e-10)


def test_cv_without_intercept_is_unchanged(fixed_five_fold):
    """Control for the intercept fix: with `fit_intercept=False` no centering
    happens on either code path, so this behavior must be identical before and
    after, down to the copied fitted state.
    """
    rng = np.random.default_rng(20260804)
    X = rng.normal(size=(80, 5))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    model = FracRidgeRegressorCV(
        fit_intercept=False, cv=fixed_five_fold, jit=False
    )
    model.fit(X, y, frac_grid=np.arange(0.1, 1.1, 0.1))

    assert model.intercept_ == 0.0
    assert model.n_features_in_ == 5
    assert model.is_fitted_
    assert np.isclose(model.best_frac_, 1.0)
    assert np.isclose(model.score(X, y), 1.0, atol=1e-10)
