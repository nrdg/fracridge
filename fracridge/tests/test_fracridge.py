import numpy as np
from fracridge import (fracridge, vec_len, FracRidgeRegressor,
                       FracRidgeRegressorCV)
from sklearn.linear_model import Ridge
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks


@parametrize_with_checks([FracRidgeRegressor(), FracRidgeRegressorCV()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def run_fracridge(X, y, fracs, jit):
    fracridge(X, y, fracs=fracs, jit=jit)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("jit", [True, False])
def test_benchmark_fracridge(nn, pp, bb, jit, benchmark):
    X, y, _, _ = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    benchmark(run_fracridge, X, y, fracs, jit)


def make_data(nn, pp, bb, fit_intercept=False):
    np.random.seed(1)
    X = np.random.randn(nn, pp)
    y = np.random.randn(nn, bb).squeeze()
    if fit_intercept:
        X = np.hstack([X, np.ones(X.shape[0])[:, np.newaxis]])
    coef_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
    pred_ols = X @ coef_ols
    if fit_intercept:
        coef_ols = coef_ols[:-1]
        X = X[:, :-1]
    return X, y, coef_ols, pred_ols


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
def test_fracridge_ols(nn, pp, bb):
    X, y, coef_ols, _ = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    coef, alpha = fracridge(X, y, fracs=fracs)
    coef = coef[:, -1, ...]
    assert np.allclose(coef, coef_ols, atol=10e-3)
    assert np.all(np.diff(alpha, axis=0) <= 0)


@pytest.mark.parametrize("frac", [0.1, 0.23, 1])
@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
def test_fracridge_fracs(frac, nn, pp, bb):
    X, y, coef_ols, _ = make_data(nn, pp, bb)
    # Make sure that you get the fraction you asked for
    coef, _ = fracridge(X, y, fracs=np.array([frac]))
    assert np.all(
        np.abs(
            frac
            - vec_len(coef, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("fit_intercept", [False])
def test_v_ols(nn, pp, bb, fit_intercept):
    X, y, coef_ols, _ = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    FR = FracRidgeRegressor(fracs=fracs, fit_intercept=fit_intercept)
    FR.fit(X, y)
    assert np.allclose(FR.coef_[:, -1, ...], coef_ols, atol=10e-3)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("frac", [0.1, 0.23, 1])
def test_v_fracs(nn, pp, bb, frac):
    X, y, coef_ols, _ = make_data(nn, pp, bb)
    FR = FracRidgeRegressor(fracs=frac)
    FR.fit(X, y)
    assert np.all(
        np.abs(
            frac
            - vec_len(FR.coef_, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("jit", [True, False])
def test_FracRidgeRegressor_predict(nn, pp, bb, fit_intercept, jit):
    X, y, coef_ols, pred_ols = make_data(nn, pp, bb, fit_intercept)
    fracs = np.arange(.1, 1.1, .1)
    FR = FracRidgeRegressor(fracs=fracs, fit_intercept=fit_intercept, jit=jit)
    FR.fit(X, y)
    pred_fr = FR.predict(X)
    assert np.allclose(pred_fr[:, -1, ...], pred_ols, atol=10e-3)


def test_FracRidge_singleton_frac():
    X = np.array([[1.64644051],
                  [2.1455681]])
    y = np.array([1., 2.])
    fracs = 0.1
    FR = FracRidgeRegressor(fracs=fracs)
    FR.fit(X, y)
    pred_fr = FR.predict(X)
    assert pred_fr.shape == y.shape

@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("jit", [True, False])
def test_FracRidgeRegressorCV(nn, pp, bb, fit_intercept, jit):
    X, y, _, _ = make_data(nn, pp, bb, fit_intercept)
    fracs = np.arange(.1, 1.1, .1)
    FRCV = FracRidgeRegressorCV(fit_intercept=fit_intercept,
                                jit=jit)
    FRCV.fit(X, y, frac_grid=fracs)

    FR = FracRidgeRegressor(fracs=FRCV.best_frac_,
                            fit_intercept=fit_intercept)
    FR.fit(X, y)
    assert np.allclose(FR.coef_, FRCV.coef_, atol=10e-3)
    RR = Ridge(alpha=FRCV.alpha_, fit_intercept=fit_intercept,
               solver='svd')
    RR.fit(X, y)
    # The coefficients in the sklearn object are transposed relative to
    # our conventions:
    assert np.allclose(RR.coef_.T, FRCV.coef_, atol=10e-3)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100), (284, 50)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
def test_fracridge_unsorted(nn, pp, bb):
    X, y, coef_ols, _ = make_data(nn, pp, bb)
    fracs = np.array([0.1, 0.8, 1.0, 0.2])
    # Frac input needs to be sorted:
    with pytest.raises(ValueError):
        coef, alpha = fracridge(X, y, fracs=fracs)


@pytest.mark.parametrize("nn", [(1000), (10), (284)])
@pytest.mark.parametrize("bb", [(1), (2), (1000)])
@pytest.mark.parametrize("fit_intercept", [(False), (True)])
def test_fracridge_single_regressor(nn, bb, fit_intercept):
    # Sometimes we want to have just one regressor
    # See: https://github.com/nrdg/fracridge/issues/24
    pp = 1
    X, y, _, pred_ols = make_data(nn, pp, bb,
                                  fit_intercept=fit_intercept)
    fracs = np.arange(.1, 1.1, .1)
    FR = FracRidgeRegressor(fracs=fracs, fit_intercept=fit_intercept)
    FR.fit(X, y)
    pred_fr = FR.predict(X).squeeze()
    assert np.allclose(pred_fr[:, -1, ...], pred_ols, atol=10e-3)


@pytest.mark.parametrize("nn", [(1000), (10), (284)])
@pytest.mark.parametrize("pp", [(1), (2), (1000)])
@pytest.mark.parametrize("fit_intercept", [(False), (True)])
@pytest.mark.parametrize("fracs", [(None), (1.0)])
@pytest.mark.parametrize("y_is_a_single_column_matrix", [(True), (False)])
def test_fracridge_singleton_target(nn, pp, fit_intercept, fracs, y_is_a_single_column_matrix):
    # Sometimes we want to have just one target, and this should work for either a vector y or single-column matrix y,
    # See: https://github.com/nrdg/fracridge/issues/39
    bb = 1

    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    X, y, _, pred_ols = make_data(nn, pp, bb,
                                  fit_intercept=fit_intercept)
    if y_is_a_single_column_matrix: # expand y from a vector to a one-column matrix
        y = np.expand_dims(y, axis=-1)
        pred_ols = np.expand_dims(pred_ols,-1) # do the same for the ols prediction
    # else, y remains a vector

    FR = FracRidgeRegressor(fracs=fracs, fit_intercept=fit_intercept)
    FR.fit(X, y)
    pred_fr = FR.predict(X)
    if hasattr(fracs, "__len__"): # if fracs is a a vector, extract the 1.0 fraction coefficient column (which should be similar to the OLS solution)
        pred_fr=pred_fr[:,np.asarray(fracs)==1.0,...].squeeze(1)

    assert np.allclose(pred_fr, pred_ols, atol=10e-3)
    assert np.array_equal(y.shape,pred_fr.shape) # predictions should be shaped like y
