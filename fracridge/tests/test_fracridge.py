import numpy as np
from fracridge import fracridge, vec_len, FracRidge
from sklearn.utils.estimator_checks import check_estimator
import pytest


def make_data(nn, pp, bb):
    np.random.seed(1)
    X = np.random.randn(nn, pp)
    y = np.random.randn(nn, bb).squeeze()
    coef_ols = np.linalg.pinv(X.T @ X) @ X.T @ y
    return X, y, coef_ols


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100)])
@pytest.mark.parametrize("bb", [(1), (2)])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fracridge_ols(nn, pp, bb, fit_intercept):
    X, y, coef_ols = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    coef, alpha = fracridge(X, y, fracs=fracs, fit_intercept=fit_intercept)
    coef = coef[:, -1, ...]
    assert np.allclose(coef, coef_ols, atol=10e-3)
    assert np.all(np.diff(alpha) <= 0)


@pytest.mark.parametrize("frac", [0.1, 0.23, 1])
@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100)])
@pytest.mark.parametrize("bb", [(1), (2)])
def test_fracridge_fracs(frac, nn, pp, bb):
    X, y, coef_ols = make_data(nn, pp, bb)
    # Make sure that you get the fraction you asked for
    coef, _ = fracridge(X, y, fracs=np.array([frac]))
    assert np.all(
        np.abs(
            frac -
            vec_len(coef, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)


check_estimator(FracRidge)

@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100)])
@pytest.mark.parametrize("bb", [(1), (2)])
def test_FracRidge_ols(nn, pp, bb):
    X, y, coef_ols = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    FR = FracRidge(fracs=fracs)
    FR.fit(X, y)
    assert np.allclose(FR.coef_[:, -1, ...], coef_ols, atol=10e-3)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100)])
@pytest.mark.parametrize("bb", [(1), (2)])
@pytest.mark.parametrize("frac", [0.1, 0.23, 1])
def test_FracRidge_fracs(nn, pp, bb, frac):
    X, y, coef_ols = make_data(nn, pp, bb)
    FR = FracRidge(fracs=frac)
    FR.fit(X, y)
    assert np.all(
        np.abs(
            frac -
            vec_len(FR.coef_, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)


@pytest.mark.parametrize("nn, pp", [(1000, 10), (10, 100)])
@pytest.mark.parametrize("bb", [(1), (2)])
def test_FracRidge_predict(nn, pp, bb):
    X, y, coef_ols = make_data(nn, pp, bb)
    fracs = np.arange(.1, 1.1, .1)
    FR = FracRidge(fracs=fracs)
    FR.fit(X, y)
    pred_ols = X @ coef_ols
    pred_fr = FR.predict(X)
    assert np.allclose(pred_fr[:, -1, ...], pred_ols, atol=10e-3)