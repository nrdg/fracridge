import numpy as np
from fracridge import fracridge, vec_len
from sklearn.linear_model import LinearRegression
import pytest


def make_data(nn, pp, bb):
    np.random.seed(1)
    X = np.random.randn(nn, pp)
    y = np.random.randn(nn, bb)
    lr = LinearRegression()
    lr.fit(X, y)
    coef_ols = lr.coef_.T
    return X, y, coef_ols


@pytest.mark.parametrize("nn,pp,bb", [(1000, 10, 2), (10, 100, 2)])
def test_fracridge_ols(nn, pp, bb):
    X, y, coef_ols = make_data(1000, 10, 2)
    fracs = np.arange(.1, 1.1, .1)
    coef, _ = fracridge(X, y, fracs=fracs)
    # Make sure that in the absence of regularization, we get
    # the same result as ols:
    assert np.allclose(coef[:, -1, :], coef_ols, atol=10e-3)


@pytest.mark.parametrize("frac", [0.1, 0.23, 1])
@pytest.mark.parametrize("nn,pp,bb", [(1000, 10, 2), (10, 100, 2)])
def test_fracridge_fracs(frac,nn,pp,bb):
    X, y, coef_ols = make_data(1000, 10, 2)
    # Make sure that you get the fraction you asked for
    coef, _ = fracridge(X, y, fracs=np.array([frac]))
    assert np.all(
        np.abs(
            frac -
            vec_len(coef, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)
