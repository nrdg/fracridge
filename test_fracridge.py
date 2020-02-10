import numpy as np
from fracridge import fracridge, vec_len
from sklearn.linear_model import LinearRegression


def test_ridgeregressiongamma():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 2)
    lr = LinearRegression()
    lr.fit(X, y)
    coef_ols = lr.coef_.T
    fracs = np.arange(.1, 1.1, .1)
    coef, alphas = fracridge(X, y, fracs=fracs)
    assert np.allclose(coef[:, -1, :], coef_ols, atol=10e-3)

    for frac in [0.1, 0.23, 1]:
        coef, alphas = fracridge(X, y, fracs=np.array([frac]))
        assert np.all(
            np.abs(
            frac - vec_len(coef, axis=0) / vec_len(coef_ols, axis=0)) < 0.01)

