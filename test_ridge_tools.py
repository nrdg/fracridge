import numpy as np
import ridge_tools
from sklearn.linear_model import LinearRegression


def test_ridgeregressiongamma():
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 500)
    lr = LinearRegression()
    lr.fit(X, y)
    hols = lr.coef_.T  # Transpose for consistent shape with "rotated" solution
    fracs = np.arange(.1, 1.1, .1)
    coef, alphas = ridge_tools.ridgeregressiongamma(X, y, fracs=fracs)
    assert np.abs(1 - (ridge_tools.vec_len(coef[:, 0, -1]) / ridge_tools.vec_len(hols[:, 0])))< 0.05
    assert np.abs(0.1 - (ridge_tools.vec_len(coef[:, 0, 0]) / ridge_tools.vec_len(hols[:, 0]))) < 0.05
