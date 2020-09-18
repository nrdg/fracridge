import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut

from fracridge import FracRidgeRegressor

np.random.seed(20)

n_targets = 15
X, y, coef_true = make_regression(
                    n_samples=100,
                    n_features=10,
                    n_informative=3,
                    n_targets=n_targets,
                    coef=True,
                    noise=1)


X_train, X_test, y_train, y_test = train_test_split(X, y)

n_alphas = 20

srr_alphas = np.logspace(-10, 10, n_alphas)
srr = RidgeCV(alphas=srr_alphas, store_cv_values=True, scoring="r2")
srr.fit(X, y)
srr_cv_values = srr.cv_values_

fracs = np.linspace(0, 1, n_alphas)
frr = FracRidgeRegressor(fracs=fracs)
loo = LeaveOneOut()
frr_cv_values = np.zeros((X.shape[0], n_targets, n_alphas))
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    frr.fit(X_train, y_train)
    y_test_hat = frr.predict(X_test)
    frr_cv_values[test_index] = y_test_hat.squeeze().T

err_srr = (y[..., np.newaxis] - srr_cv_values) ** 2
err_frr = (y[..., np.newaxis] - frr_cv_values) ** 2