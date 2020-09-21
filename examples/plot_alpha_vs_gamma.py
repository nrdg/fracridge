"""

========================================
Comparing alpha and fracs
========================================

Here we compare parameterization using FRR and standard ridge regression.

"""


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from fracridge import FracRidgeRegressorCV

np.random.seed(20)

n_targets = 15
X, y, coef_true = make_regression(
                    n_samples=100,
                    n_features=40,
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
frr = FracRidgeRegressorCV(frac_grid=fracs)
frr.fit(X, y)
