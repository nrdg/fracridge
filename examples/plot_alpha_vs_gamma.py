"""

========================================
Comparing alpha and fracs
========================================

Here we compare parameterization using fractional ridge regression (FRR) and
standard ridge regression (SRR).

We will use the cross-validation objects implemented for both of these methods.
In the case of SRR, we will use the Scikit Learn implementation in the
:class:`sklearn.linear_model.RidgeCV` object. For FRR, we use the
:class:`FracRidgeRegressorCV` object, which implements a similar API.
"""

##########################################################################
# Imports:
#

import numpy as np
from numpy.linalg import norm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV, LinearRegression
from fracridge import FracRidgeRegressorCV

##########################################################################
# Here, we use a synthetic dataset. We generate a regression dataset
# with multiple targets, multiple samples, a large number of features
# and plenty of redundancy between them (set through the
# relatively small `effective_rank` of the design matrix):
#

np.random.seed(1984)

n_targets = 15
n_features = 80
effective_rank = 20
X, y, coef_true = make_regression(
                    n_samples=250,
                    n_features=n_features,
                    effective_rank=effective_rank,
                    n_targets=n_targets,
                    coef=True,
                    noise=5)

##########################################################################
# To evaluate and compare the performance of the two algorithms, we
# split the data into test and train sets:

X_train, X_test, y_train, y_test = train_test_split(X, y)

##########################################################################
# We will start with SRR. We use a dense grid of alphas with 20
# log-spaced values -- a common heuristic used to ensure a wide sampling
# of alpha values

n_alphas = 20
srr_alphas = np.logspace(-10, 10, n_alphas)
srr = RidgeCV(alphas=srr_alphas)
srr.fit(X_train, y_train)

##########################################################################
# We sample the same number of fractions for FRR, evenly distributed between
# 1/n_alphas and 1.
#

fracs = np.linspace(1/n_alphas, 1 + 1/n_alphas, n_alphas)
frr = FracRidgeRegressorCV(frac_grid=fracs)
frr.fit(X_train, y_train)

##########################################################################
# Both models are fit and used to predict a left out set. Performance
# of the models is compared using the :func:`sklearn.metrics.r2_score`
# function (coefficient of determination).

pred_frr = frr.predict(X_test)
pred_srr = srr.predict(X_test)

frr_r2 = r2_score(y_test, pred_frr)
srr_r2 = r2_score(y_test, pred_srr)

print(frr_r2)
print(srr_r2)

##########################################################################
# In addition to a direct comparison of performance, we might ask what are the
# differences in terms of how the models have reached this point.
# The FRR CV estimator has a property that tells us what has been discovered
# as the best fraction (or 'gamma') to use:

print(frr.best_frac_)

##########################################################################
# We can also ask what `alpha` value was deemed best. For the
# multi-target case presented here, this will be a vector of values,
# one for each target:

print(frr.alpha_)

##########################################################################
# In contrast, the SRR estimator has just one value of `alpha`:

print(srr.alpha_)

##########################################################################
# But this one value causes many different changes in the coefficient
#

lr = LinearRegression()
frr.fit(X, y)
srr.fit(X, y)
lr.fit(X, y)

print(norm(frr.coef_, axis=0)  / norm(lr.coef_, axis=-1))
print(norm(srr.coef_, axis=-1)  / norm(lr.coef_, axis=-1))

print(srr.best_score_)
print(frr.best_score_)