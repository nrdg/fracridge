"""

"""
import numpy as np
try:
    from ._linalg import svd
except ImportError:
    from functools import partial
    from scipy.linalg import svd
    svd = partial(svd, full_matrices=False)

from numpy import interp
import warnings

from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted,
                                      _check_sample_weight)

from sklearn.linear_model._base import _preprocess_data, _rescale_data

# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2


__all__ = ["fracridge", "vec_len", "FracRidge"]


def fracridge(X, y, fracs=None, tol=1e-6):
    """
    Approximates alpha parameters to match desired fractions of OLS length.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix for regression, with n number of
        observations and p number of model parameters.

    y : ndarray, shape (n, b)
        Data, with n number of observations and b number of targets.

    fracs : float or 1d array, optional
        The desired fractions of the parameter vector length, relative to
        OLS solution. If 1d array, the shape is (f,).
        Default: np.arange(.1, 1.1, .1)


    Returns
    -------
    coef : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (f, b)
        The alpha coefficients associated with each solution
    Examples
    --------
    """
    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    if not hasattr(fracs , "__len__"):
            fracs = [fracs]
    fracs = np.array(fracs)

    nn, pp = X.shape
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    bb = y.shape[-1]
    ff = fracs.shape[0]

    uu, selt, v_t = svd(X)

    ynew = uu.T @ y
    del uu

    # Solve OLS for the rotated problem and replace y:
    ols_coef = (ynew.T / selt).T
    del ynew

    # Set solutions for small eigenvalues to 0 for all targets:
    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues are being treated as 0")

    ols_coef[isbad, ...] = 0

    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg_sq = sclg**2

    # Prellocate the solution
    if nn >= pp:
        first_dim = pp
    else:
        first_dim = nn

    coef = np.empty((first_dim, ff, bb))
    alphas = np.empty((ff, bb))

    for ii in range(y.shape[-1]):
        newlen = np.sqrt(sclg_sq @ ols_coef[..., ii]**2).T
        newlen = (newlen / newlen[0])
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        targetalphas = np.exp(temp) - 1
        alphas[:, ii] = targetalphas
        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        coef[..., ii] = (sc * ols_coef[..., ii]).T

    coef = np.reshape(v_t.T @ coef.reshape((first_dim, ff * bb)),
                      (pp, ff, bb))

    return coef.squeeze(), alphas


class FracRidge(BaseEstimator, MultiOutputMixin):
    """
    Fraction Ridge estimator

    Parameters
    ----------
    fracs : float or sequence

    """
    def _more_tags(self):
        return {'multioutput': True}

    def __init__(self, fracs=None, fit_intercept=False, normalize=False,
                 copy_X=True):
        self.fracs = fracs
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X,
                                                 dtype=X.dtype)

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight,
            return_mean=True)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y = _rescale_data(X, y, sample_weight)

        self.is_fitted_ = True
        coef, alpha = fracridge(X, y, fracs=self.fracs)
        self.alpha_ = alpha
        self.coef_ = coef
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        pred = np.tensordot(X, self.coef_, axes=(1))
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_
        """
        if self.fit_intercept:
            if len(self.coef_.shape) <= 2:
                self.coef_ = self.coef_ / X_scale[:, np.newaxis]
            else:
                self.coef_ = self.coef_ / X_scale[:, np.newaxis, np.newaxis]
            self.intercept_ = y_offset - np.tensordot(X_offset,
                                                      self.coef_, axes=(1))
        else:
            self.intercept_ = 0.


def vec_len(vec, axis=0):
    return np.sqrt((vec * vec).sum(axis=axis))


def optimize_for_frac(X, fracs):
    """
    Empirically find the alpha that gives frac reduction in vector length of
    the solution

    """
    u, s, v = svd(X)

    val1 = 10e3 * s[0] ** 2  # Huge bias
    val2 = 10e-3 * s[-1] ** 2  # Tiny bias

    alphas = np.concatenate(
        [np.array([0]), 10 ** np.arange(np.floor(np.log10(val2)),
                                        np.ceil(np.log10(val1)), 0.1)])

    results = np.zeros(alphas.shape[0])
    for ii, alpha in enumerate(alphas):
        results[ii] = frac_reduction(X, alpha, s=s)

    return interp1d(results, alphas, bounds_error=False, fill_value="extrapolate")(np.asarray(fracs))


def frac_reduction(X, alpha, s=None):
    """
    Calculates the expected fraction reduction in the length of the
    coefficient vector $\beta$ from OLS to ridge, given a design matrix X and
    a regularization metaparameter alpha.
    """
    if s is None:
        u, s, v = svd(X)
    new = s / (s ** 2 + alpha)
    olslen = np.sqrt(np.sum((1 / s) ** 2))
    rrlen = np.sqrt(np.sum(new ** 2))
    return rrlen / olslen


def frac_reduction_flat(X, alpha, s=None):
    """
    This is the version that assumes a flat eigenvalue spectrum
    """
    if s is None:
        u, s, v = svd(X)
    return np.mean(s ** 2 / (s ** 2 + alpha))


def reg_alpha_flat(X, gamma, s=None):
    """
    This is the version that assumes a flat eigenvalue spectrum
    """
    if s is None:
        u, s, v = svd(X)
    return (s ** 2) * (1 / gamma - 1)
