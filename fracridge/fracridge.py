"""

"""
import numpy as np
from numpy import interp
import warnings
import collections

from sklearn.base import BaseEstimator, MultiOutputMixin
from sklearn.utils.validation import (check_X_y, check_array, check_is_fitted,
                                      _check_sample_weight)
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.model_selection import GridSearchCV

# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.2


__all__ = ["fracridge", "vec_len", "FracRidgeRegressor",
           "FracRidgeRegressorCV"]


def _do_svd(X, y, jit=True):
    """
    Helper function to produce SVD outputs
    """
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    # Per default, we'll try to use the jit-compiled SVD, which should be
    # more performant:
    use_scipy = False
    if jit:
        try:
            from ._linalg import svd
        except ImportError:
            warnings.warn("The `jit` key-word argument is set to `True` ",
                          "but numba could not be imported, or just-in time ",
                          "compilation failed. Falling back to ",
                          "`scipy.linalg.svd`")
            use_scipy = True

    # If that doesn't work, or you asked not to, we'll use scipy SVD:
    if not jit or use_scipy:
        from functools import partial
        from scipy.linalg import svd  # noqa
        svd = partial(svd, full_matrices=False)

    if X.shape[0] > X.shape[1]:
        uu, ss, v_t = svd(X.T @ X)
        selt = np.sqrt(ss)
        ynew = np.diag(1./selt) @ v_t @ (X.T @ y)

    else:
        # This rotates the targets by the unitary matrix uu.T:
        uu, selt, v_t = svd(X)
        ynew = uu.T @ y

    ols_coef = (ynew.T / selt).T

    return selt, v_t, ols_coef


def fracridge(X, y, fracs=None, tol=1e-10, jit=True):
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
        OLS solution. If 1d array, the shape is (f,). This input is required
        to be sorted. Otherwise, raises ValueError.
        Default: np.arange(.1, 1.1, .1).

    jit : bool, optional
        Whether to speed up computations by using a just-in-time compiled
        version of core computations. This may not work well with very large
        datasets. Default: True

    Returns
    -------
    coef : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (f, b)
        The alpha coefficients associated with each solution

    Examples
    --------

    Generate random data:

    >>> np.random.seed(0)
    >>> y = np.random.randn(100)
    >>> X = np.random.randn(100, 10)

    Calculate coefficients with naive OLS:

    >>> coef = np.linalg.inv(X.T @ X) @ X.T @ y
    >>> print(np.linalg.norm(coef))  # doctest: +NUMBER
    0.35

    Call fracridge function:

    >>> coef2, alpha = fracridge(X, y, 0.3)
    >>> print(np.linalg.norm(coef2))  # doctest: +NUMBER
    0.10
    >>> print(np.linalg.norm(coef2) / np.linalg.norm(coef))  # doctest: +NUMBER
    0.3

    Calculate coefficients with naive RR:

    >>> alphaI = alpha * np.eye(X.shape[1])
    >>> coef3 = np.linalg.inv(X.T @ X + alphaI) @ X.T @ y
    >>> print(np.linalg.norm(coef2 - coef3))  # doctest: +NUMBER
    0.0
    """
    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    if hasattr(fracs, "__len__"):
        if np.any(np.diff(fracs) < 0):
            raise ValueError("The `frac` inputs to the `fracridge` function ",
                             f"must be sorted. You provided: {fracs}")

    else:
        fracs = [fracs]
    fracs = np.array(fracs)

    nn, pp = X.shape
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    bb = y.shape[-1]
    ff = fracs.shape[0]

    # Calculate the rotation of the data
    selt, v_t, ols_coef = _do_svd(X, y, jit=jit)

    # Set solutions for small eigenvalues to 0 for all targets:
    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues are being treated as 0")

    ols_coef[isbad, ...] = 0

    # Limits on the grid of candidate alphas used for interpolation:
    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    # Generates the grid of candidate alphas used in interpolation:
    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    # The scaling factor applied to coefficients in the rotated space is
    # lambda**2 / (lambda**2 + alpha), where lambda are the singular values
    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg_sq = sclg**2

    # Prellocate the solution:
    if nn >= pp:
        first_dim = pp
    else:
        first_dim = nn

    coef = np.empty((first_dim, ff, bb))
    alphas = np.empty((ff, bb))

    # The main loop is over targets:
    for ii in range(y.shape[-1]):
        # Applies the scaling factors per alpha
        newlen = np.sqrt(sclg_sq @ ols_coef[..., ii]**2).T
        # Normalize to the length of the unregularized solution,
        # because (alphagrid[0] == 0)
        newlen = (newlen / newlen[0])
        # Perform interpolation in a log transformed space (so it behaves
        # nicely), avoiding log of 0.
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        # Undo the log transform from the previous step
        targetalphas = np.exp(temp) - 1
        # Allocate the alphas for this target:
        alphas[:, ii] = targetalphas
        # Calculate the new scaling factor, based on the interpolated alphas:
        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        # Use the scaling factor to calculate coefficients in the rotated
        # space:
        coef[..., ii] = (sc * ols_coef[..., ii]).T

    # After iterating over all targets, we unrotate using the unitary v
    # matrix and reshape to conform to desired output:
    coef = np.reshape(v_t.T @ coef.reshape((first_dim, ff * bb)),
                      (pp, ff, bb))

    return coef.squeeze(), alphas


class FracRidgeRegressor(BaseEstimator, MultiOutputMixin):
    """
    Parameters
    ----------
    fracs : float or sequence
        The desired fractions of the parameter vector length, relative to
        OLS solution.
        Default: np.arange(.1, 1.1, .1)

    fit_intercept : bool, optional
        Whether to fit an intercept term. Default: False.

    normalize : bool, optional
        Whether to normalize the columns of X. Default: False.

    copy_X : bool, optional
        Whether to make a copy of the X matrix before fitting. Default: True.

    tol : float, optional.
        Tolerance under which singular values of the X matrix are considered
        to be zero. Default: 1e-10.

    jit : bool, optional.
        Whether to use jit-accelerated implementation. Default: True.

    Attributes
    ----------
    coef_ : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction. Where p number of model parameters, f number of
        fractions and b number of targets.


    alpha_ : ndarray, shape (f, b)
        The alpha coefficients associated with each solution. Where f number
        of fractions and b number of targets.

    Examples
    --------

    Generate random data:

    >>> np.random.seed(1)
    >>> y = np.random.randn(100)
    >>> X = np.random.randn(100, 10)

    Calculate coefficients with naive OLS:

    >>> coef = np.linalg.inv(X.T @ X) @ X.T @ y

    Initialize the estimator with a single fraction:

    >>> fr = FracRidgeRegressor(fracs=0.3)

    Fit estimator:

    >>> fr.fit(X, y)
    FracRidgeRegressor(fracs=0.3)

    Check results:

    >>> coef_ = fr.coef_
    >>> alpha_ = fr.alpha_
    >>> print(np.linalg.norm(coef_) / np.linalg.norm(coef)) # doctest: +NUMBER
    0.29
    """
    def __init__(self, fracs=None, fit_intercept=False, normalize=False,
                 copy_X=True, tol=1e-10, jit=True):
        self.fracs = fracs
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.tol = tol
        self.jit = jit

    def _validate_input(self, X, y, sample_weight=None):
        """
        Helper function to validate the inputs
        """
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
        return X, y, X_offset, y_offset, X_scale

    def fit(self, X, y, sample_weight=None):
        X, y, X_offset, y_offset, X_scale = self._validate_input(
            X, y, sample_weight=sample_weight)
        coef, alpha = fracridge(X, y, fracs=self.fracs, tol=self.tol,
                                jit=self.jit)
        self.alpha_ = alpha
        self.coef_ = coef
        self._set_intercept(X_offset, y_offset, X_scale)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        if len(self.coef_.shape) == 0:
            pred_coef = self.coef_[np.newaxis]
        else:
            pred_coef = self.coef_
        pred = np.tensordot(X, pred_coef, axes=(1))
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

    def score(self, X, y, sample_weight=None):
        """
        Score the fracridge fit
        """
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        if len(y_pred.shape) > len(y.shape):
            y = y[..., np.newaxis]
        y = np.broadcast_to(y, y_pred.shape)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def _more_tags(self):
        return {'multioutput': True}


class FracRidgeRegressorCV(FracRidgeRegressor):
    """
    Uses :class:`sklearn.model_selection.GridSearchCV` to find the best
    value of `frac` given the data, using cross-validation.

    Parameters
    ----------
    frac_grid : sequence or float, optional
        The values of frac to consider. Default: np.arange(.1, 1.1, .1)

    fit_intercept : bool, optional
        Whether to fit an intercept term. Default: False.

    normalize : bool, optional
        Whether to normalize the columns of X. Default: False.

    copy_X : bool, optional
        Whether to make a copy of the X matrix before fitting. Default: True.

    tol : float, optional.
        Tolerance under which singular values of the X matrix are considered
        to be zero. Default: 1e-10.

    jit : bool, optional.
        Whether to use jit-accelerated implementation. Default: True.

    cv : int, cross-validation generator or an iterable
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  # noqa

    scoring : str, callable, list/tuple or dict, default=None
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html  # noqa

    Attributes
    ----------
    best_frac_ : float
        The best fraction as determined by cross-validation.s

    alpha_ : ndarray, shape (b)
        The alpha coefficients associated with this fraction for each
        target.

    coef_ : ndarray, shape (p, b)
        The coefficients corresponding to the best solution. Where p number
        of parameters and b number of targets.

    Examples
    --------
    Generate random data:

    >>> np.random.seed(1)
    >>> y = np.random.randn(100)
    >>> X = np.random.randn(100, 10)

    Fit model with cross-validation:

    >>> frcv = FracRidgeRegressorCV()
    >>> frcv.fit(X, y)
    FracRidgeRegressorCV(frac_grid=array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
    >>> print(frcv.best_frac_)
    0.1
    """
    def __init__(self, frac_grid=None, fit_intercept=False, normalize=False,
                 copy_X=True, tol=1e-10, jit=True, cv=None, scoring=None):

        self.frac_grid = frac_grid
        if self.frac_grid is None:
            self.frac_grid = np.arange(.1, 1.1, .1)
        super().__init__(self, fit_intercept=False, normalize=False,
                         copy_X=True, tol=tol, jit=True)
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y, sample_weight=None):
        X, y, _, _, _ = self._validate_input(
            X, y, sample_weight=sample_weight)

        parameters = {'fracs': self.frac_grid}
        gs = GridSearchCV(
                FracRidgeRegressor(
                    fit_intercept=self.fit_intercept,
                    normalize=self.normalize,
                    copy_X=self.copy_X,
                    tol=self.tol,
                    jit=self.jit),
                parameters, cv=self.cv, scoring=self.scoring)

        gs.fit(X, y, sample_weight=sample_weight)
        estimator = gs.best_estimator_
        self.best_score_ = gs.best_score_
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.best_frac_ = estimator.fracs
        self.alpha_ = estimator.alpha_
        self.is_fitted_ = True

        return self

    def _more_tags(self):
        return {'multioutput': True}


def vec_len(vec, axis=0):
    return np.sqrt((vec * vec).sum(axis=axis))
