"""

"""
import numpy as np
from scipy.linalg import svd
import numpy.linalg as nla
from numpy.core.multiarray import interp
from scipy.interpolate import interp1d
import warnings


# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.25


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

    fracs : 1d array, optional
        The desired fractions of the parameter vector length, relative to
        OLS solution. If 1d array, the shape is (f,).
        Default: np.arange(.1, 1.1, .1)


    Returns
    -------
    coef : ndarray, shape (pp, ff, bb)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (ff, bb)
        The alpha coefficients associated with each solution
    Examples
    --------
    """
    if fracs is None:
        fracs = np.arange(.1, 1.1, .1)

    nn, pp = X.shape
    bb = y.shape[-1]
    ff = fracs.shape[0]

    # This is the expensive step:
    uu, selt, vv = svd(X, full_matrices=False)
    # selt = selt / selt[0]

    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues of X are small" +
                      " and being treated as zero.")
        selt[isbad] = 0
    # Rotate the data:
    ynew = uu.T @ y

    # Solve OLS for the rotated problem and replace y:
    ols_coef = (ynew.T / selt).T

    # Set solutions for small eigenvalues to 0 for all targets:
    ols_coef[isbad, :] = 0

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
    coef1 = np.empty((pp, ff, bb))
    coef2 = np.empty((pp, ff, bb))
    alphas = np.empty((ff, bb))

    for ii in range(y.shape[-1]):
        newlen = np.sqrt(sclg_sq @ ols_coef[:, ii]**2).T
        newlen = (newlen / newlen[0])
        # Alternative fast interpolation
        # temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        temp = interp1d(newlen,
                        np.log(1 + alphagrid),
                        bounds_error=False,
                        fill_value="extrapolate",
                        kind='linear')(fracs)
        targetalphas = np.exp(temp) - 1

        sc = seltsq / (seltsq + targetalphas[np.newaxis].T)
        coef1[:, :, ii] = (sc * ols_coef[:, ii]).T

        for p in range(len(targetalphas)):
            sc = seltsq / (seltsq + targetalphas[p])
            coef2[:, p, ii] = sc * ols_coef[:, ii]

    assert np.allclose(coef1, coef2)
    coef = vv.T @ coef2.reshape((pp, ff * bb))
    coef = coef.reshape((pp, ff, bb))
    return coef, alphas


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
