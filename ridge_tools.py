"""

"""
import numpy as np
from scipy.linalg import svd
from numpy import interp
import warnings


# Module-wide constants
BIG_BIAS = 10e3
SMALL_BIAS = 10e-3
BIAS_STEP = 0.25


def ridgeregressiongamma(X, y, fracs=None, tol=1e-6):
    """
    Approximates alpha parameters to match desired fractions of OLS length.

    Parameters
    ----------
    X : ndarray, shape (n, p)
        Design matrix for regression, with n number of
        observations and p number of model parameters.

    y : ndarray, shape (n, b)
        Data, with n number of observations and b number of simultaneous
        measurement units (e.g., channels).

    fracs : float or 1d array
        The desired fractions of the parameter vector length, relative to
        OLS solution. If 1d array, the shape is (f,)

    Returns
    -------
    coef : ndarray, shape (p, f, b)
        The full estimated parameters across units of measurement for every
        desired fraction.
    alphas : ndarray, shape (f,)

    Examples
    --------
    """
    n, p = X.shape
    b = y.shape[-1]
    if hasattr(fracs, "__len__"):
        fracs = np.asanyarray(fracs)
        f = fracs.shape[0]
    else:
        f = 1

    # This is the expensive step:
    uu, selt, vv = svd(X, full_matrices=False)

    isbad = selt < tol
    if np.any(isbad):
        warnings.warn("Some eigenvalues of X are small" +
                      " and being treated as zero.")

    # Rotate the data:
    ynew = uu.T @ y

    # Solve OLS for the rotated problem:
    hols_new = (ynew.T / selt).T

    # Set solutions for small eigenvalues to 0 for all b:
    hols_new[isbad, :] = 0

    val1 = BIG_BIAS * selt[0] ** 2
    val2 = SMALL_BIAS * selt[-1] ** 2

    alphagrid = np.concatenate(
        [np.array([0]),
         10 ** np.arange(np.floor(np.log10(val2)),
                         np.ceil(np.log10(val1)), BIAS_STEP)])

    seltsq = selt**2
    sclg = seltsq / (seltsq + alphagrid[:, None])
    sclg[:, isbad] = 0

    # Prellocate the solution
    coef = np.empty((p, b, f))
    alphas = np.empty((f, b))
    
    for vx in range(y.shape[-1]):
        newlen = (sclg @ ynew[:,vx]**2).T
        newlen = (newlen / newlen[0])
        temp = interp(fracs, newlen[::-1], np.log(1 + alphagrid)[::-1])
        # temp = interp1d(newlen, np.log(1 + alphagrid), bounds_error=False, fill_value="extrapolate")(fracs)
        targetalphas = np.exp(temp) - 1
        for p in range(len(targetalphas)):
            sc = seltsq / (seltsq + targetalphas[p])
            coef[:, vx, p] = vv.T @ (sc * hols_new[:, vx])
    return coef, alphas


def vec_len(vec):
    return np.sqrt(np.sum(vec ** 2))


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
