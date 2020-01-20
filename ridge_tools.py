"""

"""
import numpy as np
from scipy.linalg import svd
from scipy.interpolate import interp1d


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

