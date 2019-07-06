# function lambda = calcridgeregressionlambda(X,frac)

# % function lambda = calcridgeregressionlambda(X,frac)
# %
# % <X> is a design matrix
# % <frac> is a fraction between 0 and 1 (inclusive).
# %   can also be a vector of fractions.
# %
# % return the lambda value that approximately achieves
# % vector-length-reduction of the ordinary-least-squares (OLS)
# % solution at the level given by <frac>.
# %
# % we use some hard-coded constants in the function for interpolation
# % purposes. because of this, some very extreme values for
# % <frac> (such as 0) will generate NaN as the lambda.
# %
# % note that we silently ignore regressors that are all zeros.
# %
# % example:
# % y = randn(100,1);
# % X = randn(100,10);
# % h = inv(X'*X)*X'*y;
# % lambda = calcridgeregressionlambda(X,0.3);
# % h2 = inv(X'*X + lambda*eye(size(X,2)))*X'*y;
# % vectorlength(h)
# % vectorlength(h2)
# % vectorlength(h2) ./ vectorlength(h)

# % ignore bad regressors (those that are all zeros)
# bad = all(X==0,1);
# good = ~bad;
# X = X(:,good);

# % decompose X
# [u,s,v] = svd(X,0);  % u is 100 x 80, s is 80 x 80, v is 80 x 80

# % extract the diagonal (the eigenvalues of s)
# selt = diag(s);  % 80 x 1

# % first, we need to find a grid of lambdas that will span a reasonable range
# % with reasonable level of granularity
# val1 = 10^3 *selt(1)^2;    % huge bias (take the biggest eigenvalue down to ~.001 of what OLS would be)
# val2 = 10^-3*selt(end)^2;  % tiny bias (just add a small amount)
# lambdas = [0 10.^(floor(log10(val2)):0.1:ceil(log10(val1)))];  % no bias to tiny bias to huge bias

# % next, we need to estimate how much the vector-length reduction will be for each lambda value.
# yval = [];
# for qq=1:length(lambdas)
#   ref = selt ./ (selt.^2);
#   new = selt ./ (selt.^2 + lambdas(qq));     % the regularized result
#   fracreduction = new./ref;                  % what fraction is the regularized result?
#   reflen = sqrt(length(selt));               % vector length assuming variance 1 for each dimension
#   newlen = sqrt(sum(fracreduction.^2));      % estimated vector length
#   yval(qq) = newlen./reflen;                 % vector-length reduction
# %   olslen = sqrt(sum((1./selt).^2));
# %   rrlen  = sqrt(sum(new.^2));
# %   yval(qq) = rrlen ./ olslen;
# end
# % in general, we should save the eigenvalues (selt) and the lambdas chosen for evaluation (lambdas)!

# % finally, use cubic interpolation to find the lambda that achieves the desired level
# lambda = interp1(yval,lambdas,frac,'pchip',NaN);


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

    return interp1d(results, alphas)(np.asarray(fracs))


def frac_reduction(X, alpha, s=None):
    """
    Calculates the expected fraction reduction in the length of the
    coefficient vector $\beta$ from OLS to ridge, given a design matrix X and
    a regularization metaparameter alpha.
    """
    if s is None:
        u, s, v = svd(X)

    ols_betas = 1 / s
    rr_betas = s / (s ** 2 + alpha)

    ref_len = vec_len(ols_betas)
    new_len = vec_len(rr_betas)

    # Below is based on Kendrick's original code (I think):
    # ref_len = np.sqrt(s.shape[0])
    # new_len = np.sqrt(np.sum((rr_betas/ols_betas)**2))

    return new_len / ref_len


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

