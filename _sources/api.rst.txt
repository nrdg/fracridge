####################
``fracridge`` API
####################

MATLAB
======

The MATLAB version of the software provides the function ``fracridge``::

    function [coef,alphas,offset] = fracridge(X,fracs,y,tol,mode,standardizemode)

Parameters
----------

X : The design matrix (d x p) with different data points in the rows and
    different regressors in the columns.

fracs : a vector (1 x f) of one or more fractions between 0 and 1.
        Fractions can be exactly 0 or exactly 1. However, values in between
        0 and 1 should be no less than 0.001 and no greater than 0.999.
        For example, ``fracs`` could be ``0:.05:1`` or ``0:.1:1``.

y : The data (d x t)
    One or more target variables in the columns.

tol : (optional) is a tolerance value such that eigenvalues
      below the tolerance are treated as 0. Default: 1e-6.

mode : (optional) can be:

  0 means the default behavior, ``fracs`` is interpreted as a series of
  requested fractions.

  1 means to interpret ``fracs`` as exact alpha values that are desired.
  This case involves using the rotation trick to speed up the ridge
  regression but does not involve the fraction-based tailoring method.
  In the case that ``mode`` is 1, the output ``alphas`` is returned as [].

  Default: 0.

standardizemode : (optional) can be:

  0 means the default behavior (do not modify any of the regressors). In this case,
  we do not add an offset term to the model. Note that the user may choose to include
  an constant regressor in ``X`` if so desired.

  1 means to add an offset term to the model. In this case, the offset term is fully
  estimated using ordinary least squares, and ridge regression is applied to
  de-meaned data and de-meaned regressors. (The user should not include a constant
  regressor in ``X`` if using this mode.)

  2 means to standardize the regressors before performing ridge regression. In this case,
  an offset term is added to the model, and is fully estimated using ordinary least
  squares. Ridge regression is applied to de-meaned data and standardized
  regressors. The returned regression weights will refer to the original regressors
  (i.e., they will be adjusted for the effects of standardization). This mode may be
  preferred for most applications given that the effects of regularization are
  influenced by the scale of the regressors. (The user should not include a constant
  regressor in ``X`` if using this mode.)

  Default: 0.

Returns:
--------

 coef : as the estimated regression weights (p x f x t)
   for all fractional levels for all target variables.

 alphas : as the alpha values (f x t) that correspond to the
   requested fractions. Note that alpha values can be Inf
   in the case that the requested fraction is 0, and can
   be 0 in the case that the requested fraction is 1.
 offset : as the offset term for each target variable (f x t).
   Note that when <standardizemode> is 0, no offset term is added,
   and <offset> is returned as all zeros.

Notes
------

- We silently ignore regressors that are all zeros.

- The class of ``coef`` and ``alphas`` is matched to the class of ``X``.

- All values in ``X`` and ``y`` should be finite. (A check for this is performed.)

- If a given target variable consists of all zeros, this is
  a degenerate case, and we will return regression weights that
  are all zeros and alpha values that are all zeros.

Examples
--------

Example 1 (Demonstrate that fracridge achieves the correct fractional length)


.. code-block:: matlab

    y = randn(1000,1);
    X = randn(1000,10);
    coef = inv(X'*X)*X'*y;
    [coef2,alpha] = fracridge(X,0.3,y);
    coef3 = inv(X'*X + alpha*eye(size(X,2)))*X'*y;
    coef4 = fracridge(X,alpha,y,[],1);
    norm(coef)
    norm(coef2)
    norm(coef2) ./ norm(coef)
    norm(coef2-coef3)
    norm(coef4-coef3)

Example 2 (Compare execution time between naive ridge regression and fracridge)

.. code-block:: matlab

    y = randn(1000,300);        % 1000 data points x 300 target variables
    X = randn(1000,3000);       % 1000 data points x 3000 predictors
    % naive approach
    tic;
    alphas = 10.^(-4:.5:5.5);  % guess 20 alphas
    cache1 = X'*X;
    cache2 = X'*y;
    coef = zeros(3000,length(alphas),300);
    for j=1:length(alphas)
        coef(:,j,:) = permute(inv(cache1 + alphas(j)*eye(size(X,2)))*cache2,[1 3 2]);
    end
    toc;
    % fracridge approach
    tic;
    fracs = .05:.05:1;         % get 20 equally-spaced lengths
    coef2 = fracridge(X,fracs,y);
    toc;
    % fracridge implementation of simple rotation
    tic;
    coef3 = fracridge(X,alphas,y,[],1);
    toc;
    assert(all(abs(coef(:)-coef3(:))<1e-4));

Example 3 (Plot coefficient paths and vector length for a simple example)

.. code-block:: matlab

    y = randn(100,1);
    X = randn(100,6)*(1+rand(6,6));
    fracs = .05:.05:1;
    [coef,alphas] = fracridge(X,fracs,y);
    figure;
    subplot(1,2,1); hold on;
    plot(fracs,coef');
    xlabel('Fraction');
    title('Trace plot of coefficients');
    subplot(1,2,2); hold on;
    plot(fracs,sqrt(sum(coef.^2,1)),'ro-');
    xlabel('Fraction');
    ylabel('Vector length');

Example 4 (Demonstrate how fracridge handles standardization of regressors)

.. code-block:: matlab

    X = 20 + randn(50,2);
    y = X*rand(2,1) + randn(50,1);
    fracs = 0.1:0.1:1;
    [coef,alphas,offset] = fracridge(X,fracs,y,[],[],2);
    modelfit = X*coef + repmat(offset',[50 1]);
    figure; hold on;
    cmap0 = copper(length(fracs));
    h = [];
    legendstr = {};
    for p=1:length(fracs)
        h(p) = plot(modelfit(:,p),'-','Color',cmap0(p,:));
        legendstr{p} = sprintf('Frac %.1f',fracs(p));
    end
    hdata = plot(y,'k-','LineWidth',2);
    legend([hdata h],['Data' legendstr],'Location','EastOutside');


Python
======

The ``fracridge`` API includes a functional interface and an object-oriented
interface. The object-oriented interface is consistent with the
`Scikit Learn <https://scikit-learn.org/stable/>`_ API, providing an
estimator that can be integrated into pipelines that use scikit learn.

fracridge
---------

This is the functional API for the software.


.. autosummary::
   :toctree: generated/
   :template: function.rst

   fracridge.fracridge

FracRidgeRegressor
------------------

This is the object-oriented interface for the software.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fracridge.FracRidgeRegressor



FracRidgeRegressorCV
--------------------

This object uses :class:`sklearn.model_selection.GridSearchCV` to find the
best value of ``frac`` for provided data using cross-validation.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fracridge.FracRidgeRegressorCV
