# fracridge

Is an implementation of fractional ridge regression (FRR).

## Installation:

### Matlab

Download and copy the files from the [https://github.com/nrdg/fracridge/tree/master/matlab](Matlab directory) into your
Matlab path.

### Python

To install the release version:

    pip install fracridge

Or to install the development version:

    pip install -r requirements.txt
    pip install .

## Usage

### Matlab

    [coef,alphas] = fracridge(X,fracs,y,tol,mode)


### Python

There's a functional API:

    from fracridge import fracridge
    coefs, alphas = fracridge(X, y, fracs)

Or a sklearn-compatible OO API:

    from fracridge import FracRidge
    fr = FracRridge(fracs=fracs)
    fr.fit(X, y)
    coefs = fr.coef_
    alphas = fr.alpha_

## How to cite

"Fractional ridge regression: a fast, interpretable reparameterization of ridge regression", Rokem & Kay (in preparation)
