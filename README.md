# fracridge

[![DOI](https://zenodo.org/badge/261540866.svg)](https://zenodo.org/badge/latestdoi/261540866)

Is an implementation of fractional ridge regression (FRR).

## Installation:

### MATLAB

Download and copy the files from the
[https://github.com/nrdg/fracridge/tree/master/matlab](MATLAB directory) into
your MATLAB path.

### Python

To install the release version:

    pip install fracridge

Or to install the development version:

    pip install -r requirements.txt
    pip install .

## Usage

### MATLAB

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

## Online documentation

[https://nrdg.github.io/fracridge/](https://nrdg.github.io/fracridge/)

## How to cite

Please cite our preprint: "Fractional ridge regression: a fast, interpretable
reparameterization of ridge regression" (2020), Rokem & Kay,
https://arxiv.org/abs/2005.03220