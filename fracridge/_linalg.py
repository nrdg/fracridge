
"""
This code is taken from the hyperlearn library: https://github.com/danielhanchen/hyperlearn/

And distributed together with the license therein:


BSD 3-Clause License

Copyright (c) 2020, Daniel Han-Chen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from . import _numba as numba
from ._numba import njit, USE_NUMBA, sign, arange
import numpy as np
from scipy.linalg import lapack as get_lapack_funcs


def svd_flip(U, VT, U_decision=True):
    """
    Flips the signs of U and VT for SVD in order to force deterministic output.
    Follows Sklearn convention by looking at U's maximum in columns
    as default.
    """
    if U_decision:
        max_abs_cols = abs(U).argmax(0)
        signs = sign(U[max_abs_cols, arange(U.shape[1])])
    else:
        # rows of v, columns of u
        max_abs_rows = abs(VT).argmax(1)
        signs = sign( VT[arange(VT.shape[0]), max_abs_rows])

    U *= signs
    VT *= signs[:, np.newaxis]
    return U, VT



def svd(X, fast=True, U_decision=False, transpose=True):
    """
    [Edited 9/11/2018 --> Modern Big Data Algorithms p/n ratio check]
    Computes the Singular Value Decomposition of any matrix.
    So, X = U * S @ VT. Note will compute svd(X.T) if p > n.
    Should be 99% same result. This means this implementation's
    time complexity is O[ min(np^2, n^2p) ]

    Speed
    --------------
    If USE_GPU:
        Uses PyTorch's SVD. PyTorch uses (for now) a NON divide-n-conquer algo.
        Submitted report to PyTorch:
        https://github.com/pytorch/pytorch/issues/11174
    If CPU:
        Uses Numpy's Fortran C based SVD.
        If NUMBA is not installed, uses divide-n-conqeur LAPACK functions.
    If Transpose:
        Will compute if possible svd(X.T) instead of svd(X) if p > n.
        Default setting is TRUE to maintain speed.

    Stability
    --------------
    SVD_Flip is used for deterministic output. Does NOT follow Sklearn convention.
    This flips the signs of U and VT, using VT_based decision.
    """
    transpose = True if (transpose and X.shape[1] > X.shape[0]) else False
    if transpose:
        X, U_decision = X.T, not U_decision

    n, p = X.shape
    ratio = p/n
    #### TO DO: If memory usage exceeds LWORK, use GESVD
    if ratio >= 0.001:
        if USE_NUMBA:
            U, S, VT = numba.svd(X)
        else:
            #### TO DO: If memory usage exceeds LWORK, use GESVD
            U, S, VT, __ = get_lapack_funcs("gesdd")(X, full_matrices=False)
    else:
        U, S, VT, __ = get_lapack_funcs("gesvd")(X, full_matrices=False)

    U, VT = svd_flip(U, VT, U_decision=U_decision)

    if transpose:
        return VT.T, S, U.T
    return U, S, VT


