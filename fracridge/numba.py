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

from numpy import ones, eye, float32, float64, \
				sum as __sum, arange as _arange, sign as __sign, uint as _uint, \
				abs as __abs, minimum as _minimum, maximum as _maximum
from numpy.linalg import svd as _svd, pinv as _pinv, eigh as _eigh, \
					cholesky as _cholesky, lstsq as _lstsq, qr as _qr, \
					norm as _norm
from numba import njit, prange
USE_NUMBA = True


__all__ = ['svd', 'pinv', 'eigh', 'cholesky', 'lstsq', 'qr','norm',
			'mean', '_sum', 'sign', 'arange', '_abs', 'minimum', 'maximum',
			'multsum', 'squaresum', '_sign']


@njit(fastmath = True, nogil = True, cache = True)
def svd(X):
	return _svd(X, full_matrices = False)


@njit(fastmath = True, nogil = True, cache = True)
def pinv(X):
	return _pinv(X)


@njit(fastmath = True, nogil = True, cache = True)
def eigh(XTX):
	return _eigh(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def cholesky(XTX):
	return _cholesky(XTX)


@njit(fastmath = True, nogil = True, cache = True)
def lstsq(X, y):
	return _lstsq(X, y.astype(X.dtype))[0]


@njit(fastmath = True, nogil = True, cache = True)
def qr(X):
	return _qr(X)


@njit(fastmath = True, nogil = True, cache = True)
def norm(v, d = 2):
	return _norm(v, d)


@njit(fastmath = True, nogil = True, cache = True)
def _0mean(X, axis = 0):
	return __sum(X, axis)/X.shape[axis]


def mean(X, axis = 0):
	if axis == 0 and X.flags['C_CONTIGUOUS']:
		return _0mean(X)
	else:
		return X.mean(axis)


@njit(fastmath = True, nogil = True, cache = True)
def sign(X):
	return __sign(X)


@njit(fastmath = True, nogil = True, cache = True)
def arange(i):
	return _arange(i)


@njit(fastmath = True, nogil = True, cache = True)
def _sum(X, axis = 0):
	return __sum(X, axis)


@njit(fastmath = True, nogil = True, cache = True)
def _abs(v):
	return __abs(v)


@njit(fastmath = True, nogil = True, cache = True)
def maximum(X, i):
    return _maximum(X, i)


@njit(fastmath = True, nogil = True, cache = True)
def minimum(X, i):
    return _minimum(X, i)


@njit(fastmath = True, nogil = True, cache = True)
def _min(a,b):
    if a < b:
        return a
    return b


@njit(fastmath = True, nogil = True, cache = True)
def _max(a,b):
    if a < b:
        return b
    return a


@njit(fastmath = True, nogil = True, cache = True)
def _sign(x):
    if x < 0:
        return -1
    return 1


@njit(fastmath = True, nogil = True, parallel = True)
def multsum(a,b):
    s = a[0]*b[0]
    for i in prange(1,len(a)):
        s += a[i]*b[i]
    return s


@njit(fastmath = True, nogil = True, parallel = True)
def squaresum(v):
	if len(v.shape) == 1:
	    s = v[0]**2
	    for i in prange(1,len(v)):
	        s += v[i]**2
	# else:

 #    return s


## TEST
print("""Note that first time import of HyperLearn will be slow, """
		"""since NUMBA code has to be compiled to machine code """
		"""for optimization purposes.""")

y32 = ones(2, dtype = float32)
y64 = ones(2, dtype = float64)


X = eye(2, dtype = float32)
A = svd(X)

X = eye(2, dtype = float64)
A = svd(X)

A = None
X = None
y32 = None
y64 = None
