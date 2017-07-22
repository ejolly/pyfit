from __future__ import division

'''
pyfit loss functions
=======================
Covenience wrappers for loss functions

'''

__all__ ['SSE','LL']

import numpy as np
from scipy.stats import norm

def SSE(y, yhat):
    """ Sum of squared error loss. """

    return np.sum(np.power(yhat - y, 2))

def LL(y, yhat):
    """ Negative log liklihood. SD comes from data."""

    return -np.sum(norm.logpdf(y, loc=yhat, scale=np.std(y)))
