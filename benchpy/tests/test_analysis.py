# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_allclose

from benchpy.analysis import ridge_regression


def test_ridge_regression():
    X = np.random.uniform(size=(100, 3))
    w = np.random.uniform(size=3)
    y = np.dot(X, w)
    Xy = np.c_[X, y]

    w_hat = ridge_regression(Xy)
    assert_allclose(w, w_hat, atol=1e-2)
