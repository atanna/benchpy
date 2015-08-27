# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles

from .utils import cached_property

Regression = namedtuple("Regression", 'stat_w stat_y r2')


class StatMixin(object):
    def __init__(self, full_time, batch_sizes, with_gc,
                 gc_time=None, func_name="",
                 confidence=0.95):
        self.full_time = full_time
        self.n_samples, self.n_batches = full_time.shape
        self.batch_sizes = batch_sizes
        self.with_gc = with_gc
        self.name = func_name
        self.init_features(full_time, gc_time)
        self.confidence = confidence
        self.stat_time = None
        self.regr = None

    def init_features(self, full_time, gc_time, alpha=0.5,
                      min_used_samples=10):
        y = full_time
        self.n_used_samples = self.n_samples
        if self.n_samples > min_used_samples:
            # Reduce number of used samples to
            # max(min_used_samples, $\alpha$*n_samples).
            # choose best time samples
            self.n_used_samples = \
                max(min_used_samples, (int(alpha*self.n_samples)))
            order = full_time.argsort(axis=0)

            ind = (order, range(full_time.shape[1]))
            self.n_used_samples = max(min_used_samples,
                                      (int(alpha*self.n_samples)))
            y = full_time[ind][:self.n_used_samples]
            if gc_time is not None:
                gc_time = gc_time[ind][:self.n_used_samples]
        self.gc_time = np.mean(np.mean(gc_time, axis=0)
                               / self.batch_sizes)

        self.feature_names = np.array(["batch", "const"])
        self.n = len(self.feature_names)

        X_y = np.empty((self.n_used_samples, self.n_batches, self.n + 1))
        X_y[:, :, 0] = self.batch_sizes
        X_y[:, :, 1] = 1
        X_y[:, :, 2] = y

        self.X_y = X_y
        self.X = X_y[:, :, :-1]
        self.y = y

    @cached_property
    def time(self):
        self.regr = self.regression(confidence=self.confidence)
        self.stat_time = self.regr.stat_y
        return self.stat_time.mean

    @cached_property
    def x_y(self):
        # FIXME: we never use y here. Also, a better name?
        assert self.batch_sizes[0] == 1
        return self.X_y[:, 0, :].mean(axis=0)

    def get_stat_table(self):
        mean_time = self.y / self.batch_sizes[:, np.newaxis].T
        return dict(Name=self.name,
                    Time=self.time,
                    CI=np.maximum(self.stat_time.ci, 0),
                    Std=self.stat_time.std,
                    Min=mean_time.min(), Max=mean_time.max(),
                    R2=self.regr.r2,
                    Features_time=self.x_y[:-1] * self.regr.stat_w.mean,
                    gc_time=self.gc_time,
                    Time_without_gc=self.time - self.gc_time,
                    fit_info=dict(with_gc=self.with_gc,
                                  samples=self.n_samples,
                                  batches=self.batch_sizes))

    def info_to_plot(self):
        return self.X.mean(axis=0), self.y.mean(axis=0), self.regr.stat_w

    def regression(self, B=1000, **kwargs):
        n_samples = len(self.X_y)
        indices = np.random.randint(0, n_samples, size=(B, self.n_batches))
        # bootstrap
        resamples = self.X_y[indices, np.arange(self.n_batches)]
        arr_st_w = np.array([ridge_regression(resample)
                             for resample in resamples])
        stat_w = get_mean_stat(arr_st_w, **kwargs)

        self.arr_X_y = resamples
        self.arr_st_w = arr_st_w
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)

        w = stat_w.mean
        w_r2 = np.array([r2(self.y, X.dot(w))
                         for X in self.X]).mean()
        return Regression(stat_w, stat_y, w_r2)


def ridge_regression(X_y, alpha=0.15):
    r"""Fits an L2-penalized linear regression to the data.

    The ridge coefficients are guaranteed to be non-negative and minimize

    .. math::

       \min\limits_w ||X w - y||_2 + \alpha ||w||_2

    Parameters
    ----------
    Xy : (N, M + 1) array_like
        Observation matrix. The first M columns are observations. The
        last column corresponds to the target values.
    alpha : float
        Penalization strength. Larger values make the solution more robust
        to collinearity.

    Returns
    -------
    w : (M, ) ndarray
        Non-negative ridge coefficients.
    """
    X_y = np.atleast_2d(X_y)
    X, y = X_y[:, :-1], X_y[:, -1]

    M = X.shape[1]
    X_new = np.append(X, alpha * np.eye(M), axis=0)
    y_new = np.append(y, np.zeros(M))
    w, _residuals = nnls(X_new, y_new)
    return w


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std else np.inf


Stat = namedtuple("Stat", "mean std ci")


def get_mean_stat(values, confidence=0.95):
    alpha = (1 - confidence) / 2
    lowhigh = mquantiles(values, prob=[alpha, 1-alpha], axis=0)
    return Stat(np.mean(values, axis=0),
                np.std(values, ddof=1, axis=0),
                np.asarray(lowhigh))
