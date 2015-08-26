# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles

from .utils import cached_property

Regression = namedtuple("Regression", 'stat_w stat_y r2')


class Features(object):
    def __init__(self, feature_names, features, y):
        self.feature_names = np.array(feature_names)
        self.y = y
        self.n = len(self.feature_names)
        _shape = y.shape + (1,)
        # _shape = (n_samples, n_batches, 1)  - shape of one of the
        # feature_columns in full fitting matrix X
        # with shape (n_samples, n_batches, n_features)
        # note: n_samples is number of used samples
        # (it can be less then what has been measured)
        # The last feature_column in matrix X_y is y (~time)
        # X_y.shape = (n_samples, n_batches, n_features+1)
        # if feature has less dimension (f.e. `const`) then y has,
        # we use extension of it.
        self.X_y = np.concatenate(
            [np.array([feature] *
                      np.prod(y.shape[:y.ndim-np.array(feature).ndim]))
            .reshape(_shape) for feature in features] +
            [y.reshape(_shape)],
            axis=2)
        self.X = self.X_y[:, :, :-1]


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

    def init_features(self, full_time, gc_time=None, alpha=0.5,
                      min_used_samples=10):
        y = full_time
        if self.n_samples > min_used_samples:
            # Reduce number of used samples to
            # max(min_used_samples, $\alpha$*n_samples).
            # choose best time samples
            self.n_used_samples = \
                max(min_used_samples, (int(alpha*self.n_samples)))
            ind = (full_time.argsort(axis=0),
                   range(full_time.shape[1]),
                   slice(self.n_used_samples))
            y = full_time[ind]
            gc_time = gc_time[ind]
        self.gc_time = np.mean(np.mean(gc_time, axis=0)
                               / self.batch_sizes)
        self.features = Features(["batch", "const"],
                                 [self.batch_sizes, 1.],
                                 y=y)

    @cached_property
    def time(self):
        try:
            self.regr = self.regression(confidence=self.confidence)
            self.stat_time = self.regr.stat_y
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                get_mean_stat(self._av_time, self.confidence)
        return self.stat_time.mean

    @property
    def ci(self):
        if self.stat_time is None:
            self.time()
        _ci = self.stat_time.ci
        _ci[0] = max(_ci[0], 0.)
        return _ci

    @property
    def std(self):
        if self.stat_time is None:
            self.time()
        return self.stat_time.std

    @cached_property
    def min(self):
        return np.min(self._av_time)

    @cached_property
    def max(self):
        return np.max(self.full_time / self.batch_sizes[:, np.newaxis].T)

    @cached_property
    def _av_time(self):
        return self.features.y / self.batch_sizes[:, np.newaxis].T

    @property
    def stat_w(self):
        if self.regr is None:
            self.time()
        return self.regr.stat_w

    @cached_property
    def X(self):
        return self.features.X.mean(axis=0)

    @cached_property
    def y(self):
        return self.features.y.mean(axis=0)

    @cached_property
    def features_time(self):
        return self.x_y[:-1] * self.regr.stat_w.mean

    @property
    def predicted_time_witout_gc(self):
        return self.time - self.gc_time

    @property
    def r2(self):
        if self.regr is None:
            self.time()
        return self.regr.r2

    @cached_property
    def fit_info(self):
        return dict(with_gc=self.with_gc,
                    samples=self.n_samples,
                    batches=self.batch_sizes)

    @cached_property
    def x_y(self):
        assert self.batch_sizes[0] == 1
        return self.features.X_y[:, 0, :].mean(axis=0)

    @property
    def feature_names(self):
        return self.features.feature_names

    def regression(self, B=1000, **kwargs):
        n_samples = len(self.features.X_y)
        indices = np.random.randint(0, n_samples, size=(B, self.n_batches))
        # bootstrap
        resamples = self.features.X_y[indices, np.arange(self.n_batches)]
        arr_st_w = np.array([ridge_regression(resample)
                             for resample in resamples])
        stat_w = get_mean_stat(arr_st_w, **kwargs)

        self.arr_X_y = resamples
        self.arr_st_w = arr_st_w
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)

        w = stat_w.mean
        w_r2 = np.array([r2(self.features.y, X.dot(w))
                         for X in self.features.X]).mean()
        return Regression(stat_w, stat_y, w_r2)


def ridge_regression(Xy, alpha=0.15):
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
    Xy = np.atleast_2d(Xy)
    X, y = Xy[:, :-1], Xy[:, -1]

    N = X.shape[1]
    X_new = np.append(X, alpha * np.eye(N), axis=0)
    y_new = np.append(y, np.zeros(N))
    w, _residuals = nnls(X_new, y_new)
    return w


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std \
        else np.inf * (1 - np.mean((y_true-y_pred)**2))


Stat = namedtuple("Stat", "mean std ci")


def get_mean_stat(values, confidence=0.95):
    alpha = (1 - confidence) / 2
    lowhigh = mquantiles(values, prob=[alpha, 1 - alpha], axis=0)
    return Stat(np.mean(values, axis=0),
                np.std(values, ddof=1, axis=0),
                np.asarray(lowhigh))
