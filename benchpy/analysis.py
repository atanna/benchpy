# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles

from .utils import cached_property

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'stat_w stat_y r2')


class Features(object):
    def __init__(self, feature_names, features, y):
        self.feature_names = np.array(feature_names)
        self.y = y
        self.n = len(self.feature_names)
        _shape = y.shape + (1,)
        # _shape = (n_samples, n_batches, 1)  - one of feature_column in
        # full fitting matrix X with shape (n_samples, n_batches, n_features)
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
                 gc_time=None, func_name="", gamma=0.95, type_ci="efr"):
        self.full_time = full_time
        self.n_samples, self.n_batches = full_time.shape
        self.batch_sizes = batch_sizes
        self.with_gc = with_gc
        self.name = func_name
        self._init(gamma,  type_ci)
        self.init_features(full_time, gc_time)

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

    def _init(self, gamma=0.95, type_ci="tquant"):
        self.stat_time = None
        self.regr = None
        self._ci_params = dict(gamma=gamma, type_ci=type_ci)

    @cached_property
    def time(self):
        try:
            self.regr = self.regression(**self._ci_params)
            self.stat_time = self.regr.stat_y
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                get_mean_stat(self._av_time)
        return self.stat_time.val

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
        return self.x_y[:-1] * self.regr.stat_w.val

    @property
    def predicted_time_witout_gc(self):
        return self.time - self.gc_time

    @property
    def r2(self):
        if self.regr is None:
            self.time()
        return self.regr.r2

    @property
    def ci_params(self):
        return self._ci_params

    @cached_property
    def fit_info(self):
        return dict(with_gc=self.with_gc,
                    samples=self.n_samples,
                    batches=self.batch_sizes)

    @property
    def x_y(self):
        if self.batch_sizes[0] == 1:
            return get_mean_stat(self.features.X_y[:, 0, :], **self._ci_params)\
                .val
        else:
            _x_y = np.mean(self.features.X_y / self.batch_sizes[:, np.newaxis],
                           axis=(0, 1))
            ind_const = self.features._get_id("const")
            if ind_const > 0:
                _x_y[ind_const] = 1.
            return _x_y

    @property
    def feature_names(self):
        return self.features.feature_names

    def regression(self, B=1000, **kwargs):
        n_samples = len(self.features.X_y)
        indices = np.random.randint(0, n_samples, size=(B, self.n_batches))
        resamples = self.features.X_y[indices, np.arange(self.n_batches)]
        arr_X_y = resamples
        arr_st_w = bootstrap(ridge_regression, self.features.X_y, resamples)
        # XXX apparently, I'm doing something wrong here.
        # mean_st_w = ridge_regression(np.concatenate(
        #     self.features.X_y / self.batch_sizes[:, np.newaxis], axis=0))
        stat_w = get_mean_stat(arr_st_w, mean_val=None, **kwargs)

        self.arr_X_y = arr_X_y
        self.arr_st_w = arr_st_w
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)

        w = stat_w.val
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


def bootstrap(statistic, X, resamples):
    """
    ``\hat(theta|values) = f_stat(values)``
    Count estimation of statistic ``theta`` (with confidence interval and se)
    on values using bootstrapping.
    :param values:
    :param f_stat: function for `\theta` estimation
    :param with_arr_values: flag to return bootstrap samples
    :param bootstrap_kwargs: parameters for bootstrapping
    :param ci_kwargs: parameters for confidence interval.
    """
    res = []
    for resample in resamples:
        res.append(statistic(resample))

    return np.array(res)


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std \
        else np.inf * (1 - np.mean((y_true-y_pred)**2))


def _mean_and_se(stat_values, stat=None):
    """
    Method for bootstrapping
    Count mean and se. Use statistic in the bootstrap samples
    :param stat_values: statistic values in the bootstrap samples
    :param stat: statistic value in the full sample
    :return: (mean(stat_values), se(stat_values))
    """
    mean_stat = np.mean(stat_values, axis=0)
    se_stat = np.maximum(np.std(stat_values, ddof=1, axis=0),
                         np.finfo(float).eps)
    if stat is not None:
        mean_stat = 2 * stat - mean_stat
    return mean_stat, se_stat


def get_mean_stat(values, type_ci="efr", gamma=0.95, mean_val=None):
    method = type_ci    # better name
    confidence = gamma  # better name
    alpha = (1 - confidence) / 2
    if len(values) == 1:
        [value] = values
        return Stat(value, 0., np.array([value, value]))

    mean_stat, se_stat = _mean_and_se(values, mean_val)
    if method == "efr":
        low, high = mquantiles(values, prob=[alpha, 1-alpha], axis=0)
    elif method == "quant":
        q = mquantiles(values - mean_stat, prob=[alpha, 1-alpha], axis=0)
        low, high = mean_stat - q[1], mean_stat - q[0]
    elif method == "tquant":
        q = mquantiles((values - mean_stat) / se_stat,
                       prob=[alpha, 1-alpha], axis=0)
        low, high = mean_stat - se_stat * q[1], mean_stat - se_stat * q[0]
    else:
        raise ValueError("unknown method: {0!r}".format(method))

    return Stat(mean_stat, se_stat, np.array([low, high]))
