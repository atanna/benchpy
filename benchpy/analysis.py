# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles

from .exceptions import BenchException
from .utils import cached_property

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'stat_w stat_y r2')


class Features(object):
    def __init__(self, feature_names, features, y,
                 main_feature="batch",
                 fixed_dependency=False):
        self.feature_names = np.array(feature_names)
        self.y = y
        self.main_feature = main_feature
        _shape = (y.shape) +(1,)
        _features = []
        for feature in features:
            if np.array(feature).ndim == y.ndim:
                _feature = feature
            else:
                ndim = np.array(feature).ndim
                _feature = np.array(list([feature]) *
                                    np.prod(y.shape[:y.ndim-ndim])) \
                    .reshape(_shape)
            _features.append(_feature)
        _features.append(y.reshape(_shape))
        self.X_y = np.concatenate(_features, axis=2)
        self._renumbered()
        if fixed_dependency:
            self.fixed_dependency()

    def delete(self, features):
        indexes = self._indexes(features)
        self._del_id(indexes)

    def delete_all_features_except(self, features):
        indexes = self._indexes(features)
        del_indexes = list(set(range(self.n)) - set(indexes))
        self._del_id(del_indexes)

    def delete_gc(self):
        gc_features = self._get_gc_features()
        self.delete(gc_features)

    def get_X_without(self, features):
        _indexes = self._indexes(features)
        indexes = list(set(range(self.n)) - set(_indexes))
        return self.X[:, :, indexes]

    def get_X_with(self, features):
        indexes = self._indexes(features)
        return self.X[:, :, indexes]

    def _get_gc_features(self):
        return list(filter(lambda x: x.startswith("gc"), self.feature_names))

    def _del_id(self, indexes):
        self.X_y = np.delete(self.X_y, indexes, axis=2)
        self.feature_names = np.delete(self.feature_names, indexes)
        self._renumbered()

    def _indexes(self, features):
        return [self._get_id(feature) for feature in features]

    def _renumbered(self):
        for i, feature in enumerate(self.feature_names):
            self._set_id(feature, i)

    def _get_id(self, feature):
        return self.__dict__.get(feature, -1)

    def _set_id(self, feature, i):
        self.__dict__[feature] = i

    @property
    def X(self):
        return self.X_y[:, :, :-1]

    @property
    def n(self):
        return len(self.feature_names)

    def is_depended(self, X=None, threshold=1e-1):
        if X is None:
            X = self.X
        if X.shape[2] < 2:
            return False
        s = np.linalg.svd(X)[1].sum(axis=0)
        return sum(s < threshold * len(X))

    def fixed_dependency(self, threshold=1e-1):
        if self.X.shape[1] == 1:
            self.delete_all_features_except([self.main_feature])

        if not self.is_depended(threshold=threshold):
            return

        for i, feature in enumerate(self.feature_names[::-1]):
            if feature != self.main_feature:
                dep = self.is_depended(self.get_X_without([feature]), threshold)
                if not dep:
                    self.delete([feature])
                    return
        self.delete(self.feature_names[-2:])
        self.fixed_dependency()


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

    def init_features(self, full_time, gc_time=None, alpha=0.4, threshold=10):
        y = full_time
        if self.n_samples > threshold:
            order = full_time.argsort(axis=0)
            ind = (order, range(full_time.shape[1]))
            self.n_used_samples = max(threshold, (int(alpha*self.n_samples)))
            y = full_time[ind][:self.n_used_samples]
            if gc_time is not None:
                gc_time = gc_time[ind][:self.n_used_samples]
        if gc_time is not None:
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
        indexes = np.random.random_integers(0, n_samples-1,
                                            size=(B, self.n_batches))
        stat_w, arr_X_y, arr_st_w = \
            get_statistic(self.features.X_y, ridge_regression,
                          with_arr_values=True,
                          bootstrap_kwargs=
                          dict(indexes=(indexes, np.arange(self.n_batches))),
                          **kwargs)
        self.arr_X_y = arr_X_y
        self.arr_st_w = arr_st_w
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)

        w = stat_w.val
        w_r2 = np.array([r2(self.features.y, X.dot(w))
                         for X in self.features.X]).mean()
        return Regression(stat_w, stat_y, w_r2)


def _mean_and_se(stat_values, stat=None, eps=1e-9):
    """
    Method for bootstrapping
    Count mean and se. Use statistic in the bootstrap samples
    :param stat_values: statistic values in the bootstrap samples
    :param stat: statistic value in the full sample
    :param eps: we use eps to define se when it has zero value
    (to avoid ZeroDivisionError).
    :return: (mean(stat_values), se(stat_values))
    """
    mean_stat = np.mean(stat_values, axis=0)
    n = len(stat_values)
    se_stat = np.std(stat_values, axis=0) * np.sqrt(n / (n-1))
    if type(se_stat) is np.ndarray:
        se_stat[se_stat < eps] = eps
    else:
        se_stat = max(se_stat, eps)
    if stat is not None:
        mean_stat = 2 * stat - mean_stat
    return mean_stat, se_stat


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


def resample(X, B=1000, indexes=None, **kwargs):
    """
    Return new `B` samples, where every sample has n elements and i-th element
    of sample be chosen from i-th column of the matrix X,
    where (n,m) is X shape.
    If indexes is not None, we return X[indexes]
    :param X: matrix for bootstrapping
    :param B: number of samples (useful only if indexes is None)
    :param indexes:
    :param kwargs:
    """
    n = len(X)
    if indexes is None:
        indexes = np.random.random_integers(0, n-1, size=(B, n))
    return X[indexes]


def get_statistic(values, f_stat, with_arr_values=False,
                  bootstrap_kwargs=None,
                  **ci_kwargs):
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
    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}
    arr_values = resample(values, **bootstrap_kwargs)
    arr_stat = []
    for _values in arr_values:
        try:
            arr_stat.append(f_stat(_values))
        except Exception:
            continue

    mean_val = None
    try:
        mean_val = f_stat(values)
    except Exception:
        pass
    arr_stat = np.array(arr_stat)
    if with_arr_values:
        return get_mean_stat(arr_stat, mean_val=mean_val, **ci_kwargs), \
               arr_values, arr_stat
    return get_mean_stat(arr_stat, mean_val=mean_val, **ci_kwargs)


def get_mean_stat(values, type_ci="efr", **kwargs):
    if len(values) == 1:
        return _const_stat(values[0])
    dict_ci = dict(efr=mean_stat_efr,
                   quant=mean_stat_quant,
                   tquant=mean_stat_tquant)
    if type_ci in dict_ci:
        return dict_ci[type_ci](values, **kwargs)
    else:
        raise BenchException("unknown type of confidence interval '{}'"
                             .format(type_ci))


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std \
        else np.inf * (1 - np.mean((y_true-y_pred)**2))


def mean_stat_efr(arr, gamma=0.95, mean_val=None):
    alpha = (1-gamma) / 2
    return Stat(*_mean_and_se(np.array(arr), mean_val),
                ci=np.array(mquantiles(arr, prob=[alpha, 1-alpha],
                                       axis=0).T))


def mean_stat_quant(arr, gamma=0.95, mean_val=None):
    alpha = (1-gamma) / 2
    mean_stat, se_stat = _mean_and_se(arr, mean_val)
    q = np.array(mquantiles(arr-mean_stat,
                            prob=[alpha, 1-alpha], axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat-q[1], mean_stat-q[0]]).T)


def mean_stat_tquant(arr, gamma=0.95, mean_val=None):
    alpha = (1-gamma) / 2
    mean_stat, se_stat = _mean_and_se(arr, mean_val)
    q = np.array(mquantiles((arr-mean_stat) / se_stat,
                            prob=[alpha, 1-alpha],
                            axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - se_stat * q[1],
                          mean_stat - se_stat * q[0]]).T)


def _const_stat(x):
    return Stat(x, 0., np.array([x, x]))
