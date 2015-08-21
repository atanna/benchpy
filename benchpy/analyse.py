import numpy as np
from cached_property import cached_property
from collections import namedtuple, OrderedDict
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from .exception import BenchException

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'stat_w, stat_y')


def const_stat(x):
    return Stat(x, 0., np.array([x, x]))


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
                 gc_time=None, func_name="", gamma=0.95, type_ci="tquant"):
        self.n_samples, self.n_batches = full_time.shape
        self.full_time = full_time
        self.batch_sizes = batch_sizes
        self.with_gc = with_gc
        self.name = func_name
        self._init(gamma,  type_ci, gc_time)
        self.init_features()

    def init_features(self):
        self.features = Features(["batch", "const"],
                                 [self.batch_sizes, 1.],
                                 y=self.full_time)

    def _init(self, gamma=0.95, type_ci="tquant", gc_time=None):
        self.stat_time = None
        self.regr = None
        self._ci_params = dict(gamma=gamma, type_ci=type_ci)
        if gc_time is not None:
            self.gc_time = np.mean(np.mean(gc_time, axis=0)
                                   / self.batch_sizes)

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
        return np.max(self._av_time)

    @cached_property
    def _av_time(self):
        return self.full_time / self.batch_sizes[:, np.newaxis].T

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
        return self.full_time.mean(axis=0)

    @cached_property
    def features_time(self):
        return self.x_y[:-1] * self.regr.stat_w.val

    @property
    def ci_params(self):
        return self._ci_params

    @cached_property
    def fit_info(self):
        return dict(with_gc=self.with_gc,
                    samples=self.n_samples,
                    batches=self.batch_sizes)

    @cached_property
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
        indexes = np.random.random_integers(0, self.n_samples-1,
                                            size=(B, self.n_batches))
        stat_w, arr_X_y, arr_st_w = \
            get_statistic(self.features.X_y, lin_regression,
                          with_arr_values=True,
                          bootstrap_kwargs=
                          dict(indexes=(indexes, np.arange(self.n_batches))),
                          **kwargs)
        self.arr_X_y = arr_X_y
        self.arr_st_w = arr_st_w
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)
        return Regression(stat_w, stat_y)


def mean_and_se(stat_values, stat=None, eps=1e-9):
    """
    :param stat_values:
    :param stat:
    :param eps:
    :return: (mean(stat_values), se(stat_values))
    """
    mean_stat = np.mean(stat_values, axis=0)
    if stat is not None:
        mean_stat = 2 * stat - mean_stat
    n = len(stat_values)
    se_stat = np.std(stat_values, axis=0) * np.sqrt(n / (n-1))
    if type(se_stat) is np.ndarray:
        se_stat[se_stat < eps] = eps
    else:
        se_stat = max(se_stat, eps)
    return mean_stat, se_stat


def lin_regression(X, y=None, alpha=0.15):
    if y is None:
        _X, _y = X[:, :-1], X[:, -1]
    else:
        _X, _y = X, y
    n = _X.shape[1]
    X_new = np.concatenate([_X, alpha*np.eye(n)])
    y_new = np.concatenate([_y, np.zeros(n)])
    return nnls(X_new, y_new)[0]


def bootstrap(X, B=1000, indexes=None, **kwargs):
    n = len(X)
    if indexes is None:
        indexes = np.random.random_integers(0, n-1, size=(B, n))
    return X[indexes]


def get_statistic(values, f_stat, with_arr_values=False,
                  bootstrap_kwargs=None,
                  **ci_kwargs):
    """
    Return class Stat with statistic, std, confidence interval
    :param values:
    :param f_stat: f_stat(sample) = statistic on sample
    :return:
    """
    if bootstrap_kwargs is None:
        bootstrap_kwargs = {}
    arr_values = bootstrap(values, **bootstrap_kwargs)
    arr_stat = []
    for _values in arr_values:
        try:
            arr_stat.append(f_stat(_values))
        except:
            continue
    arr_stat = np.array(arr_stat)
    if with_arr_values:
        return get_mean_stat(arr_stat, **ci_kwargs), arr_values, arr_stat
    return get_mean_stat(arr_stat, **ci_kwargs)


def get_mean_stat(values, type_ci="efr", **kwargs):
    if len(values) == 1:
        return const_stat(values[0])
    dict_ci = dict(efr=mean_stat_efr,
                   quant=mean_stat_quant,
                   tquant=mean_stat_tquant,
                   hard_efron=mean_stat_hard_efron,
                   hard_efron2=mean_stat_hard_efron2)
    if type_ci in dict_ci:
        return dict_ci[type_ci](values, **kwargs)
    else:
        raise BenchException("unknown type of confidence interval '{}'"
                             .format(type_ci))


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std \
        else np.inf * (1 - np.mean((y_true-y_pred)**2))


def mean_stat_efr(arr, gamma=0.95):
    alpha = (1-gamma) / 2
    return Stat(*mean_and_se(np.array(arr)),
                ci=np.array(mquantiles(arr, prob=[alpha, 1-alpha],
                                       axis=0).T))


def mean_stat_quant(arr, gamma=0.95):
    alpha = (1-gamma) / 2
    mean_stat, se_stat = mean_and_se(arr)
    q = np.array(mquantiles(arr-mean_stat,
                            prob=[alpha, 1-alpha], axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat-q[1], mean_stat-q[0]]).T)


def mean_stat_tquant(arr, gamma=0.95):
    alpha = (1-gamma) / 2
    mean_stat, se_stat = mean_and_se(arr)
    q = np.array(mquantiles((arr-mean_stat) / se_stat,
                            prob=[alpha, 1-alpha],
                            axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - se_stat * q[1],
                          mean_stat - se_stat * q[0]]).T)


def _get_z_alph(a, z_0, alpha):
    _z_alpa = norm.ppf(alpha)
    return z_0 + (z_0 + _z_alpa) / (1 - a * (z_0 + _z_alpa))


def mean_stat_hard_efron(arr, gamma=0.95, **bootstrap_kwargs):
    alpha = (1-gamma) / 2
    mean_x = np.mean(arr)
    a = 1./6 * np.sum((arr-mean_x) ** 3) / \
        (np.sum((arr-mean_x)**2) ** (3/2))
    X_b = bootstrap(arr, **bootstrap_kwargs)
    stat_b = np.array(X_b).mean(axis=1)
    mean_stat, se_stat = mean_and_se(stat_b)
    stat_b.sort()
    n = len(stat_b)
    z_0 = norm.ppf(stat_b.searchsorted(mean_stat) / n)
    ci = mquantiles(stat_b, prob=[norm.cdf(_get_z_alph(a, z_0, alpha)),
                                  norm.cdf(_get_z_alph(a, z_0, 1-alpha))])
    return Stat(mean_stat, se_stat, ci)


def mean_stat_hard_efron2(arr, gamma=0.95):
    alpha = (1-gamma) / 2
    mean_x = np.mean(arr)
    a = 1./6 * np.sum((arr-mean_x) ** 3) / (
        np.sum((arr-mean_x)**2) ** (3 / 2))
    _z_alpha = norm.ppf(alpha)
    _z_alpha2 = norm.ppf(1 - alpha)
    sigma = arr.std()
    q1 = sigma * (_z_alpha + a*(2*_z_alpha**2 + 1))
    q2 = sigma * (_z_alpha2 + a*(2*_z_alpha2**2 + 1))
    return Stat(mean_x, sigma, [mean_x+q1, mean_x+q2])
