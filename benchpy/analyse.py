import functools
import numpy as np
from cached_property import cached_property
from collections import namedtuple, OrderedDict
from numpy.linalg import LinAlgError
from scipy.optimize import nnls
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from .exception import BenchException

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'X y stat_w stat_y r2')
Regression2 = namedtuple("Regression2", 'stat_w, stat_y')

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))


def const_stat(x):
    return Stat(x, 0., np.array([x, x]))


class StatMixin(object):
    def __init__(self, full_time, batch_sizes,
                 gc_time=None, gc_collected=None,
                 func_name="", gamma=0.95, type_ci="tquant"):
        self.n_samples, self.n_batches = full_time.shape
        self.full_time = full_time
        self.batch_sizes = batch_sizes
        self.func_name = func_name
        self._init(gamma, type_ci)
        self.init_features(gc_collected)
        if gc_time is not None:
            self.gc_time = np.mean(np.mean(gc_time, axis=0)
                                   / self.batch_sizes)

    def init_features(self, gc_collected):
        n_gc_generations = gc_collected.shape[-1]
        self.with_gc = False
        for i in range(1, n_gc_generations+1):
            if gc_collected[:,:,-i].sum() == 0:
                n_gc_generations -= 1
        if n_gc_generations > 0:
            self.with_gc = True
        _shape = (self.n_samples, self.n_batches, 1)
        self.features = ["batch", "const"] + ["gc_{}".format(i+1)
                                              for i in range(n_gc_generations)]
        self.X_y = \
            np.c_[np.array(list([self.batch_sizes])
                           *self.n_samples).reshape(_shape),
                  np.ones(_shape),
                  gc_collected[:, :, :n_gc_generations],
                  self.full_time.reshape(_shape)]

    def _init(self, gamma=0.95, type_ci="tquant"):
        self.stat_time = None
        self._r2 = None
        self._ci_params = dict(gamma=gamma, type_ci=type_ci)

    @property
    def name(self):
        return self.func_name

    @cached_property
    def time(self):
        try:
            self.regr = self.regression(**self._ci_params)
            self.stat_time = self.regr.stat_y
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                get_mean_stat(self.X_y[:,:,-1] / self.batch_sizes)
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
        self.min_res = np.min(self.full_time /
                              self.batch_sizes[:, np.newaxis].T)
        return self.min_res

    @cached_property
    def max(self):
        self.max_res = np.max(self.full_time /
                              self.batch_sizes[:, np.newaxis].T)
        return self.max_res

    @property
    def stat_w(self):
        self.time
        return self.regr.stat_w

    @cached_property
    def X(self):
        _X = self.X_y[:, :, :-1].mean(axis=0)
        for i, feature in enumerate(self.features):
            if feature == "const":
                _X[:, i] = 1.
                break
        return _X

    @cached_property
    def y(self):
        return self.full_time.mean(axis=0)

    @cached_property
    def features_time(self):
        return self.x_y[:-1] * self.regr.stat_w.val

    @cached_property
    def gc_predicted_time(self):
        res = 0
        for feature, time in zip(self.features, self.features_time):
            if feature.startswith("gc"):
                res += time
        return res

    @cached_property
    def time_without_gc_pred(self):
        return self.time - self.gc_predicted_time

    @property
    def ci_params(self):
        return self._ci_params

    @cached_property
    def fit_info(self):
        return dict(with_gc=self.with_gc,
                    samples=self.n_samples,
                    batches=self.batch_sizes)

    def evaluate_stats(self, arr_samples, f_stat, **kwargs):
        stats = [get_statistic(values=sample, f_stat=f_stat, **kwargs)
                 for sample in arr_samples]
        return stats

    def regression(self, B=1000, **kwargs):
        indexes = np.random.random_integers(0, self.n_samples-1,
                                            size=(B, self.n_batches))
        stat_w, arr_st_w = \
            get_statistic(self.X_y, lin_regression,
                          with_arr_values=True,
                          bootstrap_kwargs=
                          dict(indexes=(indexes, np.arange(self.n_batches))),
                          **kwargs)
        self.x_y = np.mean(self.X_y / self.batch_sizes[:, np.newaxis],
                           axis=(0, 1))
        self.arr_st_w = arr_st_w
        for i, feature in enumerate(self.features):
            if feature == "const":
                self.x_y[i] = 1.
                break
        arr_st_y = np.array([self.x_y[:-1].dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)
        return Regression2(stat_w, stat_y)


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


def lin_regression(X, y=None):
    if y is None:
        _X, _y = X[:, :-1], X[:, -1]
    else:
        _X, _y = X, y
    return nnls(_X, _y)[0]


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
        return get_mean_stat(arr_stat, **ci_kwargs), arr_stat
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
