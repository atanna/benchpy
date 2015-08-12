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

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))


def const_stat(x):
    return Stat(x, 0., np.array([x, x]))


class StatMixin(object):
    def __init__(self, res, gc_collections, batch_sizes,
                 func_name="", gamma=0.95, type_ci="tquant",
                 gc_time=None):
        self.res = res
        self.all_collections = gc_collections
        self.batch_sizes = batch_sizes
        self.func_name = func_name
        self.gc_time = 0
        if gc_time is not None:
            self.gc_time = np.mean(gc_time/ self.batch_sizes)
        self.n_batches = len(batch_sizes)
        self.n_samples = len(res)
        self._init(gamma, type_ci)

    def _init(self, gamma=0.95, type_ci="tquant"):
        self.stat_time = None
        self._r2 = None
        self._ci_params = dict(gamma=gamma, type_ci=type_ci)
        self.X, self.y

    @property
    def name(self):
        return self.func_name

    @cached_property
    def time(self):
        try:
            self.regr = self.regression(**self._ci_params)
            self.stat_time = self.regr.stat_y
            self._r2 = self.regr.r2
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                get_mean_stat(self.y / self.batch_sizes)
            self._r2 = 0
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
        self.min_res = np.min(self.res / self.batch_sizes)
        return self.min_res

    @cached_property
    def max(self):
        self.max_res = np.max(self.res / self.batch_sizes)
        return self.max_res

    @property
    def r2(self):
        if self._r2 is None:
            self.time()
        return self._r2

    @cached_property
    def X(self):
        self.features = ["batch_size", "const"]
        if self.collections is not None:
            n_col = self.collections.shape[1]
            for i, col in enumerate(self.collections.T):
                if np.sum(col) == 0:
                    n_col = i
                    break
                self.features.append("gc_{}".format(i+1))
            return np.concatenate((self.batch_sizes[:, np.newaxis],
                                  np.ones((self.n_batches,1)),
                                self.collections[:, :n_col]), axis=1)
        return np.concatenate((self.batch_sizes[:, np.newaxis],
                                  np.ones((self.n_batches, 1))),
                               axis=1)

    @cached_property
    def features_time(self):
        return self.x * self.regr.stat_w.val

    @cached_property
    def x(self):
        return np.mean(self.X / self.batch_sizes[:, np.newaxis], axis=0)

    @cached_property
    def gc_predicted_time(self):
        res = 0
        for feature, time in zip(self.features, self.features_time):
            if feature.startswith("gc"):
                res += time
        return res

    @cached_property
    def collections(self):
        if self.all_collections is None:
            return None
        self.stat_collections = self.evaluate_stats(self.all_collections,
                                                    f_stat=functools.partial(np.mean, axis=0),
                                                    **self._ci_params)
        return np.array([stat_coll.val
                                for stat_coll in self.stat_collections])

    # @cached_property
    # def gc_collections(self):
    #     if self.collections is None:
    #         return 0.
    #     return self.regr.stat_w.val[1:] * self.x[1:]

    @cached_property
    def time_without_gc_pred(self):
        if self.collections is None:
            return self.time
        return self.time - self.gc_predicted_time

    @property
    def with_gc(self):
        return self.collections is not None

    @property
    def ci_params(self):
        return self._ci_params

    @cached_property
    def fit_info(self):
        return dict(with_gc=self.with_gc,
                    samples=self.n_samples,
                    batches=self.batch_sizes)

    @cached_property
    def y(self):
        self.stat_means = self.evaluate_stats(self.res.T,
                                              f_stat=np.mean,
                                              **self._ci_params)
        return np.array([stat_mean.val
                                for stat_mean in self.stat_means])

    def evaluate_stats(self, arr_samples, f_stat, **kwargs):
        stats = [get_statistic(values=sample, f_stat=f_stat, **kwargs)
                 for sample in arr_samples]
        return stats

    def regression(self, **kwargs):
        X, y = self.X, self.y
        if len(X) == 1:
            w = lin_regression(X, y)
            return Regression(X, y, const_stat(w[0]),
                              const_stat(y[0] / self.batch_sizes[0]), None)

        stat_w, arr_st_w = \
            get_statistic(np.concatenate((X, y[:, np.newaxis]), axis=1),
                          lin_regression,
                          with_arr_values=True, **kwargs)
        arr_st_y = np.array([self.x.dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)
        # self.stat_Y = get_mean_stat(np.array([X.dot(w) for w in arr_st_w]), **kwargs)
        return Regression(X, y, stat_w, stat_y, r2(y, X.dot(stat_w.val)))


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


def bootstrap(X, B=1000, **kwargs):
    n = len(X)
    indexes = np.random.random_integers(0, n-1, size=(B, n))
    return list(map(lambda ind: X[ind], indexes))


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
