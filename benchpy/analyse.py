import numpy as np
from collections import namedtuple, OrderedDict
from numpy.linalg import LinAlgError, inv
from prettytable import PrettyTable
from scipy.stats.mstats import mquantiles
from scipy.stats import norm
from . import BmException, GC_NUM_GENERATIONS

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'X y stat_w stat_y r2')

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))


def const_stat(x):
    return Stat(x, 0., np.array([x, x]))


class BenchRes():
    table_keys = ['Name', 'Time', 'CI', 'Std', 'Min', 'Max', 'R2',
                  'gc_collections', 'fit_info']

    def __init__(self, res, gc_info, batch_sizes, with_gc, func_name=""):
        self.res = res
        self.gc_info = gc_info
        self.batch_sizes = batch_sizes
        self.with_gc = with_gc
        self.func_name = func_name
        self.n_batches = len(batch_sizes)
        self.n_samples = len(res)
        self._init_stats()

    def _init_stats(self, gamma=0.95, type_ci="tquant"):
        self.ci_params = dict(gamma=gamma, type_ci=type_ci)
        self.stat_means = self.evaluate_stats(f_stat=np.mean, **self.ci_params)
        self.means = np.array([stat_mean.val for stat_mean in self.stat_means])
        self.min_res = np.min(self.res/self.batch_sizes)
        self.max_res = np.max(self.res/self.batch_sizes)
        self.collections = self._get_collections()
        self.mean_collections = np.mean(self.collections /
                                self.batch_sizes)
        try:
            self.regr = self.regression(**self.ci_params)
            self.stat_time = self.regr.stat_y
            self.r2 = self.regr.r2
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                Stat(np.mean(self.means / self.batch_sizes),
                     None, [None, None])
            self.r2 = 0

    def evaluate_stats(self, f_stat, **kwargs):
        stats = [get_stat(X=batch_sample, f_stat=f_stat, **kwargs)
                 for batch_sample in self.res.T]
        return stats

    def _get_collections(self):
        res = []
        for batch in self.batch_sizes:
            _res = 0
            for i in range(GC_NUM_GENERATIONS):
                if batch in self.gc_info:
                    _res += self.gc_info[batch][0][i].get("collections", 0)
            res.append(_res)
        return np.array(res)

    def regression(self, **kwargs):
        X, y = self.get_features()
        if len(X) == 1:
            w = lin_regression(X, y)
            return Regression(X, y, const_stat(w[0]),
                              const_stat(y[0] / self.batch_sizes[0]), None)
        X_b = bootstrap(np.concatenate((X, y[:, np.newaxis]), axis=1),
                        **kwargs)
        arr_st_w = collect_stat(X_b=X_b, f_stat=lin_regression, **kwargs)
        stat_w = get_stat(arr_stat=arr_st_w, **kwargs)

        if arr_st_w.shape[1] > 1:
            x = np.array([1, self.mean_collections])
        else:
            x = np.array([1])
        arr_st_y = np.array([x.dot(w) for w in arr_st_w])
        stat_y = get_stat(arr_stat=arr_st_y, **kwargs)
        return Regression(X, y, stat_w, stat_y, r2(y, X.dot(stat_w.val)))

    def get_features(self):
        if len(self.gc_info):
            X = np.array([self.batch_sizes,
                          self.collections]).T
        else:
            X = np.array([self.batch_sizes]).T
        y = self.means
        return X, y

    def get_table(self, measure='s'):
        w = time_measures[measure]
        fit_info = "{with_gc}, {n_samples} samples, {n_batches} batches " \
                   "[{min_batch}..{max_batch}]"\
            .format(with_gc="with_gc" if self.with_gc else "without_gc",
                    n_samples=self.n_samples,
                    n_batches=self.n_batches,
                    min_batch=self.batch_sizes[0],
                    max_batch=self.batch_sizes[-1])
        table = dict(Name=self.func_name,
                     Time=self.stat_time.val * w,
                     CI=self.stat_time.ci * w,
                     Std=self.stat_time.std * w,
                     Min=self.min_res * w,
                     Max=self.max_res * w,
                     R2=self.r2,
                     gc_collections=self.mean_collections,
                     fit_info=fit_info)
        return table

    def choose_time_measure(self, perm_n_points=2):
        t = self.stat_time.val
        c = 10 ^ perm_n_points
        for measure, w in time_measures.items():
            if int(t * w * c):
                return measure

    def _get_pretty_table_header(self, measure=None, table_keys=None):
        if table_keys is None:
            table_keys = self.table_keys

        if measure is None:
            measure = self.choose_time_measure()
        pretty_table = PrettyTable(list(
            map(lambda key:
                "CI_{}[{}]"
                .format(self.ci_params["type_ci"],
                        self.ci_params["gamma"]) if key == "CI" else
                "Time ({})".format(measure) if key == "Time" else key,
                table_keys)))
        return pretty_table

    def _repr(self, table_keys=None, with_empty=True):
        measure = self.choose_time_measure()
        if table_keys is None:
            table_keys = ["Name", "Time", "CI"]
            if self.with_gc:
                table_keys.append("gc_collections")
        elif table_keys is "Full":
            table_keys = self.table_keys
        _table_keys = []
        table = self.get_table(measure)
        for key in table_keys:
            if key not in table:
                raise BmException("'{}' is unknown key".format(key))
            if not with_empty and not len(str(table[key])):
                continue
            _table_keys.append(key)
        pretty_table = self._get_pretty_table_header(measure, _table_keys)

        pretty_table.add_row([table[key] for key in _table_keys])
        return "\n" + str(pretty_table)

    def __repr__(self):
        return self._repr(with_empty=False)


class GroupRes():
    table_keys = BenchRes.table_keys

    def __init__(self, name, results):
        self.name = name
        self.results = results
        res = results[0]
        self.ci_params = res.ci_params
        self.n_samples = res.n_samples
        self.batch_sizes = res.batch_sizes
        self.n_batches = res.n_batches

    def _repr(self, table_keys=None, with_empty=True):
        if table_keys is None:
            table_keys = ["Name", "Time", "CI", "gc_collections"]
        elif table_keys is "Full":
            table_keys = self.table_keys
        first_res = self.results[0]
        measure = first_res.choose_time_measure()
        tables = [bm_res.get_table(measure) for bm_res in self.results]
        n_results = len(self.results)
        _table_keys = []
        for key in table_keys:
            n_empty_values = 0
            for table in tables:
                if key not in table:
                    raise BmException("'{}' is unknown key".format(key))
                if not len(str(table[key])):
                    n_empty_values += 1
            if not with_empty and n_empty_values == n_results:
                continue
            _table_keys.append(key)

        pretty_table = first_res._get_pretty_table_header(measure, _table_keys)
        pretty_table.align = 'l'
        for bm_res in self.results:
            table = bm_res.get_table(measure)
            pretty_table.add_row([table[key] for key in _table_keys])
        title = "\n{group:~^{n}}\n" \
            .format(group=self.name, n=10)
        return title + str(pretty_table)

    def __repr__(self):
        return self._repr(with_empty=False)


def _get_mean_se_stat(stat_b, stat=None, eps=1e-9):
    mean_stat = np.mean(stat_b, axis=0)
    if stat is not None:
        mean_stat = 2*stat - mean_stat
    n = len(stat_b)
    se_stat = np.std(stat_b, axis=0) * np.sqrt(n / (n - 1))
    eps = 1e-9
    if type(se_stat) is np.ndarray:
        se_stat[se_stat<eps] = eps
    else:
        se_stat = max(se_stat, eps)
    return mean_stat, se_stat


def get_stat(type_ci="efr", X=None, f_stat=None, arr_stat=None, **kwargs):
    """
    :param X:
    :param f_stat:
    :param B: bootstrap sample size (to define ci)
    :param type_ci: type of confidence interval {'efr', 'quant', 'tquant'}
    :param kwargs:
    :return:
    """
    if X is not None and len(X) == 1:
        m = f_stat(X)
        return const_stat(m)
    if arr_stat is not None and len(arr_stat) == 1:
        return const_stat(arr_stat[0])
    if type_ci == "efr":
        res = confidence_interval_efr(X=X, f_stat=f_stat, arr_stat=arr_stat,
                                      **kwargs)
    elif type_ci == "quant":
        res = confidence_interval_quant(X=X, f_stat=f_stat, arr_stat=arr_stat,
                                        **kwargs)
    elif type_ci == "tquant":
        res = confidence_interval_tquant(X=X, f_stat=f_stat, arr_stat=arr_stat,
                                         **kwargs)
    else:
        raise BmException("type of confidence interval '{}' is not defined"
                          .format(type_ci))
    return res


def r2(y_true, y_pred):
    std = y_true.std()
    return 1 - np.mean((y_true-y_pred)**2) / std if std \
        else np.inf * (1 - np.mean((y_true-y_pred)**2))


def collect_stat(X=None, f_stat=None, X_b=None, arr_stat=None,
                 **bootstrap_kwargs):
    if arr_stat is None:
        if f_stat is None:
           raise BmException("f_stat must be defined")
        if X_b is None and X is None:
            raise BmException("X or X_b must be defined")
        if X_b is None:
            X_b = bootstrap(X, **bootstrap_kwargs)
        arr_stat = []

        for x_b in X_b:
            try:
                arr_stat.append(f_stat(x_b))
            except Exception:
                continue
        arr_stat = np.array(arr_stat)
    return arr_stat


def confidence_interval_efr(gamma=0.95, **kwargs):
    alpha = (1 - gamma) / 2
    stat_b = collect_stat(**kwargs)
    return Stat(*_get_mean_se_stat(np.array(stat_b)),
                ci=np.array(mquantiles(stat_b, prob=[alpha, 1 - alpha],
                                       axis=0).T))


def confidence_interval_quant(gamma=0.95, **kwargs):
    alpha = (1 - gamma) / 2
    stat_b = collect_stat(**kwargs)
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    q = np.array(mquantiles(stat_b - mean_stat,
                            prob=[alpha, 1 - alpha], axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - q[1], mean_stat - q[0]]).T)


def confidence_interval_tquant(gamma=0.95, **kwargs):
    alpha = (1 - gamma) / 2
    stat_b = collect_stat(**kwargs)
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    q = np.array(mquantiles((stat_b - mean_stat) / se_stat,
                            prob=[alpha, 1 - alpha],
                            axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - se_stat * q[1],
                          mean_stat - se_stat * q[0]]).T)


def _get_z_alph(a, z_0, alpha):
    _z_alpa = norm.ppf(alpha)
    return z_0 + (z_0 + _z_alpa) / (1 - a * (z_0 + _z_alpa))


def confidence_interval_for_mean(X, gamma=0.95, **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(X)
    a = 1. / 6 * np.sum((X - mean_x) ** 3) / (
        np.sum((X - mean_x) ** 2) ** (3 / 2))
    X_b = bootstrap(X, **bootstrap_kwargs)
    stat_b = np.array(X_b).mean(axis=1)
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    stat_b.sort()
    n = len(stat_b)
    z_0 = norm.ppf(stat_b.searchsorted(mean_stat) / n)
    ci = mquantiles(stat_b, prob=[norm.cdf(_get_z_alph(a, z_0, alpha)),
                                  norm.cdf(_get_z_alph(a, z_0, 1 - alpha))])
    return Stat(mean_stat, se_stat, ci)


def confidence_interval_for_mean2(X, gamma=0.95):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(X)
    a = 1. / 6 * np.sum((X - mean_x) ** 3) / (
        np.sum((X - mean_x) ** 2) ** (3 / 2))
    _z_alpha = norm.ppf(alpha)
    _z_alpha2 = norm.ppf(1 - alpha)
    sigma = X.std()
    q1 = sigma * (_z_alpha + a * (2 * _z_alpha ** 2 + 1))
    q2 = sigma * (_z_alpha2 + a * (2 * _z_alpha2 ** 2 + 1))
    return Stat(mean_x, sigma, [mean_x + q1, mean_x + q2])


def index_bootstrap(n, size):
    return np.random.random_integers(0, n - 1, size=(size, n))


def bootstrap(X, B=1000, **kwargs):
    indexes = index_bootstrap(len(X), size=B)
    return list(map(lambda ind: X[ind], indexes))


def lin_regression(X, y=None):
    if y is None:
        _X, _y = X[:, :-1], X[:, -1]
    else:
        _X, _y = X, y
    w = inv(_X.T.dot(_X)).dot(_X.T).dot(_y)
    return w



