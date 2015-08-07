import numpy as np
from collections import namedtuple, OrderedDict
from numpy.linalg import LinAlgError, inv
from prettytable import PrettyTable
from scipy.stats.mstats import mquantiles
from scipy.stats import norm

Stat = namedtuple("Stat", 'val std ci')
Regression = namedtuple("Regression", 'X y stat_w stat_y r2')

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))
GC_NUM_GENERATIONS = 3

def const_stat(x):
    return Stat(x, 0., np.array([x, x]))


class BenchResult():
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
                get_mean_stat(self.means / self.batch_sizes)
            self.r2 = 0

    def evaluate_stats(self, f_stat, **kwargs):
        stats = [get_statistic(values=batch_sample, f_stat=f_stat, **kwargs)
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

        stat_w, arr_st_w = \
            get_statistic(np.concatenate((X, y[:, np.newaxis]), axis=1),
                          lin_regression,
                          with_arr_values=True, **kwargs)
        x = np.mean(X / self.batch_sizes[:, np.newaxis], axis=0)
        arr_st_y = np.array([x.dot(w) for w in arr_st_w])
        stat_y = get_mean_stat(arr_st_y, **kwargs)
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
                raise BenchException("'{}' is unknown key".format(key))
            if not with_empty and not len(str(table[key])):
                continue
            _table_keys.append(key)
        pretty_table = self._get_pretty_table_header(measure, _table_keys)

        pretty_table.add_row([table[key] for key in _table_keys])
        return "\n" + str(pretty_table)

    def __repr__(self):
        return self._repr(with_empty=False)


class GroupResult():
    table_keys = BenchResult.table_keys

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
                    raise BenchException("'{}' is unknown key".format(key))
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


def mean_and_se(stat_values, stat=None, eps=1e-9):
    """
    :param stat_values:
    :param stat:
    :param eps:
    :return: (mean(stat_values), se(stat_values))
    """
    mean_stat = np.mean(stat_values, axis=0)
    if stat is not None:
        mean_stat = 2*stat - mean_stat
    n = len(stat_values)
    se_stat = np.std(stat_values, axis=0) * np.sqrt(n / (n - 1))
    if type(se_stat) is np.ndarray:
        se_stat[se_stat<eps] = eps
    else:
        se_stat = max(se_stat, eps)
    return mean_stat, se_stat


def lin_regression(X, y=None):
    if y is None:
        _X, _y = X[:, :-1], X[:, -1]
    else:
        _X, _y = X, y
    w = inv(_X.T.dot(_X)).dot(_X.T).dot(_y)
    return w


def bootstrap(X, B=1000, **kwargs):
    n = len(X)
    indexes = np.random.random_integers(0, n - 1, size=(B, n))
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
    alpha = (1 - gamma) / 2
    return Stat(*mean_and_se(np.array(arr)),
                ci=np.array(mquantiles(arr, prob=[alpha, 1 - alpha],
                                       axis=0).T))


def mean_stat_quant(arr, gamma=0.95):
    alpha = (1 - gamma) / 2
    mean_stat, se_stat = mean_and_se(arr)
    q = np.array(mquantiles(arr - mean_stat,
                            prob=[alpha, 1 - alpha], axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - q[1], mean_stat - q[0]]).T)


def mean_stat_tquant(arr, gamma=0.95):
    alpha = (1 - gamma) / 2
    mean_stat, se_stat = mean_and_se(arr)
    q = np.array(mquantiles((arr - mean_stat) / se_stat,
                            prob=[alpha, 1 - alpha],
                            axis=0))
    return Stat(mean_stat, se_stat,
                np.array([mean_stat - se_stat * q[1],
                          mean_stat - se_stat * q[0]]).T)


def _get_z_alph(a, z_0, alpha):
    _z_alpa = norm.ppf(alpha)
    return z_0 + (z_0 + _z_alpa) / (1 - a * (z_0 + _z_alpa))


def mean_stat_hard_efron(arr, gamma=0.95, **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(arr)
    a = 1. / 6 * np.sum((arr - mean_x) ** 3) / (
        np.sum((arr - mean_x) ** 2) ** (3 / 2))
    X_b = bootstrap(arr, **bootstrap_kwargs)
    stat_b = np.array(X_b).mean(axis=1)
    mean_stat, se_stat = mean_and_se(stat_b)
    stat_b.sort()
    n = len(stat_b)
    z_0 = norm.ppf(stat_b.searchsorted(mean_stat) / n)
    ci = mquantiles(stat_b, prob=[norm.cdf(_get_z_alph(a, z_0, alpha)),
                                  norm.cdf(_get_z_alph(a, z_0, 1 - alpha))])
    return Stat(mean_stat, se_stat, ci)


def mean_stat_hard_efron2(arr, gamma=0.95):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(arr)
    a = 1. / 6 * np.sum((arr - mean_x) ** 3) / (
        np.sum((arr - mean_x) ** 2) ** (3 / 2))
    _z_alpha = norm.ppf(alpha)
    _z_alpha2 = norm.ppf(1 - alpha)
    sigma = arr.std()
    q1 = sigma * (_z_alpha + a * (2 * _z_alpha ** 2 + 1))
    q2 = sigma * (_z_alpha2 + a * (2 * _z_alpha2 ** 2 + 1))
    return Stat(mean_x, sigma, [mean_x + q1, mean_x + q2])


