import functools
import gc
import os
import timeit
import numpy as np
import pylab as plt
from collections import defaultdict, namedtuple
from numpy.linalg import LinAlgError
from prettytable import PrettyTable
from scipy.linalg import inv
from scipy.stats.mstats import mquantiles
from scipy.stats import norm

GC_NUM_GENERATIONS = 3
Stat = namedtuple("Stat", 'val std ci')
Bench = namedtuple("Case", 'name f run_params')
Group = namedtuple("Group", 'name group run_params')
Regression = namedtuple("Regression", 'X y stat_w stat_y r2')


class BmRes():
    def __init__(self, res, gc_info, batch_sizes, with_gc, func_name=""):
        self.res = res
        self.gc_info = gc_info
        self.batch_sizes = batch_sizes
        self.with_gc = with_gc
        self.func_name = func_name
        self.n_batches = len(batch_sizes)
        self.n_samples = len(res)

        self.ci_params = dict(gamma=0.95, type_ci="tquant")
        self.stat_means = self.evaluate_stats(f_stat=np.mean, **self.ci_params)
        self.means = np.array([stat_mean.val for stat_mean in self.stat_means])
        self.collections = self._get_collections()
        self.mean_collections = np.mean(self.collections[1:] /
                                self.batch_sizes[1:])
        try:
            self.regr = self.regression(**self.ci_params)
            self.stat_time = self.regr.stat_y
        except LinAlgError:
            self.regr = None
            self.stat_time = \
                Stat(np.mean(self.means[1:] / self.batch_sizes[1:]),
                     None, [None, None])

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

    def calculate_outliers(self, arr_r2, threshold=0.9):
        return np.sum(arr_r2 < threshold) / len(arr_r2)

    def get_features(self):
        if len(self.gc_info):
            X = np.array([self.batch_sizes,
                          self.collections]).T
        else:
            X = np.array([self.batch_sizes]).T
        y = self.means
        return X, y

    def info(self):
        table = PrettyTable(["Batch's size",
                             "mean",
                             "CI_{}[{}]".format(self.ci_params["type_ci"],
                                                self.ci_params["gamma"]),
                             "std",
                             "GC collections"])

        for batch, stat_mean, gc_c in \
                zip(self.batch_sizes, self.stat_means, self.collections):
            table.add_row([batch,
                           stat_mean.val, stat_mean.ci, stat_mean.std,
                           gc_c])

        table.align["mean"] = 'l'

        str_ = "{n_samples} samples, {n_batches} batches  # {gc}\n" \
            .format(func_name=self.func_name,
                    gc="with_gc" if self.with_gc else "without_gc",
                    n_samples=self.n_samples,
                    n_batches=self.n_batches)

        title = "\n{func_name:~^{s}}\n" \
            .format(func_name=self.func_name, s=len(str_)) \
            if len(self.func_name) \
            else "\n"
        return title + str_ + str(table)

    def __repr__(self):
        table = PrettyTable(["Time",
                             "CI_{}[{}]".format(self.ci_params["type_ci"],
                                                self.ci_params["gamma"]),
                             "std",
                             "R2",
                             "GC collections"])
        n_point = 8
        table.add_row([np.round(self.stat_time.val, n_point),
                       np.round(self.stat_time.ci, n_point),
                       np.round(self.stat_time.std, n_point),
                       np.round(self.regr.r2, n_point),
                       np.round(self.mean_collections, 2)])

        str_ = "{n_samples} samples, {n_batches} batches  # {gc}\n" \
            .format(func_name=self.func_name,
                    gc="with_gc" if self.with_gc else "without_gc",
                    n_samples=self.n_samples,
                    n_batches=self.n_batches)

        title = "\n{func_name:~^{s}}\n" \
            .format(func_name=self.func_name, s=len(str_)) \
            if len(self.func_name) \
            else "\n"
        return title + str_ + str(table)


class GroupRes():
    def __init__(self, name, results):
        self.name = name
        self.results = results
        res = results[0]
        self.ci_params = res.ci_params
        self.n_samples = res.n_samples
        self.batch_sizes = res.batch_sizes
        self.n_batches = res.n_batches

    def __repr__(self):
        table = PrettyTable(["func_name",
                             "Time",
                             "CI_{}[{}]".format(self.ci_params["type_ci"],
                                                self.ci_params["gamma"]),
                             "std",
                             "R2",
                             "GC collections",
                             "with_gc"])
        n_point = 8
        for bm_res in self.results:
            table.add_row([bm_res.func_name,
                           np.round(bm_res.stat_time.val, n_point),
                           np.round(bm_res.stat_time.ci, n_point),
                           np.round(bm_res.stat_time.std, n_point),
                           np.round(bm_res.regr.r2, n_point),
                           np.round(bm_res.mean_collections, 4),
                           bm_res.with_gc])
        str_ = "{n_samples} samples, {n_batches} batches \n" \
            .format(func_name=self.name,
                    n_samples=self.n_samples,
                    n_batches=self.n_batches)

        title = "\n{group:~^{s}}\n" \
            .format(group=self.name, s=len(str_))
        return title + str(table)


class BmException(Exception):
    pass


def _get_mean_se_stat(stat_b, stat=None):
    mean_stat = np.mean(stat_b, axis=0)
    if stat is not None:
        mean_stat = 2*stat - mean_stat
    n = len(stat_b)
    se_stat = np.std(stat_b, axis=0) * np.sqrt(n / (n - 1))
    return mean_stat, se_stat


def get_stat(type_ci="efr", **kwargs):
    """
    :param X:
    :param f_stat:
    :param B: bootstrap sample size (to define ci)
    :param type_ci: type of confidence interval {'efr', 'quant', 'tquant'}
    :param kwargs:
    :return:
    """
    if type_ci == "efr":
        res = confidence_interval_efr(**kwargs)
    elif type_ci == "quant":
        res = confidence_interval_quant(**kwargs)
    elif type_ci == "tquant":
        res = confidence_interval_tquant(**kwargs)
    else:
        raise BmException("type of confidence interval '{}' is not defined"
                          .format(type_ci))
    return res


def r2(y_true, y_pred):
    return 1 - np.mean((y_true-y_pred)**2) / y_true.std()


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


def bench(f, *args, run_params=None, func_name="", **kwargs):
    if run_params is None:
        run_params = {}
    return Bench(func_name,
                functools.partial(f, *args, **kwargs),
                run_params)


def group(name, group, **run_params):
    return Group(name, group, run_params)


def run(case, *args, **kwargs):
    if isinstance(case, Group):
        return GroupRes(case.name, [run(bench, *args,
                                            **dict(kwargs, **case.run_params))
                                    for bench in case.group])
    elif isinstance(case, Bench):
        return _run(case.f, *args, **dict(kwargs, func_name=case.name,
                                          **case.run_params))
    elif type(case) == list:
        return [run(_case, *args, **kwargs) for _case in case]
    else:
        raise BmException("Case must be Bench or Group or list")


def _warm_up(f, n=2):
    for i in range(n):
        f()


def _run(f, n_samples=10, max_batch=100, n_batches=10, with_gc=True,
         func_name="", n_efforts=2):
    """
    :param f: function without arguments
    :param batch_sizes:
    :param n_samples:
    :param with_gc:
    :return:
    """
    batch_sizes = np.arange(0, int(max_batch), int(max_batch / n_batches))

    try:
        _warm_up(f)
    except Exception as e:
        raise BmException("_warm_up")

    gc_disable = gc.disable
    if with_gc:
        gc.disable = lambda: None

    n_batches = len(batch_sizes)
    res = np.zeros((n_samples, n_batches))
    gc_info = defaultdict(list)

    for sample in range(n_samples):
        for i, batch in enumerate(batch_sizes):
            prev_stats = None
            for _ in range(n_efforts):
                try:
                    gc.collect()
                    prev_stats = gc.get_stats()
                    time = timeit.Timer(f).timeit(batch)
                    res[sample, i] = time
                    break
                except Exception as e:
                    if _ == n_efforts - 1:
                        raise BmException(e.args)
            diff, is_diff = diff_stats(prev_stats, gc.get_stats())
            if with_gc and is_diff and \
                    (batch not in gc_info or
                         not diff_equal(gc_info[batch][-1], diff)):
                gc_info[batch].append(diff)

    gc.disable = gc_disable
    return BmRes(res, gc_info, batch_sizes, with_gc, func_name)


def diff_stats(gc_stats0, gc_stats1):
    res = []
    is_diff = False
    for st0, st1 in zip(gc_stats0, gc_stats1):
        res.append({})
        for key in st0.keys():
            diff = st1[key] - st0[key]
            if diff:
                res[-1][key] = diff
                is_diff = True
    return res, is_diff


def diff_equal(diff1, diff2):
    for d1, d2 in zip(diff1, diff2):
        for key in set().union(d1.keys()).union(d2.keys()):
            if key not in d1 \
                    or key not in d2 \
                    or d1[key] != d2[key]:
                return False
    return True


def _dark_color(color, alpha=0.1):
    return np.array(list(map(lambda x: 0 if x < 0 else x, color - alpha)))


def _plot_result(bm_res, fig=None, n_ax=0, label="",
                c=None,
                title="",
                s=240, shift=0., alpha=0.2,
                text_size=20, linewidth=2,
                add_text=True
                ):
    if c is None:
        c = np.array([[0], [0.], [0.75]])

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[n_ax]

    batch_shift = shift * bm_res.batch_sizes[1]
    batch_sizes_ = bm_res.batch_sizes + batch_shift

    for res_ in bm_res.res:
        ax.scatter(batch_sizes_, res_, c=c, s=s, alpha=alpha)
    ax.scatter(0, 0, c=c, label=label)
    ax.plot(batch_sizes_, bm_res.means,
            c=c, linewidth=linewidth, label="{}_mean".format(label))

    if bm_res.regr is not None:
        w = bm_res.regr.stat_w.val
        ax.plot(batch_sizes_, bm_res.regr.X.dot(w), 'r--', c=_dark_color(c),
                label="{}_lin_regr, w={}".format(label, w),
                linewidth=linewidth)

    ymin, ymax = ax.get_ylim()
    gc_collect = False
    for n_cs, batch in zip(bm_res.collections, bm_res.batch_sizes):
        if n_cs:
            ax.text(batch, ymin + shift * (ymax - ymin), n_cs,
                    color=tuple(c.flatten()), size=text_size)
            gc_collect = True

    ax.legend()
    if add_text:
        if gc_collect:
            ax.text(0, ymin, "gc collections:", size=text_size)

        ax.set_xlabel('batch_sizes')
        ax.set_ylabel('time')
        ax.grid(True)
        ax.set_title(title)
    return fig


def _plot_group(gr_res, labels=None, **kwargs):
    list_res = gr_res.results
    n_res = len(gr_res.results)
    if labels is None:
        labels = list(range(n_res))
        for i, res in enumerate(list_res):
            if len(res.func_name):
                labels[i] = res.func_name

    batch_shift = 0.15 / n_res
    fig = plt.figure()
    fig.add_subplot(111)
    add_text=True
    for i, res, label in zip(range(n_res), list_res, labels):
        d = dict(fig=fig, label=label,
                                        title=gr_res.name,
                                        c=np.random.rand(3, 1),
                                        add_text=add_text,
                                        shift=batch_shift * i)
        d.update(kwargs)
        fig = _plot_result(res, **d)
        add_text=False
    return fig


def plot_results(res, **kwargs):
    if isinstance(res, BmRes):
        return _plot_result(res, **kwargs)
    elif isinstance(res, GroupRes):
        return _plot_group(res, **kwargs)
    elif type(res) == list:
        return [plot_results(_res) for _res in res]
    else:
        raise BmException("res must be BmRes or GroupRes or list")


def lin_regression(X, y=None):
    if y is None:
        _X, y = X[:,:-1], X[:,-1]
    else:
        _X = X
    w = inv(_X.T.dot(_X)).dot(_X.T).dot(y)
    return w


def save_plot(fig, func_name="f", path=None, dir="img"):
    if path is None:
        dir_ = "{}/{}/".format(dir, func_name)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        path = "{}/{}.jpg".format(dir_, np.random.randint(100))
    fig.savefig(path)
    return path

