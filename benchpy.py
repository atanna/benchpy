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
Stat = namedtuple('Stat', 'val se ci')


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
        self.means = [stat_mean.val for stat_mean in self.stat_means]
        self.collections = self._get_collections()

    def evaluate_stats(self, f_stat, **kwargs):
        stats = [get_stat(batch_sample*100, f_stat, **kwargs)
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
        return res

    def get_features(self):
        if len(self.gc_info):
            X = np.array([self.batch_sizes,
                          self.collections]).T
        else:
            X = np.array([self.batch_sizes]).T
        y = self.means
        return X, y

    def __repr__(self):
        table = PrettyTable(["Batch's size",
                             "Mean time",
                             "SE",
                             "CI_{}[{}]".format(self.ci_params["type_ci"],
                                                    self.ci_params["gamma"]),
                             "GC collections"])

        for batch, stat_mean, gc_c in \
                zip(self.batch_sizes, self.stat_means, self.collections):
            table.add_row([batch,
                           stat_mean.val, stat_mean.se, stat_mean.ci,
                           gc_c])

        table.align["Mean time"] = 'l'

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


def _get_mean_se_stat(stat_b, stat=None):
    mean_stat = np.mean(stat_b) if stat is None \
        else 2*stat - np.mean(stat_b)
    n = len(stat_b)
    se_stat = np.std(stat_b) * np.sqrt(n/(n-1))
    return mean_stat, se_stat


def get_stat(X, f_stat, B=100, type_ci="efr", **ci_params):
    X_b = bootstrap(X, B=B)
    if type_ci == "efr":
        res = confidence_interval_efr(X, f_stat, X_b=X_b, **ci_params)
    elif type_ci == "quant":
        res = confidence_interval_quant(X, f_stat, X_b=X_b, **ci_params)
    elif type_ci == "tquant":
        res = confidence_interval_tquant(X, f_stat, X_b=X_b, **ci_params)
    else:
        assert False, "confidence interval {} is not defined"
    return res


def confidence_interval_efr(X, f_stat, X_b=None, gamma=0.95,
                            **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    if X_b is None:
        X_b = bootstrap(X, **bootstrap_kwargs)
    stat_b = np.array(list(map(f_stat, X_b)))
    return Stat(*_get_mean_se_stat(stat_b),
                ci=mquantiles(stat_b, prob=[alpha, 1-alpha]))


def confidence_interval_quant(X, f_stat, X_b=None, gamma=0.95,
                              **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    if X_b is None:
        X_b = bootstrap(X, **bootstrap_kwargs)
    stat_b = np.array(list(map(f_stat, X_b)))
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    q = mquantiles(stat_b-mean_stat, prob=[alpha, 1-alpha])
    return Stat(mean_stat, se_stat, [mean_stat-q[1], mean_stat-q[0]])


def confidence_interval_tquant(X, f_stat, X_b=None, gamma=0.95,
                               **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    if X_b is None:
        X_b = bootstrap(X, **bootstrap_kwargs)
    stat_b = np.array(list(map(f_stat, X_b)))
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    q = mquantiles((stat_b-mean_stat)/se_stat, prob=[alpha, 1-alpha])
    return Stat(mean_stat, se_stat,
                [mean_stat-se_stat*q[1], mean_stat-se_stat*q[0]])


def _get_z_alph(a, z_0, alpha):
    _z_alpa = norm.ppf(alpha)
    return z_0 + (z_0 + _z_alpa) / (1 - a*(z_0 + _z_alpa))


def confidence_interval_for_mean(X, gamma=0.95, **bootstrap_kwargs):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(X)
    a = 1./6 * np.sum((X-mean_x)**3) / (np.sum((X-mean_x)**2)**(3/2))
    X_b = bootstrap(X, **bootstrap_kwargs)
    stat_b = np.array(X_b).mean(axis=1)
    mean_stat, se_stat = _get_mean_se_stat(stat_b)
    stat_b.sort()
    n = len(stat_b)
    z_0 = norm.ppf(stat_b.searchsorted(mean_stat) / n)
    ci = mquantiles(stat_b, prob=[norm.cdf(_get_z_alph(a, z_0, alpha)),
                                  norm.cdf(_get_z_alph(a, z_0, 1-alpha))])
    return Stat(mean_stat, se_stat, ci)


def confidence_interval_for_mean2(X, gamma=0.95):
    alpha = (1 - gamma) / 2
    mean_x = np.mean(X)
    a = 1./6 * np.sum((X-mean_x)**3) / (np.sum((X-mean_x)**2)**(3/2))
    _z_alpha = norm.ppf(alpha)
    _z_alpha2 = norm.ppf(1-alpha)
    sigma = X.std()
    q1 = sigma*(_z_alpha + a*(2*_z_alpha**2 + 1))
    q2 = sigma*(_z_alpha2 + a*(2*_z_alpha2**2 + 1))
    return Stat(mean_x, sigma, [mean_x + q1, mean_x + q2])


def index_bootstrap(n, size):
    return np.random.random_integers(0, n-1, size=(size, n))


def bootstrap(X, B=1000):
    indexes = index_bootstrap(len(X), size=B)
    return list(map(lambda ind: X[ind], indexes))


def case(f, *args, run_params=None, func_name="", **kwargs):
    if run_params is None:
        run_params = {}
    return dict(f=functools.partial(f, *args, **kwargs),
                run_params=dict(run_params, func_name=func_name))


def run(cases, *args, **kwargs):
    return [_run(case['f'], *args, **dict(kwargs, **case['run_params']))
            for case in cases]


def _warm_up(f, n=2):
    for i in range(n):
        f()


def _run(f, n_samples=10, max_batch=100, n_batches=10, with_gc=True,
         func_name=""):
    """
    :param f: function without arguments
    :param batch_sizes:
    :param n_samples:
    :param with_gc:
    :return:
    """
    batch_sizes = np.arange(0, int(max_batch), int(max_batch / n_batches))
    _warm_up(f)
    gc_disable = gc.disable
    if with_gc:
        gc.disable = lambda: None

    n_batches = len(batch_sizes)
    res = np.zeros((n_samples, n_batches))
    gc_info = defaultdict(list)

    for sample in range(n_samples):
        for i, batch in enumerate(batch_sizes):
            gc.collect()
            prev_stats = gc.get_stats()
            res[sample, i] = timeit.Timer(f).timeit(batch)
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
                 c=np.array([[0], [0.], [0.75]]),
                 title="",
                 s=240, shift=0., lin_regr=True, alpha=0.2,
                 text_size=20, linewidth=2):
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

    if lin_regr:
        try:
            X, y = bm_res.get_features()
            w = lin_regression(X, y)
            ax.plot(batch_sizes_, X.dot(w), 'r--', c=_dark_color(c),
                    label="{}_lin_regr, w={}".format(label, w),
                    linewidth=linewidth)
        except LinAlgError:
            pass

    ymin, ymax = ax.get_ylim()
    gc_collect = False
    for n_cs, batch in zip(bm_res.collections, bm_res.batch_sizes):
        if n_cs:
            ax.text(batch, ymin+shift*(ymax-ymin), n_cs,
                    color=tuple(c.flatten()), size=text_size)
            gc_collect = True
    if gc_collect:
        ax.text(0, ymin, "gc collections:", size=text_size)

    ax.legend()
    ax.set_xlabel('batch_sizes')
    ax.set_ylabel('time')
    ax.grid(True)
    ax.set_title(title)
    return fig


def show_work_gc(case, *args, **kwargs):
    f, run_params = case["f"], case["run_params"]
    res_without_gc = _run(f, *args, with_gc=False, **dict(kwargs, **run_params))
    res_with_gc = _run(f, *args, with_gc=True, **dict(kwargs, **run_params))

    return plot_results([res_with_gc, res_without_gc],
                        labels=["with_gc", "without_gc"])


def plot_results(list_res, labels=None, **kwargs):
    n_res = len(list_res)
    if labels is None:
        labels = list(range(n_res))
        for i, res in enumerate(list_res):
            if len(res.func_name):
                labels[i] = res.func_name

    batch_shift = 0.15 / n_res
    fig = plt.figure()
    fig.add_subplot(111)
    for i, res, label in zip(range(n_res), list_res, labels):
        fig = _plot_result(res, fig=fig, label=label, c=np.random.rand(3, 1),
                           shift=batch_shift * i, **kwargs)
    return fig


def lin_regression(X, y):
    w = inv(X.T.dot(X)).dot(X.T).dot(y)
    return w


def save_plot(fig, func_name="f", path=None, dir="img"):
    if path is None:
        dir_ = "{}/{}/".format(dir, func_name)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        path = "{}/{}.jpg".format(dir_, np.random.randint(100))
    fig.savefig(path)
    return path
