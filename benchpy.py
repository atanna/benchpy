import gc
import timeit
import functools
import os
import numpy as np
import pylab as plt
from collections import defaultdict
from scipy.linalg import inv
from scipy.stats.mstats import mquantiles

GC_NUM_GENERATIONS = 3


class BmRes():
    def __init__(self, res, gc_info, batch_sizes):
        self.res = res
        self.gc_info = gc_info
        self.batch_sizes = batch_sizes
        self.n_batches = len(batch_sizes)
        self.means, self.ci_means = self.evaluate_stats(f_stat=np.mean)
        self.collections = self._get_collections()

    def evaluate_stats(self, f_stat, B=10):
        stats, ci = [], []
        for i, batch_sample in enumerate(self.res.T):
            _stat, _ci = bootstrap(batch_sample, f_stat=f_stat, B=B)
            stats.append(_stat)
            ci.append(_ci)
        return stats, ci

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


def confidence_interval(x, alpha=0.95):
    beta = (1 - alpha) / 2
    return mquantiles(x, prob=[beta, 1 - beta])


def index_bootstrap(n, size):
    return np.random.random_integers(0, n - 1, size=(size, n))


def bootstrap(X, f_stat=np.mean, B=10, alpha=0.95):
    indexes = index_bootstrap(len(X), size=B)
    res = [f_stat(X[ind]) for ind in indexes]
    return np.mean(res), confidence_interval(res, alpha)


def case(f, *args, run_params={}, **kwargs):
    return dict(f=functools.partial(f, *args, **kwargs), run_params=run_params)


def run(cases, *args, **kwargs):
    return [_run(case['f'], *args, **dict(kwargs, **case['run_params']))
            for case in cases]


def _warm_up(f, n=2):
    for i in range(n):
        f()


def _run(f, n_samples=10, max_batch=100, n_batches=10, with_gc=True):
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
    return BmRes(res, gc_info, batch_sizes)


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
                print(key, d1, d2)


def _dark_color(color, alpha=0.1):
    return np.array(list(map(lambda x: 0 if x < 0 else x, color - alpha)))


def _plot_result(bm_res, fig=None, n_ax=0, label="",
                 c=np.array([[0], [0.], [0.75]]),
                 title="",
                 s=240, batch_shift=0., lin_regr=True, alpha=0.2):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[n_ax]

    batch_sizes_ = bm_res.batch_sizes
    batch_sizes_ += batch_shift * batch_sizes_[1]
    gc_info = bm_res.gc_info
    for res_ in bm_res.res:
        ax.scatter(batch_sizes_, res_, c=c, s=s, alpha=alpha)
    ax.scatter(0, 0, c=c, label=label)
    ax.plot(batch_sizes_, bm_res.means,
            c=c, linewidth=2, label="{}_mean".format(label))

    if lin_regr:
        X, y = bm_res.get_features()
        w = lin_regression(X, y)
        ax.plot(batch_sizes_, X.dot(w), 'r--', c=_dark_color(c),
                label="{}_lin_regr, w={}".format(label, w), linewidth=2)

    ymin, _ = ax.get_ylim()
    for batch in gc_info:
        text = ""
        for i in range(GC_NUM_GENERATIONS):
            text += "{}\n".format(gc_info[batch][0][i]
                                  .get("collections", 0))
        ax.annotate(text[:-1], (batch + batch_shift, ymin),
                    color=tuple((c.flatten())),
                    size=25)

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
        labels = range(n_res)

    batch_shift = 0.15 / n_res
    fig = plt.figure()
    fig.add_subplot(111)
    for i, res, label in zip(range(n_res), list_res, labels):
        fig = _plot_result(res, fig=fig, label=label, c=np.random.rand(3, 1),
                           batch_shift=batch_shift * i, **kwargs)
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
