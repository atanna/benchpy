import gc
import os
import timeit
import functools
import numpy as np
import pylab as plt
from io import StringIO
from collections import defaultdict
from django.template import Context
from scipy.linalg import inv
from sample_functions import n_queens, bm_call, test_django, \
    test_iterative_count, test_list, \
    test_html5libs, fib, test_circle_list


class BenchmarkTest():
    NUM_GENERATIONS = 3

    def __init__(self, f, batch_sizes, n_samples, func_name=""):
        self.f = f
        self.batch_sizes = batch_sizes
        self.n_batches = len(batch_sizes)
        self.n_samples = n_samples
        self.func_name = func_name

    def check_gc(self):
        print("{}\nn_smples={} n_batches={} max_batches={}"
              .format(self.func_name,
                      self.n_samples, self.n_batches, self.batch_sizes[-1]))
        res_without_gc = self.timeit(with_gc=False)
        res_with_gc = self.timeit(with_gc=True)

        fig = self.plot_result(res_without_gc, label="without_gc", c='blue',
                               w="lin_regr")
        fig = self.plot_result(res_with_gc,
                               fig=fig, label="with_gc", c='red',
                               batch_shift=float(self.batch_sizes[1])/8,
                               w="lin_regr")

        plt.show()
        path = self.save_plot(fig)
        print(path)

    def _warm_up(self):
        gc.collect()
        self.f()
        self.f()
        print("warm_up collected {}".format(gc.collect()))

    def timeit(self, with_gc=True):
        self._warm_up()
        gc_disable = gc.disable
        if with_gc:
            gc.disable = lambda: None

        res = np.zeros((n_samples, self.n_batches))
        gc_info = defaultdict(list)

        for sample in range(self.n_samples):
            for i, batch in enumerate(self.batch_sizes):
                gc.collect()
                prev_stats = gc.get_stats()
                res[sample, i] = timeit.Timer(f).timeit(batch)
                diff, is_diff = self.diff_stats(prev_stats, gc.get_stats())
                if with_gc and is_diff and \
                        (batch not in gc_info or
                             not self.diff_equal(gc_info[batch][-1], diff)):
                    gc_info[batch].append(diff)

        gc.disable = gc_disable
        return res, gc_info

    def _plot(self, ax, full_res,
              label="", c='b', s=240, batch_shift=0.,
              w=None, alpha=0.2):
        batch_sizes_ = self.batch_sizes + batch_shift
        res, gc_info = full_res
        mean_res = res.mean(axis=0)
        for res_ in res:
            ax.scatter(batch_sizes_, res_, c=c, s=s, alpha=alpha)
        ax.scatter(0, 0, c=c, label=label)
        ax.plot(batch_sizes_, mean_res,
                c=c, linewidth=2, label="mean")

        if w is "lin_regr":
            X, y = self.get_features(full_res)
            w = self.lin_regression(X, y)
            ax.plot(batch_sizes_, X.dot(w), 'r--', c="dark"+c,
                    label="lin_regr, w={}".format(w), linewidth=2)

        ymin, _ = ax.get_ylim()
        for batch in gc_info:
            text = ""
            for i in range(self.NUM_GENERATIONS):
                if len(gc_info[batch]) > 1:
                    print("diff:: batch {}  {}".format(batch, gc_info[batch]))
                text += "{}\n".format(gc_info[batch][0][i]
                                      .get("collections", 0))
            ax.annotate(text[:-1], (batch + batch_shift, ymin), color=c,
                        size=25)

    def plot_result(self, res, fig=None, n_ax=0, **kwargs):
        """
        :param res: timeit results
        :param func_name:
        :param kwargs:
        :return:
        """
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[n_ax]
        self._plot(ax, res, **kwargs)

        ax.legend()
        ax.set_xlabel('batch_sizes')
        ax.set_ylabel('time')
        ax.grid(True)
        ax.set_title(self.func_name)
        return fig

    def save_plot(self, fig, path=None, dir="img"):
        if path is None:
            dir_ = "{}/{}/".format(dir, self.func_name)
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            path = "{}/{}_{}.jpg".format(dir_, self.n_samples,
                                         np.random.randint(100))
        fig.savefig(path)
        return path

    def get_features(self, full_res):
        res, gc_info = full_res
        if len(gc_info):
            X = np.array([self.batch_sizes,
                 np.zeros(self.n_batches)]).T

            for i, batch in enumerate(batch_sizes):
                if batch in gc_info:
                    X[i][1] = self._sum_collections(gc_info[batch])
        else:
            X = np.array([self.batch_sizes]).T
        y =res.mean(axis=0)
        return X, y

    def lin_regression(self, X, y):
        w = inv(X.T.dot(X)).dot(X.T).dot(y)
        return w

    def _sum_collections(self, gc_stat):
        res = 0
        for gen_info in gc_stat[0]:
            res += gen_info.get("collections", 0)
        return res

    @staticmethod
    def diff_stats(stats0, stats1):
        res = []
        is_diff = False
        for st0, st1 in zip(stats0, stats1):
            res.append({})
            for key in st0.keys():
                diff = st1[key] - st0[key]
                if diff:
                    res[-1][key] = diff
                    is_diff = True
        return res, is_diff

    @staticmethod
    def diff_equal(diff1, diff2):
        for d1, d2 in zip(diff1, diff2):
            for key in set().union(d1.keys()).union(d2.keys()):
                if key not in d1 \
                        or key not in d2 \
                        or d1[key] != d2[key]:
                    print(key, d1, d2)
        return True



if __name__ == "__main__":

    batch_sizes = np.arange(0, 100, 10).astype(np.int)
    n_samples = 10
    func_name = "bm_html5"

    # batch_sizes = np.arange(0, 50000, 5000).astype(np.int)
    # n_samples = 100
    #
    # batch_sizes = np.arange(0, 150000, 5000).astype(np.int)
    # n_samples = 1000
    # func_name = "bm_circle_list"

    f = {
        'bm_ai': functools.partial(n_queens, 8),
        'bm_call_sample': bm_call,
        'bm_django': functools
            .partial(test_django,
                     Context({"table": [range(150) for _ in range(150)]})),
        'bm_iterative_count': test_iterative_count,
        'bm_list': functools.partial(test_list, 100),
        'bm_html5': functools
            .partial(test_html5libs, StringIO(open(
            os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                         "html5lib_spec.html")).read())),
        'bm_fib': functools.partial(fib, 10),
        'bm_circle_list': test_circle_list
    }[func_name]

    BenchmarkTest(f, batch_sizes, n_samples, func_name=func_name).check_gc()

