import os
import numpy as np
import pylab as plt
from collections import OrderedDict
from prettytable import PrettyTable
from .exception import BenchException

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))


class VisualMixin(object):
    """
    Used only with StatMixin.
    """
    table_keys = ['Name', 'Time', 'CI', 'Std', 'Min', 'Max', 'R2',
                  'gc_collections', 'Time_without_gc', 'fit_info']

    def plot(self, **kwargs):
        _plot_result(self, **kwargs)

    def show_features(self, **kwargs):
        show_weight_features(self, **kwargs)

    def get_table(self, measure='s'):
        w = time_measures[measure]
        fit_info = ""
        for key, value in self.fit_info.items():
            _val = value
            if isinstance(value, list):
                n = len(value)
                if n > 2:
                    _val = "[{}, {}, ... ,{}]". \
                        format(value[0], value[1], value[-1])
                else:
                    _val = value
            info = "{}: {}\n".format(key, _val)
            fit_info += info
        fit_info = fit_info[:-1]

        table = dict(Name=self.name,
                     Time=self.time * w,
                     CI=self.ci * w,
                     Std=self.std * w,
                     Min=self.min * w,
                     Max=self.max * w,
                     R2=self.r2,
                     gc_collections=self.gc_collections * w,
                     Features_time=self.features_time * w,
                     gc_time=self.gc_time * w,
                     Time_without_gc=(self.time - self.gc_time) * w,
                     gc_predicted_time=self.gc_predicted_time * w,
                     Time_without_gc_pred=self.time_without_gc * w,
                     fit_info=fit_info)
        return table

    def choose_time_measure(self, perm_n_points=0):
        t = self.time
        c = 10 ** perm_n_points
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
                "Time ({})".format(measure) if key == "Time" else
                "Features: {}".format(self.features) if key == "Features_time"
                else key,
                table_keys)))
        return pretty_table

    def _repr(self, table_keys=None, with_empty=True,
              with_features=False):
        """
        Return representation of class
        :param table_keys: columns of representation table
        string or dict, default ["Name", "Time", "CI"]
        If a string, this may be "Full" or [n][t][c][s][m][M][r][g][f]
        (n='Name', t='Time', c='CI', s='Std', m='Min', M='Max', r='R2',
        g='gc_collections', f='fit_info')
        Full - all available columns  (='ntcsmMrgf')
        :param with_empty: flag to include/uninclude empty columns
        """
        measure = self.choose_time_measure()
        if table_keys is None:
            table_keys = ["Name", "Time", "CI"]
            if with_features:
                table_keys.append("Features_time")
            if self.with_gc:

                table_keys.append("gc_time")
                table_keys.append("Time_without_gc")
                table_keys.append("gc_predicted_time")
                table_keys.append("Time_without_gc_pred")
        elif table_keys == "Full":
            table_keys = self.table_keys
        elif isinstance(table_keys, str):
            if len(set(table_keys) - set('ntcsmMrgf')):
                raise BenchException("Table parameters must be "
                               "a subset of set 'ntcsmMrgf'")
            table_dict = dict(n='Name', t='Time', c='CI', s='Std',
                              m='Min', M='Max', r='R2',
                              g='gc_collections', f='fit_info')
            table_keys = map(lambda x: table_dict[x], table_keys)
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


class VisualMixinGroup(object):
    table_keys = VisualMixin.table_keys

    def plot(self):
        _plot_group(self)

    def _repr(self, table_keys=None, with_empty=True):
        if table_keys is None:
            table_keys = ["Name", "Time", "CI"]
            for bm_res in self.bench_results:
                if bm_res.with_gc:
                    table_keys.append("gc_collections")
                    table_keys.append("Time_without_gc")
                    break
        elif table_keys is "Full":
            table_keys = self.table_keys
        first_res = self.bench_results[0]
        measure = first_res.choose_time_measure()
        tables = [bm_res.get_table(measure) for bm_res in self.bench_results]
        n_results = len(self.bench_results)
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
        for bm_res in self.bench_results:
            table = bm_res.get_table(measure)
            pretty_table.add_row([table[key] for key in _table_keys])
        title = "\n{group:~^{n}}\n" \
            .format(group=self.name, n=10)
        return title + str(pretty_table)

    def __repr__(self):
        return self._repr(with_empty=False)


def plot_results(res, **kwargs):
    from .run import BenchResult, GroupResult
    if isinstance(res, BenchResult):
        return _plot_result(res, **kwargs)
    elif isinstance(res, GroupResult):
        return _plot_group(res, **kwargs)
    elif type(res) == list:
        return [plot_results(_res, **kwargs) for _res in res]
    else:
        raise BenchException("res must be BenchRes or GroupRes or list")


def _plot_result(bm_res, fig=None, n_ax=0, label="", c=None,
                 title="", s=240, shift=0., alpha=0.2, text_size=20,
                 linewidth=2, add_text=True):
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
    ax.plot(batch_sizes_, bm_res.y,
            c=c, linewidth=linewidth, label="{}_mean".format(label))

    if bm_res.regr is not None:
        w = bm_res.regr.stat_w.val
        ax.plot(batch_sizes_, bm_res.regr.X.dot(w), 'r--', c=_dark_color(c),
                label="{}_lin_regr, w={}".format(label, w),
                linewidth=linewidth)

    ax.legend()
    if add_text:

        ax.set_xlabel('batch_sizes')
        ax.set_ylabel('time')
        ax.grid(True)
        ax.set_title(title)
    return fig


def show_weight_features(bm_res, s=180, alpha=0.4):
    batch_sizes = bm_res.batch_sizes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    W = bm_res.regr.stat_w.val
    for i, x in enumerate(bm_res.X.T):
        c = np.random.rand(3, 1)
        w = W[i]
        w_x = w*x
        ax.scatter(batch_sizes, w_x, c=c, s=s, alpha=alpha,
                   label="{}  w={}".format(bm_res.features[i], w).format(i, w))
        ax.plot(batch_sizes, w_x, c=c)

    W_from, W_to = bm_res.regr.stat_w.ci.T
    ax.plot(batch_sizes, bm_res.X.dot(W), c='b', label="regr")
    ax.plot(batch_sizes, bm_res.X.dot(W_from), 'k--', c='b')
    ax.plot(batch_sizes, bm_res.X.dot(W_to), 'k--',
            c='b', label='border_regr')
    ax.plot(batch_sizes, bm_res.y, 'bo', c='r', label="y")
    ax.plot(batch_sizes, bm_res.y, c='r')
    ax.legend()
    ax.set_xlabel('batch_sizes')
    ax.set_ylabel('time')


def _plot_group(gr_res, labels=None, **kwargs):
    list_res = gr_res.bench_results
    n_res = len(gr_res.bench_results)
    if labels is None:
        labels = list(range(n_res))
        for i, res in enumerate(list_res):
            if len(res.func_name):
                labels[i] = res.func_name

    batch_shift = 0.15 / n_res
    fig = plt.figure()
    fig.add_subplot(111)
    add_text = True
    for i, res, label in zip(range(n_res), list_res, labels):
        d = dict(fig=fig, label=label, title=gr_res.name,
                 c=np.random.rand(3, 1), add_text=add_text,
                 shift=batch_shift * i)
        d.update(kwargs)
        fig = _plot_result(res, **d)
        add_text = False
    return fig


def _dark_color(color, alpha=0.1):
    return np.maximum(color - alpha, 0)


def save_plot(fig, func_name="f", path=None, dir="img"):
    if path is None:
        dir_ = "{}/{}/".format(dir, func_name)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        path = "{}/{}.jpg".format(dir_, np.random.randint(100))
    fig.savefig(path)
    return path

