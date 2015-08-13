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
                  'Time_without_gc', 'fit_info']

    def plot(self, **kwargs):
        return _plot_result(self, **kwargs)

    def show_weight_features(self, **kwargs):
        return show_weight_features(self, **kwargs)

    def save_info(self, *args, **kwargs):
        save_info(self, *args, **kwargs)

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
                     Features_time=self.features_time * w,
                     gc_time=self.gc_time * w,
                     Time_without_gc=(self.time - self.gc_time) * w,
                     gc_predicted_time=self.gc_predicted_time * w,
                     Time_without_gc_pred=self.time_without_gc_pred * w,
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

    def default_table_keys(self, with_gc=None, with_features=True):
        table_keys = ["Name", "Time", "CI"]
        if with_features:
            table_keys.append("Features_time")
        if with_gc is None:
            with_gc = self.with_gc
        if with_gc:
            table_keys.append("gc_time")
            table_keys.append("Time_without_gc")
            table_keys.append("gc_predicted_time")
            table_keys.append("Time_without_gc_pred")
        return table_keys

    def _repr(self, table_keys=None, with_empty=True,
              with_features=True):
        """
        Return representation of class
        :param table_keys: columns of representation table
        string or dict, default ["Name", "Time", "CI"]
        If a string, this may be "Full" or [n][t][c][s][m][M][r][g][f]
        (n='Name', t='Time', c='CI', s='Std', m='Min', M='Max', r='R2',
        f='fit_info')
        Full - all available columns  (='ntcsmMrgf')
        :param with_empty: flag to include/uninclude empty columns
        """
        measure = self.choose_time_measure()
        if table_keys is None:
            table_keys = self.default_table_keys(with_features=with_features)
        elif table_keys == "Full":
            table_keys = self.table_keys
        elif isinstance(table_keys, str):
            if len(set(table_keys) - set('ntcsmMrgf')):
                raise BenchException("Table parameters must be "
                               "a subset of set 'ntcsmMrgf'")
            table_dict = dict(n='Name', t='Time', c='CI', s='Std',
                              m='Min', M='Max', r='R2',
                              f='fit_info')
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

    def _repr(self, table_keys=None, with_empty=True, with_features=True):
        first_res = self.bench_results[0]
        if table_keys is None:
            table_keys = \
                first_res.default_table_keys(with_features=with_features)
        elif table_keys is "Full":
            table_keys = self.table_keys
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
                 title="", s=180, shift=0., alpha=0.2,
                 linewidth=2, add_text=True,
                 save=False, path=None):
    if c is None:
        c = np.array([[0], [0.], [0.75]])

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[n_ax]

    batch_shift = shift * bm_res.batch_sizes[1]
    batch_sizes_ = bm_res.batch_sizes + batch_shift

    for res_ in bm_res.res[:,:,0]:
        ax.scatter(batch_sizes_, res_, c=c, s=s, alpha=alpha)
    ax.scatter(0, 0, c=c, label=label)
    mean_label = "{}_mean".format(label) if len(label) else "mean"
    ax.plot(batch_sizes_, bm_res.y,
            c='r', linewidth=linewidth, label=mean_label)

    ax.legend()
    if bm_res.stat_w is not None:
        w = bm_res.stat_w.val
        regr_label = "{}_regr, w={}".format(label, w) if len(label) else "regr"
        ax.plot(batch_sizes_, bm_res.X.dot(w), 'r--', c='r',
                linewidth=linewidth,
                label=regr_label)
        ax.legend()
    ax.legend()
    plt.legend()

    if add_text:
        ax.set_xlabel('batch_sizes')
        ax.set_ylabel('time')
        ax.grid(True)
        ax.set_title(title)
    if save:
        save_plot(fig, path=path)
    return fig


def show_weight_features(bm_res, s=180, alpha=0.4, save=False, path=None):
    batch_sizes = bm_res.batch_sizes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    W = bm_res.stat_w.val
    n_features = len(bm_res.features)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/n_features) for i in range(n_features)]
    for i, x in enumerate(bm_res.X.T):
        c = colors[i]
        w = W[i]
        w_x = w*x
        ax.scatter(batch_sizes, w_x, c=c, s=s, alpha=alpha,
                   label="{}  w={}".format(bm_res.features[i], w).format(i, w))
        ax.plot(batch_sizes, w_x, c=c)

    W_from, W_to = bm_res.stat_w.ci.T
    ax.plot(batch_sizes, bm_res.X.dot(W), c='b', label="regr")
    ax.plot(batch_sizes, bm_res.X.dot(W_from), 'k--', c='b')
    ax.plot(batch_sizes, bm_res.X.dot(W_to), 'k--',
            c='b', label='border_regr')
    ax.plot(batch_sizes, bm_res.y, 'bo', c='r', label="y")
    ax.plot(batch_sizes, bm_res.y, c='r')
    ax.legend()
    ax.set_xlabel('batch_sizes')
    ax.set_ylabel('time')
    if save:
        save_plot(fig, path=path)
    return fig


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
    return np.minimum(np.maximum(color - alpha, 0), 1.)


def save_plot(fig, path=None, figsize=(25,15)):
    fig.set_size_inches(*figsize)
    if path is None:
        path="plot.jpg"
    fig.savefig("{}".format(path))
    return path


def save_info(res, path=None, path_suffix="", with_plots=True):
    if path is None:
        path = "res_info"
    if path_suffix:
        path_suffix = "_" + path_suffix
    os.makedirs(path, exist_ok=True)
    with open("{}/info{}".format(path, path_suffix), "a") as f:
        f.write("{}\n".format(res.name.capitalize()))
        f.write("max_batch {}\nn_batches {}\nn_samples {}\nwith_gc {}\n"
                .format(res.batch_sizes[-1], res.n_batches, res.n_samples,
                        res.with_gc))
        f.write("X:  {}\n{}\ny:\n{}\n".format(res.features, res.X, res.y))
        f.write(res._repr(with_features=True))
        f.write("\n\n")

    if with_plots:
        res.plot(save=True, path="{}/plot{}.jpg".format(path, path_suffix))
        res.show_weight_features(save=True,
                                 path="{}/features{}.jpg"
                                 .format(path, path_suffix))


