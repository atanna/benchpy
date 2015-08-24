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
    table_keys = ['Name', 'Time', 'CI', 'Features_time',
                  'Std', 'Min', 'Max', 'R2',
                  'gc_time', 'Time_without_gc',
                  'fit_info']

    def plot(self, **kwargs):
        return _plot_result(self, **kwargs)

    def plot_features(self, **kwargs):
        return plot_features(self, **kwargs)

    def save_info(self, *args, **kwargs):
        save_info(self, *args, **kwargs)

    def get_table(self, measure='s'):
        w = time_measures[measure]
        fit_info = ""
        for key, value in self.fit_info.items():
            _val = value
            if isinstance(value, list) or isinstance(value, np.ndarray):
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
                     Features_time=self.features_time * w,
                     gc_time=self.gc_time * w,
                     Time_without_gc=(self.time - self.gc_time) * w,
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
                "Features: {}".format(self.feature_names)
                if key == "Features_time" else
                "predicted time without gc" if key == "Time_without_gc"
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
        return table_keys

    def _repr(self, table_keys=None, with_empty=True,
              with_features=True):
        """
        Return representation of class
        :param table_keys: columns of representation table
        string or dict, default ["Name", "Time", "CI"]
        If a string, this may be "Full" or [n][t][c][f][s][m][M][r][g][i]
        (n='Name', t='Time', c='CI', f='Features_time',
         s='Std', m='Min', M='Max', r="R2", g='gc_time',
         i='fit_info')
        Full - all available columns  (='ntcfmMrgi')
        :param with_empty: flag to include/uninclude empty columns
        """
        measure = self.choose_time_measure()
        if table_keys is None:
            table_keys = self.default_table_keys(with_features=with_features)
        elif table_keys == "Full":
            table_keys = self.table_keys
        elif isinstance(table_keys, str):
            if len(set(table_keys) - set('ntcsmMrgfi')):
                raise BenchException("Table parameters must be "
                               "a subset of set 'ntcsmMrgf'")
            table_dict = dict(n='Name', t='Time', c='CI',
                              f='Features_time', s='Std', r='R2',
                              m='Min', M='Max', i='fit_info',
                              g='gc_time')
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
        return pretty_table

    def __repr__(self):
        return "\n" + str(self._repr(with_empty=False))


class VisualMixinGroup(object):
    table_keys = VisualMixin.table_keys

    def plot(self, **kwargs):
        return _plot_group(self, **kwargs)

    def plot_features(self, **kwargs):
        return plot_features(self, **kwargs)

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

    """
    Return plot with time values,
    all parameters of regression from bootstrap and mean
    :param res: BenchResult or GroupResult or list of BenchResult
    :param kwargs:
    """
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
                 save=False, path=None,
                 figsize=(25, 15),
                 group_plot=False,
                 **kwargs):
    if c is None:
        c = np.array([[0], [0.], [0.75]])

    if fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[n_ax]
    batch_shift = 0
    if shift > 0:
        batch_shift = shift * bm_res.batch_sizes[1]
    batch_sizes_ = bm_res.batch_sizes + batch_shift
    measure = bm_res.choose_time_measure()
    w_measure = time_measures[measure]
    for time_ in bm_res.full_time:
        ax.scatter(batch_sizes_, time_*w_measure, c=c, s=s, alpha=alpha)
    ax.scatter([], [], c=c, s=s, alpha=alpha, label=label)

    _alpha = 0.15
    color = 'r'
    if group_plot:
        _alpha = 0.10
        color = mixed_color(c, p=0.3)
    [ax.plot(batch_sizes_, X_y[:,:-1].dot(stat_w)*w_measure,
             color=color, linewidth=linewidth, alpha=_alpha)
     for X_y, stat_w in zip(bm_res.arr_X_y, bm_res.arr_st_w)]

    mean_label = "{}_mean".format(label) if len(label) else "mean"
    ax.plot(batch_sizes_, bm_res.y*w_measure,
            color=c, linewidth=linewidth, label=mean_label)
    if bm_res.stat_w is not None:
        w = bm_res.stat_w.val
        regr_label = "{}_regr, w={}".format(label, w) if len(label) else "regr"
        ax.plot(batch_sizes_, bm_res.X.dot(w)*w_measure, 'r--', color=c,
                linewidth=linewidth,
                label=regr_label)
        ax.legend()
    ax.legend()
    plt.legend()

    if add_text:
        ax.set_xlabel('batch_sizes')
        ax.set_ylabel('time, {}'.format(measure))
        ax.grid(True)
        ax.set_title(title)
    if save:
        save_plot(fig, path=path)
    return fig


def _plot_group(gr_res, labels=None, figsize=(25, 15),
                separate=False, save=False, path=None,
                **kwargs):
    if separate:
        return [res.plot(figsize=figsize, **kwargs) for
                res in gr_res.bench_results]

    list_res = gr_res.bench_results
    n_res = len(gr_res.bench_results)
    if labels is None:
        labels = list(range(n_res))
        for i, res in enumerate(list_res):
            if len(res.name):
                labels[i] = res.name

    batch_shift = 0.15 / n_res
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(111)
    add_text = True
    for i, res, label in zip(range(n_res), list_res, labels):
        d = dict(fig=fig, label=label, title=gr_res.name,
                 c=np.random.rand(3, 1), add_text=add_text,
                 shift=batch_shift * i)
        d.update(kwargs)
        fig = _plot_result(res, group_plot=True,
                           **d)
        add_text = False
    if save:
        save_plot(fig, path=path)
    return fig


def plot_features(bm_res, s=180, alpha=0.4, figsize=(25, 15),
                         save=False, path=None, **kwargs):
    """
    Return plot with every regression feature (parameter).
    """
    from .run import GroupResult
    if isinstance(bm_res, GroupResult):
        if path is None:
            path = "features.jpg"
        name, ext = os.path.splitext(path)
        for i, res in enumerate(bm_res.bench_results):
            plot_features(res, s, alpha, figsize, save,
                                 "{}{}{}".format(name, i, ext))
        return
    batch_sizes = bm_res.batch_sizes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    W = bm_res.stat_w.val
    n_features = len(bm_res.feature_names)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/n_features) for i in range(n_features)]
    measure = bm_res.choose_time_measure()
    w_measure = time_measures[measure]
    for i, x in enumerate(bm_res.X.T):
        c = colors[i]
        w = W[i]
        w_x = w*x*w_measure
        ax.scatter(batch_sizes, w_x, c=c, s=s, alpha=alpha,
                   label="{}  w={}".format(bm_res.feature_names[i], w)
                   .format(i, w))
        ax.plot(batch_sizes, w_x, color=c)

    W_from, W_to = bm_res.stat_w.ci.T
    ax.plot(batch_sizes, bm_res.X.dot(W)*w_measure, color='b', label="regr")
    ax.plot(batch_sizes, bm_res.X.dot(W_from)*w_measure, 'k--', color='b')
    ax.plot(batch_sizes, bm_res.X.dot(W_to)*w_measure, 'k--',
            color='b', label='border_regr')
    ax.plot(batch_sizes, bm_res.y*w_measure, 'bo', color='r', label="mean")
    ax.plot(batch_sizes, bm_res.y*w_measure, color='r')
    ax.legend()
    ax.set_xlabel('batch_sizes')
    ax.set_ylabel('time, {}'.format(measure))
    if save:
        save_plot(fig, path=path)
    return fig


def mixed_color(c0, c1=None, p=0.5):
    if c1 is None:
        c1 = np.array([[1], [0], [0]])
    c = c0*p + c1*(1-p)
    return c / np.sum(c)


def save_plot(fig, path=None, figsize=(25,15)):
    fig.set_size_inches(*figsize)
    if path is None:
        path = "plot.jpg"
    fig.savefig("{}".format(path))
    return path


def save_info(res, path=None, path_suffix="", with_plots=True, plot_params=None):
    """
    Save information about benchmarks and time plots.
    """
    if path is None:
        path = "res_info"
    if plot_params is None:
        plot_params = {}
    if isinstance(res, list):
        for i, _res in enumerate(res):
            save_info(_res, "{}/{}".format(path, i), path_suffix, with_plots)
        return
    if path_suffix:
        path_suffix = "_" + path_suffix
    os.makedirs(path, exist_ok=True)
    info = ""
    from .run import BenchResult
    if isinstance(res, BenchResult):
        n_used_samples = res.__dict__.get('n_used_samples')
        info += "max_batch {}\nn_batches {}\nn_samples {}  {}\n " \
               "with_gc {}\nbatch_sizes: {}\n"\
            .format(res.batch_sizes[-1],
                    res.n_batches,
                    res.n_samples,
                "(used {})".format(n_used_samples) if n_used_samples is not None else '',
                    res.with_gc,
                    res.batch_sizes)
    with open("{}/info{}".format(path, path_suffix), "a") as f:
        f.write("{}\n"\
        .format(res.name.capitalize()))
        f.write(info)
        f.write("X:  {}\n{}\ny:\n{}\n\n"
                .format(res.feature_names, res.X, res.y))
        f.write(str(res._repr(with_features=True)))

    if with_plots:
        features_path = "features{}.jpg".format(path_suffix)
        plot_path = "plot{}.jpg".format(path_suffix)
        res.plot(save=True, path="{}/{}".format(path, plot_path), **plot_params)
        res.plot_features(save=True,
                          path="{}/{}".format(path, features_path), **plot_params)

    with open("{}/report_template.html".format(os.path.split(__file__)[0]), "rt") as f:
        template = f.read()
    with open("{}/report.html".format(path), "at") as f:
        f.write(
            template.format(
                name=res.name.capitalize(),
                features_path=features_path,
                plot_path=plot_path,
                table=res._repr(with_empty=False).get_html_string(format=True)
            )
        )


