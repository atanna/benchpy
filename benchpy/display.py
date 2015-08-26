# -*- coding: utf-8 -*-

from __future__ import absolute_import
from functools import reduce

import os
from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from .utils import cached_property

from .exceptions import BenchException

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

    def get_tables(self, *args, **kwargs):
        return [self.get_table(*args, **kwargs)]

    def get_nice_fit_info(self, dict_fit_info):
        fit_info = ""
        for key, value in dict_fit_info.items():
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
        return fit_info[:-1]

    def get_table(self, measure=None, decimals=5):
        if measure is None:
            measure = self.time_measure
        w = time_measures[measure]

        table = dict(map(lambda x: (x[0], np.round(x[1], decimals))
        if x[0] == "R2"
        else (x[0], np.round(x[1] * w, decimals))
        if isinstance(x[1], float) or isinstance(x[1], np.ndarray)
        else (x[0],x[1]), self.get_stat_table().items()))

        table["fit_info"] = self.get_nice_fit_info(table["fit_info"])

        return table

    @cached_property
    def time_measure(self):
        t = self.time
        c = 10
        for measure, w in time_measures.items():
            if int(t * w * c):
                return measure

    def _get_pretty_table_header(self, measure=None, table_keys=None):
        if table_keys is None:
            table_keys = self.table_keys

        if measure is None:
            measure = self.time_measure
        pretty_table = PrettyTable(list(
            map(lambda key:
                "CI[{}]"
                .format(self.confidence) if key == "CI" else
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
        measure = self.time_measure
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

    def save_info(self, *args, **kwargs):
        save_info(self, *args, **kwargs)

    def _repr(self, table_keys=None, with_features=True):
        first_res = self.results[0]
        while not isinstance(first_res, VisualMixin):
            first_res = first_res.results[0]
        if table_keys is None:
            table_keys = \
                first_res.default_table_keys(with_features=with_features)
        elif table_keys is "Full":
            table_keys = self.table_keys
        measure = first_res.time_measure
        tables = self.get_tables(measure)

        pretty_table = first_res._get_pretty_table_header(measure, table_keys)
        pretty_table.align = 'l'
        for table in tables:
            pretty_table.add_row([table[key] for key in table_keys])

        return str(pretty_table)

    def get_tables(self, measure, decimals=5):
        tables = [bm_res.get_table(measure=measure, decimals=decimals)
                  for bm_res in self.batch_results]
        return tables

    def get_list_batch_results(self, prefix_name=""):
        list_results = []
        for res in self.results:
            _prefix_name = "{}{}".format(prefix_name, '.' if len(prefix_name) else '')
            if isinstance(res, VisualMixin):
                name = "{}{}".format(_prefix_name, res.name)
                res.name = name
                list_results.append(res)
            else:
                list_results += res.get_list_batch_results(
                    "{}{}".format(_prefix_name, self.name))
        return list_results

    def __repr__(self):
        return self._repr()


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
        raise BenchException("Type of 'res' must belong to "
                             "{BenchRes, GroupRes, list}")


def _plot_result(bm_res, fig=None, n_ax=0, label="", c=None,
                 title="", s=180, shift=0., alpha=0.2,
                 linewidth=2, add_text=True,
                 save=False, path=None,
                 figsize=(25, 15),
                 fontsize=16,
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
    measure = bm_res.time_measure
    w_measure = time_measures[measure]
    X, y, stat_w = bm_res.info_to_plot()
    for time_ in bm_res.full_time:
        ax.scatter(batch_sizes_, time_*w_measure, color=c, s=s, alpha=alpha)
    ax.scatter([], [], color=c, s=s, alpha=alpha, label=label+"time")

    used_t_color = mixed_color(c, np.array([[0], [1], [0]]), 0.1)
    for time_ in bm_res.y:
        ax.scatter(batch_sizes_, time_*w_measure, marker="*",
                   color=used_t_color, s=s/5)
    ax.scatter([], [], marker="*", color=used_t_color, s=s/5,
               label=label+"used time")

    _alpha = 0.15
    color = 'r'
    if group_plot:
        _alpha = 0.10
        color = mixed_color(c, p=0.3)
    [ax.plot(batch_sizes_, X_y[:,:-1].dot(stat_w)*w_measure,
             color=color, linewidth=linewidth, alpha=_alpha)
     for X_y, stat_w in zip(bm_res.arr_X_y, bm_res.arr_st_w)]

    mean_label = "{}_mean".format(label) if len(label) else "mean"
    ax.plot(batch_sizes_, y*w_measure,
            color=c, linewidth=linewidth, label=mean_label)

    if stat_w is not None:
        w = stat_w.mean

        regr_label = "{}_regr, w={}".format(label, np.round(w, 5)) \
            if len(label) else "regr"
        ax.plot(batch_sizes_, X.dot(w)*w_measure, 'r--', color=c,
                linewidth=linewidth,
                label=regr_label)
    ax.legend(fontsize=fontsize)

    if add_text:
        ax.set_xlabel('Batch_sizes', fontsize=fontsize)
        ax.set_ylabel('Time, {}'.format(measure), fontsize=fontsize)
        ax.grid(True)
        if not len(title):
            title = "Bootstrap regression"
        ax.set_title(title, fontsize=fontsize*1.4)
    if save:
        save_plot(fig, path=path, figsize=figsize)
    return fig


def _plot_group(gr_res, labels=None, figsize=(25, 15),
                separate=False, save=False, path=None,
                **kwargs):
    if separate:
        return [res.plot(figsize=figsize, save=save,separate=separate,
                         **kwargs) for
                res in gr_res.results]

    list_res = gr_res.batch_results
    n_res = len(gr_res.results)
    if labels is None:
        labels = list(range(n_res))
        for i, res in enumerate(list_res):
            if len(res.name):
                labels[i] = res.name

    batch_shift = 0.15 / n_res
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(111)
    add_text = True
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/n_res) for i in range(n_res)]
    for i, res, label in zip(range(n_res), list_res, labels):
        d = dict(fig=fig, label=label, title=gr_res.name,
                 c=colors[i], add_text=add_text,
                 shift=batch_shift * i)
        d.update(kwargs)
        fig = _plot_result(res, group_plot=True,
                           **d)
        add_text = False
    if save:
        save_plot(fig, path=path, figsize=figsize)
    return fig


def plot_features(bm_res, s=180, alpha=0.4,
                  figsize=(25, 15), fontsize=16,
                  save=False, path=None, **kwargs):
    """
    Return plot with every regression feature (parameter).
    """
    from .run import GroupResult
    if isinstance(bm_res, GroupResult):
        if path is None:
            path = "features.jpg"
        name, ext = os.path.splitext(path)
        for i, res in enumerate(bm_res.results):
            plot_features(res, s=s, alpha=alpha,
                          figsize=figsize, fontsize=fontsize,
                          save=save,
                          path="{}{}{}".format(name, i, ext), **kwargs)
        return
    batch_sizes = bm_res.batch_sizes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    X, y, stat_w = bm_res.info_to_plot()
    W = stat_w.mean

    n_features = len(bm_res.feature_names)
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1.*i/n_features) for i in range(n_features)]
    measure = bm_res.time_measure
    w_measure = time_measures[measure]
    for i, x in enumerate(X.T):
        c = colors[i]
        w = W[i]
        w_x = w*x*w_measure
        ax.scatter(batch_sizes, w_x, c=c, s=s, alpha=alpha,
                   label="{}  w={}".format(bm_res.feature_names[i],
                                           np.round(w*w_measure, 5))
                   .format(i, w))
        ax.plot(batch_sizes, w_x, color=c)

    W_from, W_to = stat_w.ci.T
    ax.plot(batch_sizes, X.dot(W)*w_measure, color='b', label="regr")
    ax.plot(batch_sizes, X.dot(W_from)*w_measure, 'k--', color='b')
    ax.plot(batch_sizes, X.dot(W_to)*w_measure, 'k--',
            color='b', label='border_regr')
    ax.plot(batch_sizes, y*w_measure, 'bo', color='r', label="mean")
    ax.plot(batch_sizes, y*w_measure, color='r')
    ax.legend(fontsize=fontsize)
    ax.set_xlabel('Batch_sizes', fontsize=fontsize)
    ax.set_ylabel('Time, {}'.format(measure), fontsize=fontsize)
    ax.set_title("Regression parameters", fontsize=fontsize*1.4)
    if save:
        save_plot(fig, path=path, figsize=figsize)
    return fig


def mixed_color(c0, c1=None, p=0.5):
    if c1 is None:
        c1 = np.array([[1], [0], [0]])
    if len(c0) == 4:
        c0 = np.array(c0[:-1])[:, np.newaxis]
    c = c0*p + c1*(1-p)
    return c / np.sum(c)


def save_plot(fig, path=None, figsize=(25, 15)):
    fig.set_size_inches(*figsize)
    if path is None:
        path = "plot.jpg"
    fig.savefig("{}".format(path))
    return path


def save_info(res, path=None, path_suffix="", with_plots=True,
              plot_params=None, prefix_name="",
              figsize=(20, 12), fontsize=18):
    """
    Save information about benchmarks and time plots.
    """
    if path is None:
        path = "res_info"
    from .run import GroupResult, BenchResult
    results = []
    if isinstance(res, GroupResult):
        results = res.batch_results
    if isinstance(res, list):
        results = res

    for i, _res in enumerate(results):
        save_info(_res, path=path, path_suffix="~name",
                  with_plots=with_plots,
                  plot_params=plot_params,
                  figsize=figsize, fontsize=fontsize)
    if len(results):
        return

    if not isinstance(res, BenchResult):
        raise BenchException("Type of 'res' must belong to "
                             "{BenchRes, GroupRes, list}")
    if path_suffix == "~name":
        path_suffix = res.name
    if plot_params is None:
        plot_params = {}
    if isinstance(res, list):
        for i, _res in enumerate(res):
            save_info(_res, "{}/{}".format(path, i), path_suffix, with_plots)
        return
    if path_suffix:
        path_suffix = "_" + path_suffix
    os.makedirs(path, exist_ok=True)

    n_used_samples = res.__dict__.get('n_used_samples')
    info = "max_batch {}\nn_batches {}\nn_samples {}  {}\n " \
            "with_gc {}\nbatch_sizes: {}\n" \
        .format(res.batch_sizes[-1],
                res.n_batches,
                res.n_samples,
                "(used {})".format(n_used_samples) if
                           n_used_samples is not None else '',
                res.with_gc,
                res.batch_sizes)
    with open("{}/info{}".format(path, path_suffix), "a") as f:
        f.write("{}\n" \
                .format(res.name.capitalize()))
        f.write(info)
        if isinstance(res, BenchResult):
            f.write("X:  {}\n{}\ny:\n{}\n\n"
                    .format(res.feature_names, res.X, res.y))
        f.write(str(res._repr(with_features=True)))

    if with_plots:
        features_path = "features{}.jpg".format(path_suffix)
        plot_path = "plot{}.jpg".format(path_suffix)
        res.plot(save=True, path="{}/{}".format(path, plot_path),
                 figsize=figsize, fontsize=fontsize, **plot_params)
        res.plot_features(save=True,
                          path="{}/{}".format(path, features_path),
                          figsize=figsize, fontsize=fontsize, **plot_params)

    with open("{}/report_template.html"
                      .format(os.path.split(__file__)[0]), "rt") as f:
        template = f.read()

    table = res.get_table()
    feature_table = PrettyTable(list(res.feature_names))
    feature_table.add_row(table["Features_time"])


    with open("{}/report.html".format(path), "at") as f:
        f.write(
            template.format(
                name=res.name.capitalize(),
                features_path=features_path,
                plot_path=plot_path,
                time_measure=res.time_measure,
                time=table["Time"],
                ci_0=table["CI"][0],
                ci_1=table["CI"][1],
                feature_table=
                feature_table.get_html_string(format=True,
                                              right_padding_width=2)
                    .replace("th", "td"),
                gc_time=table["gc_time"],
                t_without_gc=table["Time_without_gc"]
            )
        )
