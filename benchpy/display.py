import os
import numpy as np
import pylab as plt
from collections import OrderedDict
from IPython import get_ipython
from IPython.core.magic import magics_class, Magics, line_cell_magic
from .analyse import BenchResult, GroupResult
from .run import run, bench
from . import BenchException

time_measures = OrderedDict(zip(['s', 'ms', 'µs', 'ns'],
                                [1, 1e3, 1e6, 1e9]))


def plot_results(res, **kwargs):
    if isinstance(res, BenchResult):
        return _plot_result(res, **kwargs)
    elif isinstance(res, GroupResult):
        return _plot_group(res, **kwargs)
    elif type(res) == list:
        return [plot_results(_res) for _res in res]
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
        ax.set_ylabel('benchtime')
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


@magics_class
class ExecutionMagics(Magics):

    @line_cell_magic
    def benchpy(self, parameter_s='', cell=None):
        """
        Run benchpy.run
        %benchpy [[-i] -g<G> -m<M> -n<N> [-p] -s<S>] statement
        where statement is Bench or Group or list with benches
        %%benchpy [[-i] -g<G> -m<M> -n<N> [-p] -s<S>]
          long description of statement


        Options:
        -i: return full information about benchmark results.

        -g: use information from garbage collector (with_gc=True).
        Default: 'False'.

        -n<N>: set maximum of batch size <N> (max_batch=<N>).
        Default: 100.

        -m<M>: set number of batches for fitting regression <M> (n_batches=<M>).
        Default: 5.
        batch_sizes = [1, 1+<N>/<M>, 1+2<N>/<M>, ...]

        -p: show plot with regression.

        -r<R>: repeat the loop iteration <R> (n_samples=<R>).
        Default 5.

        -t<T>: choose columns <T> to represent result.
        <T> = [n][t][c][s][m][M][r][g][f]
        (n='Name', t='Time', c='CI', s='Std', m='Min', M='Max', r='R2',
        g='gc_collections', f='fit_info')

        Default - default in repr.

        Examples
        --------
        ::
            In [1]: import benchpy as bp

            In [2]: def f(a, b): return a + b

            In [3]: %benchpy f(69, 96)

            +--------------+---------------------------+
            |  Time (µs)   |      CI_tquant[0.95]      |
            +--------------+---------------------------+
            | 8.9438093954 | [ 8.30582379  9.22444996] |
            +--------------+---------------------------+

            In [4]: %benchpy -t tcsr f(69, 96)

            +---------------+---------------------------+----------------+----------------+
            |   Time (µs)   |      CI_tquant[0.95]      |      Std       |       R2       |
            +---------------+---------------------------+----------------+----------------+
            | 9.13863231924 | [ 7.59629832  9.79898779] | 0.589640927153 | 0.999972495022 |
            +---------------+---------------------------+----------------+----------------+

            In [5]: def cycles(n):
                      for i in range(n):
                        arr = []
                        arr.append(arr)

            In [6]: %benchpy -g cycles(100)

            +---------------+-----------------------------+----------------+
            |   Time (µs)   |       CI_tquant[0.95]       | gc_collections |
            +---------------+-----------------------------+----------------+
            | 29.4959616407 | [ 28.64601155  30.4106039 ] | 0.117333333333 |
            +---------------+-----------------------------+----------------+

            In [7]: %benchpy  cycles(100)

            +---------------+-----------------------------+
            |   Time (µs)   |       CI_tquant[0.95]       |
            +---------------+-----------------------------+
            | 20.3202796031 | [ 19.40618596  20.80962796] |
            +---------------+-----------------------------+


        """
        opts, arg_str = self.parse_options(parameter_s, 'igm:n:pr:t:',
                                           list_all=True, posix=False)
        glob = dict(self.shell.user_ns)
        if cell is not None:
            arg_str += '\n' + cell
            arg_str = self.shell.input_transformer_manager.transform_cell(cell)
        with_gc = 'g' in opts
        n_samples = opts.get('r', [5])[0]
        max_batch = opts.get('n', [100])[0]
        n_batches = min(int(max_batch), int(opts.get('m', [5])[0]))
        table_keys = None
        table_labels = opts.get('t', [None])[0]
        if table_labels is not None:
            if len(set(table_labels) - set('ntcsmMrgf')):
                BenchException("Table parameters must be "
                            "a subset of set 'ntcsmMrgf'")
            table_dict = dict(n='Name', t='Time', c='CI', s='Std',
                              m='Min', M='Max', r='R2',
                              g='gc_collections', f='fit_info')
            table_keys = map(lambda x: table_dict[x], table_labels)

        def f():
            exec(arg_str, glob)

        res = eval("run(bench(f), "
                   "with_gc={with_gc}, "
                   "n_samples={n_samples}, "
                   "max_batch={max_batch}, "
                   "n_batches={n_batches})"
                   .format(with_gc=with_gc,
                           n_samples=n_samples,
                           n_batches=n_batches,
                           max_batch=max_batch))
        if 'p' in opts:
            plot_results(res)
            plt.show()

        if 'i' in opts:
            return res._repr("Full")
        print(res._repr(table_keys, with_empty=False))


ip = get_ipython()
if ip is not None:
    ip.register_magics(ExecutionMagics)

