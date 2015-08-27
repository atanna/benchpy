# -*- coding: utf-8 -*-

from __future__ import absolute_import

from functools import partial

from matplotlib import pyplot as plt


def magic_benchpy(line='', cell=None):
    """
    Run benchpy.run
    %benchpy [[-i] [-g] [-n <N>] [-m <M>] [-p] [-r <R>] [-t <T>] -s<S>] statement
    where statement is Bench or Group or list with benches
    %%benchpy [[-i] -g<G> -m<M> -n<N> [-p] -s<S>]
      long description of statement


    Options:
    -i: return full information about benchmark results.

    -g: use information from garbage collector (with_gc=True).
    Default: 'False'.

    -n<N>: set maximum of batch size <N> (max_batch=<N>).
    Default: 10.

    -m<M>: set number of batches for fitting regression <M> (n_batches=<M>).
    Default: 10.
    batch_sizes = [1, ...,M-2<M>/<N>, M-<M>/<N>, <M>]

    -p: show plots with regression.

    -r<R>: repeat the loop iteration <R> (n_samples=<R>).
    Default 5.

    -t<T>: choose columns <T> to represent result.
    <T> = [t][c][f][s][m][M][r][g][i]
    where
    t='Time'
    c='CI' - confidence interval
    f='Features_time' - time for each regression parameter
    s='Std' - standard deviation for regression parameter (which means time)
    m='Min' - minimum of the time values
    M='Max' - maximum
    r="R2" - r2 regression score
    g='gc_time' - time for gc collections (useful only with python version >= 3.3)
    i='fit_info' - fitting information

    Default - default in repr.

    Examples
    --------
    ::
    In [1]: import benchpy as bp

    In [2]: %benchpy 10**10000
    +--------------+-------------------------------+-------------------------------+
    |  Time (µs)   |        CI_tquant[0.95]        |  Features: ['batch' 'const']  |
    +--------------+-------------------------------+-------------------------------+
    | 225.33965124 | [ 210.72239262  239.54741751] | [ 177.29140495   48.04824629] |
    +--------------+-------------------------------+-------------------------------+

    In [3]: %benchpy -t tcsrmM 10**10000
    +---------------+-------------------------------+---------------+----------------+---------------+---------------+
    |   Time (µs)   |        CI_tquant[0.95]        |      Std      |       R2       |      Min      |      Max      |
    +---------------+-------------------------------+---------------+----------------+---------------+---------------+
    | 226.600298929 | [ 213.60009798  240.16961693] | 7.00210625405 | 0.999999184569 | 179.693800055 | 226.248999752 |
    +---------------+-------------------------------+---------------+----------------+---------------+---------------+

    In [4]: def cycles(n):
       ...:     for i in range(n):
       ...:         arr = []
       ...:         arr.append(arr)
       ...:

    In [9]: %benchpy -n 1000 cycles(100)
    +---------------+-----------------------------+-----------------------------+
    |   Time (µs)   |       CI_tquant[0.95]       | Features: ['batch' 'const'] |
    +---------------+-----------------------------+-----------------------------+
    | 23.3943861198 | [  0.          25.96065552] | [ 20.87035101   2.52403511] |
    +---------------+-----------------------------+-----------------------------+

    In [10]: %benchpy -n 1000 -g cycles(100)
    +--------------+-----------------------------+-----------------------------+---------------+---------------------------+
    |  Time (µs)   |       CI_tquant[0.95]       | Features: ['batch' 'const'] |    gc_time    | predicted time without gc |
    +--------------+-----------------------------+-----------------------------+---------------+---------------------------+
    | 64.256959342 | [  0.          99.92966164] | [ 28.80691753  35.45004181] | 7.67428691294 |        56.582672429       |
    +--------------+-----------------------------+-----------------------------+---------------+---------------------------+

    """
    from IPython import get_ipython
    from IPython.core.magics import UserMagics

    ip = get_ipython()

    opts, arg_str = UserMagics(ip).parse_options(
        line, 'igm:n:pr:t:', list_all=True, posix=False)

    if cell is not None:
        arg_str += '\n' + cell
        arg_str = ip.input_transformer_manager.transform_cell(cell)
    with_gc = 'g' in opts
    n_samples = int(opts.get('r', [5])[0])
    max_batch = int(opts.get('n', [10])[0])
    n_batches = min(int(max_batch), int(opts.get('m', [10])[0]))
    table_keys = None
    table_labels = opts.get('t', [None])[0]
    if table_labels is not None:
        table_keys = table_labels
    f = partial(exec, arg_str, ip.user_ns)

    from . import run, bench
    res = run(bench("<magic>", f), with_gc=with_gc,
              n_samples=n_samples,
              n_batches=n_batches,
              max_batch=max_batch)

    if 'i' in opts:
        print(res._repr("Full"))
    else:
        print(res._repr(table_keys, with_empty=False))

    if 'p' in opts:
        res.plot()
        res.plot_features()
        plt.show()


def load_ipython_extension(ip):
    """API for IPython to recognize this module as an IPython extension."""
    ip.register_magic_function(magic_benchpy, "line_cell", magic_name="benchpy")
