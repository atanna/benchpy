from IPython import get_ipython
import pylab as plt
from IPython.core.magic import magics_class, Magics, line_cell_magic
from .display import plot_results
from .exception import BenchException


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
                raise BenchException("Table parameters must be "
                               "a subset of set 'ntcsmMrgf'")
            table_dict = dict(n='Name', t='Time', c='CI', s='Std',
                              m='Min', M='Max', r='R2',
                              g='gc_collections', f='fit_info')
            table_keys = map(lambda x: table_dict[x], table_labels)

        def f():
            exec(arg_str, glob)
        from .run import run, bench
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
