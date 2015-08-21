import numpy as np
from collections import namedtuple
from itertools import repeat
from functools import partial
from multiprocessing import Pool
from .analyse import StatMixin
from ._gc_time import get_time
from .display import VisualMixin, VisualMixinGroup
from .exception import BenchException


_Bench = namedtuple("_Bench", 'name f run_params')
_Group = namedtuple("_Group", 'name group run_params')


class Bench(_Bench):
    def run(self, **kwargs):
        return run(self, **kwargs)


class Group(_Group):
    def run(self, **kwargs):
        return run(self, **kwargs)


class BenchResult(StatMixin, VisualMixin):
    pass


class GroupResult(VisualMixinGroup):
    def __init__(self, name, results):
        self.name = name
        self.bench_results = results
        res = results[0]
        self.n_batches = res.n_batches
        self.n_samples = res.n_samples
        self.batch_sizes = res.batch_sizes


def bench(f, *args, run_params=None, func_name="", **kwargs):
    """
    :param f: function which measured
    :param args: args of f
    :param run_params: parameters for benchmark
    :param func_name: function name
    :param kwargs: kwargs of f
    :return: Bench
    """
    if run_params is None:
        run_params = {}
    return Bench(func_name,
                 partial(f, *args, **kwargs),
                 run_params)


def group(name, group, **run_params):
    """
    :param name:
    :param group: list of Benches
    :param run_params:
    :return: Group
    """
    return Group(name, group, run_params)


def run(case, *args, **kwargs):
    """
    Exec benchmark (_run) for each function in case
    See description in _run
    :param case:
    case = list of cases | Bench | Group
    :param args: args for _run
    :param kwargs: kwargs for _run
    :return:
    """
    if isinstance(case, Group):
        return GroupResult(case.name, [run(bench, *args,
                                           **dict(kwargs, **case.run_params))
                                       for bench in case.group])
    elif isinstance(case, Bench):
        params = dict(kwargs)
        params.update(func_name=case.name)
        params.update(case.run_params)
        return _run(case.f, *args, **params)
    elif isinstance(case, list):
        return [run(_case, *args, **kwargs) for _case in case]
    else:
        raise BenchException("Case must be Bench or Group or list")


def _run(f, func_name="",
         n_samples=10, max_batch=100, n_batches=10,
         with_gc=True, with_callback=True,
         multi=True):
    """

    :param f: function which measured (without arguments)
    :param func_name: name of function
    :param n_samples: number of measuring samples (for each batch)
    :param max_batch: maximum of batch size
    :param n_batches: number of batches
    :param with_gc: flag for enable/disable Garbage Collector
    :param with_callback: flag for use/not use callback
     which measure gc working time (use only with gc, python version >= 3.3)
    :param multi: flag for use/not use multiprocessing
    (note: multiprocessing does not work with magic function benchpy)
    :return: BenchResult
    """

    n_batches = min(max_batch, n_batches)
    # batch_sizes are uniformly distributed on [0, max_batch]
    # and sorted in ascending order
    # t.h. batch_sizes[-1] is equal the max_batch
    batch_sizes = \
        np.arange(int(max_batch), 0, -int(max_batch / n_batches))[::-1]
    # min of bath sizes should be 1
    if n_batches > 1:
        batch_sizes[0] = 1
    n_batches = len(batch_sizes)

    full_time = np.zeros((n_samples, n_batches))
    gc_time = np.zeros((n_samples, n_batches))

    for i in range(n_samples):
        if multi:
            with Pool() as p:
                full_time[i], gc_time[i] = \
                    zip(*p.map(get_time, zip(repeat(f),
                                             batch_sizes,
                                             repeat(with_gc),
                                             repeat(with_callback))))

        else:
            full_time[i], gc_time[i] = \
                zip(*list(map(get_time, zip(repeat(f),
                                            batch_sizes,
                                            repeat(with_gc),
                                            repeat(with_callback)))))

    return BenchResult(full_time,
                       gc_time=gc_time,
                       batch_sizes=batch_sizes,
                       with_gc=with_gc,
                       func_name=func_name)





