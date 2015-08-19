import numpy as np
from collections import namedtuple
from itertools import repeat
from functools import partial
from multiprocessing import Pool
from .analyse import StatMixin
from ._gc_time import GC_NUM_GENERATIONS, get_time
from ._mem import get_mem
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
        self.results = results
        res = results[0]
        self.n_samples = res.n_samples
        self.batch_sizes = res.batch_sizes
        self.n_batches = res.n_batches

    @property
    def bench_results(self):
        return self.results


def bench(f, *args, run_params=None, func_name="", **kwargs):
    if run_params is None:
        run_params = {}
    return Bench(func_name,
                 partial(f, *args, **kwargs),
                 run_params)


def group(name, group, **run_params):
    return Group(name, group, run_params)


def run(case, *args, **kwargs):
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


def _run(f, n_samples=10, max_batch=100, n_batches=10, with_gc=True,
         func_name="", multi=True):
    """
    :param f: function without arguments
    :param n_samples: number of samples
    :param max_batch: maximum of batch size
    :param n_batches: number of batches
    :param with_gc: {True, False} Garbage Collector
    :param func_name:
    :return:
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

    mem = np.zeros((n_samples, n_batches, 2))
    full_time = np.zeros((n_samples, n_batches))
    gc_time = np.zeros((n_samples, n_batches))
    gc_collected = np.zeros((n_samples, n_batches, GC_NUM_GENERATIONS))

    for i in range(n_samples):
        if multi:
            with Pool() as p:
                full_time[i], gc_time[i], gc_collected[i] = \
                    zip(*p.map(get_time, zip(repeat(f), batch_sizes, repeat(with_gc))))
                mem[i] = p.map(get_mem, zip(repeat(f), batch_sizes, repeat(with_gc)))

        else:
            full_time[i], gc_time[i], gc_collected[i] = \
                zip(*list(map(get_time, zip(repeat(f), batch_sizes, repeat(with_gc)))))
            mem[i] = list(map(get_mem, zip(repeat(f), batch_sizes, repeat(with_gc))))

    return BenchResult(full_time, gc_time=gc_time,
                       gc_collected=gc_collected,
                       mem=mem,
                       batch_sizes=batch_sizes,
                       func_name=func_name)





