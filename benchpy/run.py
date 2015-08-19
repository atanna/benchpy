import gc
import tracemalloc
import numpy as np
from collections import namedtuple
from itertools import repeat
from functools import partial
from multiprocessing import Pool
from time import perf_counter
from .analyse import StatMixin
from .timed_eval import get_time_perf_counter
from .display import VisualMixin, VisualMixinGroup
from .exception import BenchException

GC_NUM_GENERATIONS = 3

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

    gcold = gc.isenabled()
    gc_disable = gc.disable
    call_back_gc = CallBackGC()
    if with_gc:
        gc.disable = lambda: None
        gc.callbacks.append(call_back_gc)
    else:
        gc.disable()

    NoopTime.preprocessing(batch_sizes)
    NoopMem.prepocessing(batch_sizes)

    n_batches = len(batch_sizes)

    mem = np.zeros((n_samples, n_batches, 2))
    full_time = np.zeros((n_samples, n_batches))
    gc_time = np.zeros((n_samples, n_batches))
    gc_collected = np.zeros((n_samples, n_batches, GC_NUM_GENERATIONS))

    for i in range(n_samples):
        if multi:
            with Pool() as p:
                full_time[i], gc_time[i], gc_collected[i] = \
                    zip(*p.map(_get_time, zip(repeat(f), batch_sizes)))
                mem[i] = p.map(_get_mem, zip(repeat(f), batch_sizes))

        else:
            full_time[i], gc_time[i], gc_collected[i] = \
                zip(*list(map(_get_time, zip(repeat(f), batch_sizes))))

    gc.disable = gc_disable
    if gcold:
        gc.enable()
    if with_gc:
        del gc.callbacks[-1]
    return BenchResult(full_time, gc_time=gc_time,
                       gc_collected=gc_collected,
                       mem=mem,
                       batch_sizes=batch_sizes,
                       func_name=func_name)


def _warm_up(f, n=2):
    for i in range(n):
        f()


def _get_time(args):
    f, batch = args
    _warm_up(f)
    call_back_gc = CallBackGC()
    gc.callbacks.append(call_back_gc)
    gc.collect()
    call_back_gc.clear()
    prev_stats = gc.get_stats()
    time = max(get_time_perf_counter(f, batch) - NoopTime.time(batch),
               0.)
    gc_diff = _diff_stats(prev_stats, gc.get_stats())
    gc_time = call_back_gc.time()
    gc.collect()
    del gc.callbacks[-1], call_back_gc
    return time, gc_time, gc_diff


def _diff_stats(gc_stats0, gc_stats1):
    res = np.zeros(GC_NUM_GENERATIONS)
    key = "collected"
    for i, st0, st1 in zip(range(GC_NUM_GENERATIONS), gc_stats0, gc_stats1):
        res[i] = st1[key] - st0[key]
    return res


def _diff_equal(diff1, diff2):
    return (diff1 == diff2).all()


def noop(*args, **kwargs):
    pass


class CallBackGC(object):
    def __init__(self):
        self.starts = []
        self.stops = []

    def __call__(self, phase, info):
        if phase == "start":
            self.starts.append(perf_counter())
        elif phase == "stop":
            self.stops.append(perf_counter())

    def clear(self):
        self.starts.clear()
        self.stops.clear()

    def time(self):
        return np.sum(np.array(self.stops) - np.array(self.starts))


class NoopTime(object):
    _time = {}

    @staticmethod
    def preprocessing(batch_sizes):
        NoopTime._time = dict(zip(batch_sizes,
                         [get_time_perf_counter(noop, batch)
                          for batch in batch_sizes]))

    @staticmethod
    def time(batch):
        return NoopTime._time.get(batch, get_time_perf_counter(noop, batch))


class NoopMem(object):
    _mem = {}

    @staticmethod
    def prepocessing(batch_sizes):
        is_tracing = tracemalloc.is_tracing()
        if not is_tracing:
            tracemalloc.start()
        NoopMem._mem = dict(zip(batch_sizes,
                         [get_mem_stats(noop, batch)
                          for batch in batch_sizes]))
        if not is_tracing:
            tracemalloc.stop()

    @staticmethod
    def mem(batch):
        return NoopMem._mem.get(batch, get_mem_stats(noop, batch))


def count_size(snapshot1, snapshot2, group_by='lineno'):
    diff_stats = snapshot2.compare_to(snapshot1, group_by)
    count = sum(diff.count_diff for diff in diff_stats)
    size = sum(diff.size_diff for diff in diff_stats)
    return np.array([count, size])


def get_mem_stats(f, batch):
    snapshot1 = tracemalloc.take_snapshot()
    for i in range(batch):
        f()
    snapshot2 = tracemalloc.take_snapshot()
    return count_size(snapshot1, snapshot2)


def _get_mem(args):
    f, batch = args
    is_tracing = tracemalloc.is_tracing()
    if not is_tracing:
        tracemalloc.start()
    res = get_mem_stats(f, batch) - NoopMem.mem(batch)
    if not is_tracing:
        tracemalloc.stop()
    return res
