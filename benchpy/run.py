import gc
import numpy as np
from collections import defaultdict, namedtuple
from functools import partial
from .analyse import BenchRes, GroupRes
from . import BmException

try:
    from .benchtime.my_time import get_time_perf_counter
except:
    from my_time import get_time_perf_counter


Bench = namedtuple("Case", 'name f run_params')
Group = namedtuple("Group", 'name group run_params')


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
        return GroupRes(case.name, [run(bench, *args,
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
        raise BmException("Case must be Bench or Group or list")


def _run(f, n_samples=10, max_batch=100, n_batches=10, with_gc=True,
         func_name=""):
    """
    :param f: function without arguments
    :param n_samples: number of samples
    :param max_batch: maximum of batch size
    :param n_batches: number of batches
    :param with_gc: {True, False} Garbage Collector
    :param func_name:
    :param n_efforts:
    :return:
    """
    # if n_samples < 2 or n_batches < 2:
    #     raise BmException("number of samples and number of batches "
    #                       "must greater than 1")

    n_batches = min(max_batch, n_batches)

    batch_sizes = \
        np.arange(int(max_batch), 0, -int(max_batch / n_batches))[::-1]

    try:
        _warm_up(f)
    except Exception as e:
        raise BmException(e)

    gc_disable = gc.disable
    gcold = gc.isenabled()
    if with_gc:
        gc.disable = lambda: None
    else:
        gc.disable()
    noop_time = [get_time_perf_counter(noop, batch) for batch in batch_sizes]
    n_batches = len(batch_sizes)
    res = np.zeros((n_samples, n_batches))
    gc_info = defaultdict(list)

    for sample in range(n_samples):
        for i, batch in enumerate(batch_sizes):
            try:
                gc.collect()
                prev_stats = gc.get_stats()
                time = get_time_perf_counter(f, batch) - noop_time[i]
                diff, is_diff = _diff_stats(prev_stats, gc.get_stats())
                res[sample, i] = time

                if with_gc and is_diff and \
                    (batch not in gc_info or
                         not _diff_equal(gc_info[batch][-1], diff)):
                    gc_info[batch].append(diff)

            except Exception as e:
                raise BmException(e.args)

    gc.disable = gc_disable
    if gcold:
        gc.enable()
    return BenchRes(res, gc_info, batch_sizes, with_gc, func_name)


def _warm_up(f, n=2):
    for i in range(n):
        f()


def _diff_stats(gc_stats0, gc_stats1):
    res = []
    is_diff = False
    for st0, st1 in zip(gc_stats0, gc_stats1):
        res.append({})
        for key in st0.keys():
            diff = st1[key] - st0[key]
            if diff:
                res[-1][key] = diff
                is_diff = True
    return res, is_diff


def _diff_equal(diff1, diff2):
    for d1, d2 in zip(diff1, diff2):
        for key in set().union(d1.keys()).union(d2.keys()):
            if key not in d1 \
                    or key not in d2 \
                    or d1[key] != d2[key]:
                return False
    return True


def noop(*args, **kwargs):
    pass