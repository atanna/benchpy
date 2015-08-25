# -*- coding: utf-8 -*-

from __future__ import absolute_import

from collections import namedtuple
from itertools import repeat
from multiprocessing import Pool

import numpy as np

from .analysis import StatMixin
from .display import VisualMixin, VisualMixinGroup


_Bench = namedtuple("_Bench", "name f run_params")


class Bench(_Bench):
    def run(self, *args, **kwargs):
        kwargs.update(self.run_params, func_name=self.name)
        return _run(self.f, *args, **kwargs)


_Group = namedtuple("_Group", "name group run_params")


class Group(_Group):
    def run(self, *args, **kwargs):
        kwargs.update(self.run_params)
        return GroupResult(self.name, [
            bench.run(*args, **kwargs) for bench in self.group
        ])


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
    from ._gc_time import get_time

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
