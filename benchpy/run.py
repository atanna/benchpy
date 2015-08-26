# -*- coding: utf-8 -*-

from __future__ import absolute_import

from collections import namedtuple
from itertools import repeat
from multiprocessing import Pool
import os

import numpy as np

from .analysis import StatMixin
from ._compat import ticker
from .display import VisualMixin, VisualMixinGroup
from .garbage import gc_manager
from ._speedups import time_loop


class Bench(namedtuple("Bench", "name f run_params")):
    def run(self, *args, **kwargs):
        kwargs.update(self.run_params, func_name=self.name)
        return _run(self.f, *args, **kwargs)


class Group(namedtuple("Group", "name group run_params")):
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
        self.results = results
        res = results[0]
        self.n_batches = res.n_batches
        self.n_samples = res.n_samples
        self.batch_sizes = res.batch_sizes


def _run(f, func_name="",
         n_samples=10, max_batch=100, n_batches=10,
         with_gc=True, n_jobs=1):
    """

    :param f: function which measured (without arguments)
    :param func_name: name of function
    :param n_samples: number of measuring samples (for each batch)
    :param max_batch: maximum of batch size
    :param n_batches: number of batches
    :param with_gc: flag for enable/disable Garbage Collector
     which measure gc working time (use only with gc, python version >= 3.3)
    :param n_jobs: the number of jobs to run in parallel.
    If -1, then the number of jobs is set to the number of cores.
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
    if batch_sizes[0] > 1:
        batch_sizes = np.concatenate([[1], batch_sizes])

    n_batches = len(batch_sizes)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    full_time = np.zeros((n_samples, n_batches))
    gc_time = np.zeros((n_samples, n_batches))
    est_time_for_sample = predict_waiting_time_for_sample(f, with_gc, batch_sizes)

    start_t = ticker()
    for i in range(n_samples):
        print("start {} sample ({}/{})".format(i, i, n_samples))
        print("Estimated time to complete: {} s."
          .format(est_time_for_sample*(n_samples - i)))
        if n_jobs > 1:
            with Pool(n_jobs) as p:
                full_time[i], gc_time[i] = \
                    zip(*p.map(get_time, zip(repeat(f),
                                             batch_sizes,
                                             repeat(with_gc))))

        else:
            full_time[i], gc_time[i] = \
                zip(*list(map(get_time, zip(repeat(f),
                                            batch_sizes,
                                            repeat(with_gc)))))
        if not i:
            est_time_for_sample = ticker() - start_t

    return BenchResult(full_time,
                       gc_time=gc_time,
                       batch_sizes=batch_sizes,
                       with_gc=with_gc,
                       func_name=func_name)


def predict_waiting_time_for_sample(f, with_gc, batch_sizes):
    t = np.mean([get_time((f, 1, with_gc))[0] for _ in range(4)])
    return np.sum(batch_sizes*t)


def noop_time_preprocessing(batch_sizes):
    return dict((batch, time_loop(noop, batch)) for batch in batch_sizes)


def noop_time(batch, dict_noop_time=None):
    """
    Return time of empty function, which has run batch times.
    :param batch:
    :param dict_noop_time:
    :return:
    """
    if dict_noop_time is None:
        dict_noop_time = {}  # JFY `dict_noop_time` is always `None`.
    return dict_noop_time.get(batch, time_loop(noop, batch))


def noop(*args, **kwargs):
    """A function which does nothing."""


def _warm_up(f, n=2):
    for i in range(n):
        f()


def get_time(args):
    """
    Return time of running function `f` the `batch` times.
    :param args: tuple with
    f - function,
    batch - number of the executions in cycle
    with_gc - flag to enable/disable Garbage Collector
    collections time (useful only with python version >= 3.3)
    """
    f, batch, with_gc = args
    _warm_up(f)
    with gc_manager(with_gc) as m:
        time = max(time_loop(f, batch) - noop_time(batch), 0.)
        return time, m.collection_time
