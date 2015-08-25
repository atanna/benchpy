# -*- coding: utf-8 -*-

from __future__ import absolute_import

__all__ = ['bench', 'group', 'run', 'plot_results',
           "plot_features", "save_info"]

import functools

from .run import Bench, Group, _run
from .display import plot_results, plot_features, save_info
from .magic import load_ipython_extension


def bench(f, *args, run_params=None, func_name="", **kwargs):
    """
    :param f: function which measured
    :param args: args of f
    :param run_params: parameters for benchmark
    :param func_name: function name
    :param kwargs: kwargs of f
    :return: Bench
    """
    return Bench(func_name, functools.partial(f, *args, **kwargs),
                 run_params or {})


def group(name, group, **run_params):
    """
    :param name:
    :param group: list of Benches
    :param run_params:
    :return: Group
    """
    return Group(name, group, run_params)


def run(bench_or_benches, *args, **kwargs):
    """
    Exec benchmark for each function in case
    See _run's description
    :param case:
    case = list of cases | Bench | Group
    :param args: args for _run
    :param kwargs: kwargs for _run
    :return:
    """
    if isinstance(bench_or_benches, list):
        return [run(bench, *args, **kwargs) for bench in bench_or_benches]
    else:
        return bench_or_benches.run(*args, **kwargs)
