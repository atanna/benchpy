import tracemalloc
import numpy as np
from benchpy._gc_time import noop, gc_manager


def noop_mem_prepocessing(batch_sizes):
    is_tracing = tracemalloc.is_tracing()
    if not is_tracing:
        tracemalloc.start()
    dict_noop_mem= dict(zip(batch_sizes,
                     [get_mem_stats(noop, batch)
                      for batch in batch_sizes]))
    if not is_tracing:
        tracemalloc.stop()
    return dict_noop_mem


def noop_mem(batch, dict_noop_mem=None):
    if dict_noop_mem is None:
        dict_noop_mem = {}
    return dict_noop_mem.get(batch, get_mem_stats(noop, batch))


def get_count_and_size(snapshot1, snapshot2, group_by='lineno'):
    diff_stats = snapshot2.compare_to(snapshot1, group_by)
    count = sum(diff.count_diff for diff in diff_stats)
    size = sum(diff.size_diff for diff in diff_stats)
    return np.array([count, size])


def get_mem_stats(f, batch):
    snapshot1 = tracemalloc.take_snapshot()
    for i in range(batch):
        f()
    snapshot2 = tracemalloc.take_snapshot()
    return get_count_and_size(snapshot1, snapshot2)


def get_mem(args):
    f, batch, with_gc = args
    with gc_manager(with_gc, False):
        is_tracing = tracemalloc.is_tracing()
        if not is_tracing:
            tracemalloc.start()
        res = np.maximum(get_mem_stats(f, batch) - noop_mem(batch), 0.)
        if not is_tracing:
            tracemalloc.stop()
    return res


