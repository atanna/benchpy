import gc
import numpy as np
from time import perf_counter
from .timed_eval import get_time_perf_counter


def noop_time_preprocessing(batch_sizes):
    return dict(zip(batch_sizes,
                     [get_time_perf_counter(noop, batch)
                      for batch in batch_sizes]))


def noop_time(batch, dict_noop_time=None):
    if dict_noop_time is None:
        dict_noop_time = {}
    return dict_noop_time.get(batch, get_time_perf_counter(noop, batch))


def _warm_up(f, n=2):
    for i in range(n):
        f()


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


class gc_manager(object):

    def __init__(self, with_gc, with_callback):
        self.with_gc = with_gc
        self.gcold = gc.isenabled()
        self.with_callback = with_callback
        if not with_gc:
            gc.disable()

    def __enter__(self):
        self.call_back_gc = CallBackGC()
        if self.with_callback:
            gc.callbacks.append(self.call_back_gc)
            gc.collect()
            self.call_back_gc.clear()
        return self.call_back_gc

    def __exit__(self, *exc_info):
        gc.collect()
        if self.with_callback:
            for i, f in enumerate(gc.callbacks):
                if id(f) == id(self.call_back_gc):
                    del gc.callbacks[i]
                    break
            del self.call_back_gc
        if self.gcold:
            gc.enable()


def get_time(args):
    f, batch, with_gc, with_callback = args
    _warm_up(f)
    with gc_manager(with_gc, with_callback) as callback:
        time = max(get_time_perf_counter(f, batch) - noop_time(batch), 0.)
        if with_callback:
            return time, callback.time()
    return time, 0


def noop(*args, **kwargs):
    pass