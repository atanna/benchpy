# -*- coding: utf-8 -*-

from __future__ import absolute_import

import gc
import sys

# Monkey patch ``gc`` for Python version prior to 3.3.
try:
    gc.callbacks
except AttributeError:
    gc.callbacks = []

from ._compat import ticker


class TimingCallback(object):
    """Garbage collector callback measuring total collection time.

    Attributes
    ----------

    time : float
        The number of seconds spent on garbage collection.
    """
    def __init__(self):
        self.clear()

    def __call__(self, phase, info):
        if phase == "start":
            self.start_time = ticker()
        elif phase == "stop":
            self.collection_time += ticker() - self.start_time
            del self.start_time

    def collect(self):
        gc.collect()

    def clear(self):
        self.collection_time = 0.0


class gc_manager(object):
    """Context manager controlling and monitoring GC behaviour.

    Examples
    --------

    >>> with gc_manager() as m:
    ...     for i in range(10):
    ...         cycle = []
    ...         cycle.append(cycle)
    ..          m.collect()
    ...     print(m.time)
    ...
    0.23067400299623841  # doctest: +SKIP

    Notes
    -----
    Due to the API limitations it is impossible to monitor the garbage
    collector on Python versions prior to 3.3, thus the time reported is
    always ``0.0``.
    """
    def __init__(self, enabled=True):
        self.gc_was_enabled = gc.isenabled()
        if enabled:
            gc.enable()
        else:
            gc.disable()

    def __enter__(self):
        gc.collect()
        self.callback = callback = TimingCallback()
        gc.callbacks.append(callback)
        return callback

    def __exit__(self, *exc_info):
        gc.collect()
        gc.callbacks.remove(self.callback)
        del self.callback

        if self.gc_was_enabled:
            gc.enable()
        else:
            gc.disable()
