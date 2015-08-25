# -*- coding: utf-8 -*-

import gc
import sys

from numpy.testing.decorators import skipif

from benchpy.garbage import gc_manager


def test_gc_manager():
    gc.disable()
    assert not gc.callbacks

    with gc_manager(enabled=True) as m:
        assert gc.callbacks
        assert gc.isenabled()

    # Make sure we've rolled back the old state of ``gc``.
    assert not gc.callbacks
    assert not gc.isenabled()


@skipif(sys.version_info[:2] < (3, 3))
def test_gc_manager_timing():
    with gc_manager() as m:
        for i in range(10):
            m.collect()
            cycle = []
            cycle.append(cycle)
        assert m.collection_time > 0
