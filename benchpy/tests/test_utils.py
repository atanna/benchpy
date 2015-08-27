# -*- coding: utf-8 -*-

from benchpy.utils import cached_property, to_time_unit


def test_cached_property():
    class Foo(object):
        def __init__(self):
            self.called = 0

        @cached_property
        def bar(self):
            self.called += 1

    foo = Foo()
    assert foo.called == 0
    foo.bar
    assert foo.called == 1
    foo.bar
    assert foo.called == 1


def test_to_time_unit():
    assert to_time_unit(0.128) == (128, "ms")
    assert to_time_unit(1) == (1, "s")
    assert to_time_unit(5) == (5, "s")
    assert to_time_unit(60) == (1, "m")
    assert to_time_unit(66) == (1.1, "m")
    assert to_time_unit(126) == (2.1, "m")
    assert to_time_unit(126, unit="s") == (126, "s")
