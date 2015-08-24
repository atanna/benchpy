# -*- coding: utf-8 -*-

from benchpy.utils import cached_property


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
