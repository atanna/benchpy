# -*- coding: utf-8 -*-

class cached_property(object):
    """A property which is only computed once per instance."""
    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = instance.__dict__[self.func.__name__] = self.func(instance)
        return value
