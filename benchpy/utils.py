# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from ._compat import OrderedDict


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


TIME_UNITS = OrderedDict([
    ("d", 86400),
    ("h", 3600),
    ("m", 60),
    ("s", 1),
    ("ms", 1e-3),
    ("µs", 1e-6),
    ("ns", 1e-9)
])


def to_time_unit(seconds, unit=None):
    """Converts a given duration to the appropriate time unit.

    Parameters
    ----------
    seconds : float
        Duration in seconds.
    unit: str (optional)
        Time unit to use instead of picking the appropriate one.

    Returns
    -------
    converted : float
        Duration converted to the chosen time unit.
    unit : str
        Chosen time unit.
    """
    if unit:
        return seconds / TIME_UNITS[unit], unit

    for candidate, w in TIME_UNITS.items():
        converted = seconds / w
        if int(converted):
            return converted, candidate
