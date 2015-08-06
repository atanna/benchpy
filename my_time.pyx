import time
from eval cimport PyEval_EvalCodeEx
from cpython.ref cimport PyObject

cdef object time_clock = time.clock
cdef object time_perf_counter = time.perf_counter

def get_time_clock(f, int n):
    cdef int i
    cdef double t0, t1

    t0 = time_clock()
    for i in range(n):
        f()
    t1 = time_clock()
    return t1 - t0


def get_time_perf_counter(f, int n):
    cdef int i
    cdef double t0, t1

    t0 = time_perf_counter()
    for i in range(n):
        f()
    t1 = time_perf_counter()
    return t1 - t0


def _eval_f(f, int n):
    cdef int i
    cdef double t0, t1
    cdef object _code = f.__code__
    cdef dict _globals = f.__globals__
    cdef dict _locals = {}
    cdef object _closure = f.__closure__
    cdef PyObject **null2 = <PyObject **>NULL
    cdef PyObject *null1 = <PyObject *>NULL
    t0 = time_clock()
    for i in range(n):
        PyEval_EvalCodeEx(_code, _globals, _locals,
                          null2, 0, null2, 0, null2, 0,
                          null1, _closure)

    t1 = time_clock()
    return t1 - t0

