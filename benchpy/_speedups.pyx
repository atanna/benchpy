cimport cython

import time

# Cache aliases to clock functions so we don't do a dynamic lookup on
# each function call.
cdef object ticker = time.clock

@cython.optimize.unpack_method_calls(True)
def time_loop(f, int n):
    cdef int i
    cdef double t0, t1

    t0 = ticker()
    for i in range(n):
        f()
    t1 = ticker()
    return t1 - t0
