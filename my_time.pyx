import time

cdef object time_clock = time.clock


def get_time_clock(f, int n):
    cdef int i
    cdef double t0, t1

    t0 = time_clock()
    for i in range(n):
        f()
    t1 = time_clock()
    return t1 - t0



