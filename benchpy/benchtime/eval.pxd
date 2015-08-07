from cpython.ref cimport PyObject


cdef extern from "Python.h":

    object PyEval_EvalCodeEx(object co, object globals, object locals,
                             PyObject **args, int argcount,
                             PyObject **kws, int kwcount,
                             PyObject **defs, int defcount,
                             PyObject *kwdefs, object closure)
