import html5lib
import os
import benchpy as bp
import pylab as plt
from io import StringIO
from math import factorial


def factorial_slow(n):
    assert n >= 0
    return 1 if n == 0 else n * factorial_slow(n - 1)


def html_parse(data=""):
    html5lib.parse(data)


def circle_list(n):
    for _ in range(n):
        arr = []
        arr.append(arr)


def noop():
    pass


def factorial_sample(show_plot=False):
    n = 100
    res = bp.run([bp.case(factorial, n, func_name="factorial"),
                  bp.case(factorial_slow, n, func_name="factorial_slow")])
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def html_sample(show_plot=False):
    data = StringIO(open(
        os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                     "html5lib_spec.html")).read())

    res = bp.run([bp.case(html_parse, data=data,
                          func_name="Html",
                          run_params=dict(with_gc=True)),
                  bp.case(html_parse, data=data,
                          func_name="Html",
                          run_params=dict(with_gc=False))],
                 n_samples=100,
                 max_batch=100,
                 n_batches=10)
    print(res)

    if show_plot:
        bp.plot_results(res, ["with_gc", "without_gc"], title="HTML")
        plt.show()


def circle_list_sample(show_plot=False):
    res = bp.run([bp.case(circle_list, 100,
                          func_name="circle_list_100"),
                  bp.case(circle_list, 200,
                          func_name="circle_list_200"),
                  bp.case(circle_list, 300,
                          func_name="circle_list_300")])
    print(res)

    if show_plot:
        bp.plot_results(res, title="Circle_list")
        plt.show()


def noop_sample():
    res = bp.run([bp.case(noop)],
                 n_samples=100,
                 max_batch=100,
                 n_batches=10)
    print(res)


if __name__ == "__main__":
    # html_sample(),
    # factorial_sample()
    circle_list_sample()
    # noop_sample()
