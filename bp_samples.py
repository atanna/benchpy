import html5lib
import os
import time
import benchpy as bp
import pylab as plt
from io import StringIO
from math import factorial, pow


def factorial_slow(n):
    assert n >= 0
    return 1 if n == 0 else n * factorial_slow(n - 1)


def pow_slow(x, n):
    assert n >= 0
    return 1 if n == 0 else x * pow_slow(x, n-1)


def html_parse(data=""):
    html5lib.parse(data)


def circle_list(n):
    for _ in range(n):
        arr = []
        arr.append(arr)


def noop():
    pass


def run_exception():
    raise IndexError


def factorial_sample(show_plot=False):
    n = 100
    res = bp.run([bp.group("factorial",
                           [bp.bench(factorial, n,
                                     func_name="math_!"),
                            bp.bench(factorial_slow, n,
                                    func_name="slow_!")]),
                  bp.group("pow",
                           [bp.bench(pow, n, n,
                                     func_name="math^"),
                            bp.bench(pow_slow, n, n,
                                     func_name="simple^")])])
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def html_sample(show_plot=False):
    data = StringIO(open(
        os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                     "html5lib_spec.html")).read())

    res = bp.run(bp.group("Html",
                          [bp.bench(html_parse, data,
                                    run_params=dict(with_gc=True),
                                    func_name="with_gc"),
                           bp.bench(html_parse, data,
                                    run_params=dict(with_gc=False),
                                    func_name="without_gc")],
                          n_samples=100,
                          max_batch=100,
                          n_batches=10))
    print(res)

    if show_plot:
        bp.plot_results(res, title="HTML")
        plt.show()


def circle_list_sample(show_plot=False):
    bench_list = [bp.bench(circle_list, n,
                           func_name="{} circles".format(n))
                           for n in range(100, 2001, 100)]
    res = bp.run([bp.group("Circle list", bench_list,
                           n_samples=100,
                           max_batch=100,
                           n_batches=10),
                  bp.group("Circle list", bench_list,
                           n_samples=100,
                           max_batch=10,
                           n_batches=10)])
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def noop_sample(show_plot=False):
    res = bp.run([bp.bench(noop,
                           run_params=dict(n_samples=100,
                                           max_batch=100,
                                           n_batches=10)),
                  bp.bench(noop,
                           run_params=dict(n_samples=100,
                                           max_batch=10,
                                           n_batches=5))]
                 )
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def sleep_sample():
    sec = 0.01
    res = bp.run([bp.bench(time.sleep, sec,
                           run_params=dict(n_samples=20,
                                           max_batch=20,
                                           n_batches=4),
                           func_name="Sleep_[{}]".format(sec))])
    print(res)


def quick_noop_sample():
    res = bp.run(bp.bench(noop))
    print(res)


def exception_sample():
    res = bp.run([bp.bench(run_exception)])
    print(res)


if __name__ == "__main__":
    # html_sample(True)
    # factorial_sample(True)
    # circle_list_sample()
    # noop_sample()
    # quick_noop_sample()
    # exception_sample()
    sleep_sample()