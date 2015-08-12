from collections import OrderedDict
import html5lib
import os
import time
import benchpy as bp
import numpy as np
import pylab as plt
from io import StringIO
from math import factorial, pow


def factorial_slow(n):
    assert n >= 0
    return 1 if n == 0 else n * factorial_slow(n-1)


def pow_slow(x, n):
    assert n >= 0
    return 1 if n == 0 else x * pow_slow(x, n-1)


def html_parse(data=""):
    html5lib.parse(data)


def cycle_list(n):
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
                  bp.group("factorial without_gc",
                           [bp.bench(factorial, n,
                                     func_name="math_!"),
                            bp.bench(factorial_slow, n,
                                     func_name="slow_!")], with_gc=False),
                  bp.group("pow",
                           [bp.bench(pow, n, n,
                                     func_name="math^"),
                            bp.bench(pow_slow, n, n,
                                     func_name="simple^")]),
                  bp.group("pow_without_gc",
                           [bp.bench(pow, n, n,
                                     func_name="math^"),
                            bp.bench(pow_slow, n, n,
                                     func_name="simple^")], with_gc=False)])
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


def cycle_list_sample(show_plot=False):
    bench_list = [bp.bench(cycle_list, n,
                           func_name="{} cycles".format(n))
                  for n in range(100, 201, 100)]
    res = bp.run([bp.group("Cycle list", bench_list,
                           n_samples=100,
                           max_batch=100,
                           n_batches=10),
                  bp.group("Cycle list", bench_list,
                           n_samples=100,
                           max_batch=10,
                           n_batches=10)])
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def cycle_sample(show_plot=False):
    n = 100000
    res = bp.run(bp.group("Cycle",
                          [bp.bench(cycle_list, n,
                                    run_params=dict(with_gc=True),
                                    func_name="with_gc"),
                           bp.bench(cycle_list, n,
                                    run_params=dict(with_gc=False),
                                    func_name="without_gc")],
                          n_samples=10,
                          max_batch=100,
                          n_batches=10))
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def noop_sample(show_plot=False):
    res = bp.run(bp.group("noop",
                          [bp.bench(noop,
                                    run_params=dict(n_samples=100,
                                                    max_batch=100,
                                                    n_batches=10)),
                           bp.bench(noop,
                                    run_params=dict(n_samples=100,
                                                    max_batch=10,
                                                    n_batches=5))])
                 )
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def sleep_sample(sec=0.001):
    res = bp.run([bp.bench(time.sleep, sec,
                           run_params=dict(n_samples=2,
                                           max_batch=2,
                                           n_batches=2),
                           func_name="Sleep_[{}]".format(sec))])
    print(res)


def quick_noop_sample():
    res = bp.run(bp.bench(noop,
                          func_name="noop"),
                 n_samples=2,
                 max_batch=10,
                 n_batches=2)
    print(res)


def exception_sample():
    res = bp.run([bp.bench(run_exception)])
    print(res)


def features_sample():
    n = 1000
    max_batch = 1000
    n_batches = 100
    n_samples = 40

    max_batch = 100
    n_batches = 100
    n_samples = 40

    run_params = OrderedDict(max_batch=max_batch,
                  n_batches=n_batches,
                  n_samples=n_samples)
    path = "img_multiprocessing3/cycle/{}_{}_{}/{}/".format(max_batch, n_batches, n_samples, np.random.randint(n))
    print(path)

    bp.bench(cycle_list, n,
             run_params=run_params).run().save_info(path, "gc")

    run_params["with_gc"] = False
    bp.bench(cycle_list, n,
             run_params=run_params).run().save_info(path, with_plots=False)


if __name__ == "__main__":
    features_sample()
    # html_sample()
    # factorial_sample()
    # circle_list_sample()
    # circle_sample()
    # noop_sample()
    # quick_noop_sample()
    # exception_sample()
    # sleep_sample(1e-3)
