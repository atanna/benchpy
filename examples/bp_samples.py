from collections import OrderedDict
import html5lib
import os
import time
import benchpy as bp
import numpy as np
import pylab as plt
from io import StringIO
from math import factorial, pow

DIR_SAMPLE_RESULTS = "results/"

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


def html_sample():
    data = StringIO(open(
        os.path.join(os.path.join(os.path.dirname(__file__), "data"),
                     "html5lib_spec.html")).read())

    max_batch = 70
    n_batches = 70
    n_samples = 80

    # max_batch = 10
    # n_batches = 10
    # n_samples = 10

    run_params = OrderedDict(max_batch=max_batch,
                  n_batches=n_batches,
                  n_samples=n_samples)
    path = get_path("html_parse", "", max_batch, n_batches, n_samples)
    print(path)

    bp.bench(html_parse, data,
             run_params=run_params).run().save_info(path, "gc")

    run_params["with_gc"] = False
    bp.bench(html_parse, data,
             run_params=run_params).run().save_info(path)


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
                           max_batch=45,
                           n_batches=80)])
    print(res)

    if show_plot:
        bp.plot_results(res)
        plt.show()


def cycle_sample(show_plot=False):
    n = 10
    res = bp.run(bp.group("Cycle",
                          [bp.bench(cycle_list, n,
                                    run_params=dict(with_gc=True),
                                    func_name="with_gc"),
                           bp.bench(cycle_list, n,
                                    run_params=dict(with_gc=False),
                                    func_name="without_gc")],
                          n_samples=10,
                          max_batch=10,
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
    name = "sleep_{}".format(sec)
    max_batch = 100
    n_batches = 40
    n_samples = 100


    max_batch = 50
    n_batches = 50
    n_samples = 50
    path = get_path("sleep", sec, max_batch, n_batches, n_samples)
    print(path)
    res = bp.run(bp.bench(time.sleep, sec,
                           run_params=dict(n_samples=n_samples,
                                           max_batch=max_batch,
                                           n_batches=n_batches),
                           func_name=name))
    res.save_info(path)
    print(res)


def quick_noop_sample():
    res = bp.run(bp.bench(noop,
                          func_name="noop"),
                 n_samples=5,
                 max_batch=10,
                 n_batches=2,
                 with_gc=False)
    print(res)


def exception_sample():
    res = bp.run([bp.bench(run_exception)])
    print(res)


def features_sample():
    n = 1000
    max_batch = 4000
    n_batches = 100
    n_samples = 40

    n = 100
    max_batch = 500
    n_batches = 60
    n_samples = 100

    n = 100
    max_batch = 100
    n_batches = 40
    n_samples = 80

    run_params = OrderedDict(max_batch=max_batch,
                  n_batches=n_batches,
                  n_samples=n_samples)
    path = get_path("cycle", n, max_batch, n_batches, n_samples)
    print(path)

    bp.bench(cycle_list, n,
             run_params=run_params).run().save_info(path, "gc")

    run_params["with_gc"] = False
    bp.bench(cycle_list, n,
             run_params=run_params).run().save_info(path)


def get_path(name, params, max_batch, n_batches, n_samples):
    dir_results = "results_with_mem2"
    path = "{dir_res}/{name}/{params}/" \
           "{max_batch}_{n_batches}_{n_samples}/{folder}/"\
        .format(dir_res=dir_results,
                name=name,
                params=params,
                max_batch=max_batch,
                n_batches=n_batches,
                n_samples=n_samples,
                folder=np.random.randint(1000))
    return path


def cycle_and_sleep(n, t):
    cycle_list(n)
    time.sleep(t)


def sample(f, *params, name=None,
           max_batch=100, n_batches=40,
           n_samples=100, path=None):
    if name is None:
        name = f.__name__ + str(params)
    if path is None:
        path = get_path(f.__name__, params, max_batch, n_batches, n_samples)
    print(path)
    run_params=dict(n_samples=n_samples,
                    max_batch=max_batch,
                    n_batches=n_batches)

    res = bp.bench(f, *params, run_params=run_params, func_name=name+"with gc")\
        .run()
    res.save_info(path, "gc")
    print(res)

    run_params["with_gc"] = False
    res = bp.bench(f, *params, run_params=run_params, func_name=name+"without gc").run()
    res.save_info(path)
    print(res)


if __name__ == "__main__":
    # features_sample()
    # html_sample()
    # factorial_sample()
    # cycle_list_sample()
    # cycle_sample(True)
    # noop_sample()
    # quick_noop_sample()
    # exception_sample()
    # sleep_sample(1e-2)
    # sample(time.sleep, 5e-3, max_batch=100, n_batches=100, n_samples=20)
    # sample(cycle_and_sleep, 100, 1e-2, max_batch=50, n_batches=50, n_samples=80)
    sample(cycle_and_sleep, 100, 1e-3, max_batch=45, n_batches=45, n_samples=40)
    # sample(cycle_and_sleep, 100, 1e-9, max_batch=35, n_batches=35, n_samples=15)
    # sample(cycle_list, 100, max_batch=15, n_batches=10, n_samples=10)


