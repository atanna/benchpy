import glob
import html5lib
import os
import time
import benchpy as bp
import numpy as np
from collections import OrderedDict
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


def cycle_and_sleep(n, t):
    cycle_list(n)
    time.sleep(t)


def noop():
    pass


def run_exception():
    raise IndexError


def exception_sample():
    res = bp.run([bp.bench(run_exception)])
    print(res)


def html_sample(save=True, path=None, doc=None):
    _dir = os.path.join(os.path.dirname(__file__), "data")
    if doc is None:
        doc = np.random.choice(glob.glob(_dir+"/*.html"))
    data = StringIO(open(
        os.path.join(_dir, doc)).read())

    max_batch = 70
    n_batches = 70
    n_samples = 80

    # max_batch = 20
    # n_batches = 20
    # n_samples = 20

    run_params = OrderedDict(max_batch=max_batch,
                  n_batches=n_batches,
                  n_samples=n_samples)

    case = bp.bench(html_parse, data,
                    run_params=run_params,
                    func_name="html___with_gc")
    path = path if path is not None else \
        get_path("html_parse_"+os.path.splitext(os.path.basename(doc))[0],
                 "", case=case)
    run_sample_case(case, save=save, path=path, path_suffix="gc")

    run_params["with_gc"] = False
    case = bp.bench(html_parse, data,
                    run_params=run_params,
                    func_name="html_without_gc")
    run_sample_case(case, save=save, path=path)


def list_group_sample(save=True, path=None):
    n = 100
    case = [bp.group("factorial",
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
                             func_name="simple^")], with_gc=False)]
    run_sample_case(case, save, name="factorial_group", path=path)


def cycle_list_sample(save=False):
    bench_list = [bp.bench(cycle_list, n,
                           func_name="{} cycles".format(n))
                  for n in range(100, 201, 100)]
    name = "Cycle list"
    case = [bp.group(name, bench_list,
                    max_batch=100,
                    n_batches=10,
                    n_samples=100),
            bp.group(name, bench_list,
                     max_batch=45,
                     _batches=45,
                     n_samples=100)]
    run_sample_case(case, save)


def quick_noop_sample():
    res = bp.run(bp.bench(noop,
                          func_name="noop"),
                 n_samples=5,
                 max_batch=10,
                 n_batches=2,
                 with_gc=False)
    print(res)


def cycle_sample(**kwargs):
    n = 10
    group = bp.group("Cycle",
                     [bp.bench(cycle_list, n,
                               run_params=dict(with_gc=True),
                               func_name="with_gc"),
                      bp.bench(cycle_list, n,
                               run_params=dict(with_gc=False),
                               func_name="without_gc")],
                     n_samples=10,
                     max_batch=10,
                     n_batches=10)

    run_sample_case(group, **kwargs)


def get_path(name="", params=None, max_batch=-1, n_batches=-1,
             n_samples=-1, case=None):
    if case is not None:
        if isinstance(case, list):
            return get_path("List" if name is None else name)
        return get_path(name=name,
                        max_batch=case.run_params.get('max_batch', -1),
                        n_batches=case.run_params.get('n_batches', -1),
                        n_samples=case.run_params.get('n_samples', -1))
    dir_results = "results_fixed_cpu_3"

    if max_batch > 0:
        inter_folder = "/{max_batch}_{n_batches}_{n_samples}/"\
            .format(max_batch=max_batch,
                    n_batches=n_batches,
                    n_samples=n_samples)
    else:
        inter_folder = ""
    if params is None:
        params = ""
    path = "{dir_res}/{name}/{params}{inter_folder}/{folder}/"\
        .format(dir_res=dir_results,
                name=name,
                params=params,
                inter_folder=inter_folder,
                folder=np.random.randint(1000))
    return path


def run_sample_case(case, save=True, name=None, path=None, **kwargs):
    if save and path is None:
       path = get_path(name=name, case=case)
    print(path)

    res = bp.run(case)
    print(res)

    if save:
        bp.save_info(res, path, **kwargs)


def sample(f, *params, name=None,
           max_batch=100, n_batches=40, n_samples=100,
           save=True, path=None):
    if name is None:
        name = f.__name__ + str(params)
    if save:
        if path is None:
            path = get_path(f.__name__, params, max_batch, n_batches, n_samples)

    run_params = dict(n_samples=n_samples,
                      max_batch=max_batch,
                      n_batches=n_batches)

    case = bp.bench(f, *params, run_params=run_params,
                    func_name=name+"with gc")
    run_sample_case(case, save=save, path=path, path_suffix="gc")

    run_params["with_gc"] = False
    case = bp.bench(f, *params, run_params=run_params,
                    func_name=name+"without gc")
    run_sample_case(case, save=save, path=path)


if __name__ == "__main__":
    # html_sample()
    # list_group_sample(True)
    # cycle_list_sample()
    # cycle_sample()
    # noop_sample()
    # quick_noop_sample()
    # exception_sample()
    # sample(time.sleep, 5e-3, max_batch=100, n_batches=100, n_samples=20)
    # sample(cycle_and_sleep, 100, 1e-2, max_batch=50, n_batches=50, n_samples=80)
    # sample(cycle_and_sleep, 100, 1e-3, max_batch=45, n_batches=45, n_samples=40)
    # sample(cycle_and_sleep, 100, 1e-3, max_batch=100, n_batches=80, n_samples=80)
    # sample(cycle_list, 1000, max_batch=100, n_batches=80, n_samples=80)
    # sample(cycle_list, 1000, max_batch=10, n_batches=10, n_samples=40)
    # sample(cycle_list, 100, max_batch=10, n_batches=10, n_samples=20)
    sample(cycle_list, 100, max_batch=5, n_batches=2, n_samples=2)



