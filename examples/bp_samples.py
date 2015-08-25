import glob
import html5lib
import os
import benchpy as bp
import numpy as np
from collections import OrderedDict
from io import StringIO


def html_parse(data=""):
    html5lib.parse(data)


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

    run_params = OrderedDict(max_batch=max_batch, n_batches=n_batches,
                             n_samples=n_samples)

    case = bp.bench("html___with_gc", html_parse, data, run_params=run_params)
    path = path if path is not None else \
        get_path("html_parse_"+os.path.splitext(os.path.basename(doc))[0],
                 "", case=case)
    run_sample_case(case, save=save, path=path, path_suffix="gc")

    run_params["with_gc"] = False
    case = bp.bench("html_without_gc", html_parse, data, run_params=run_params)
    run_sample_case(case, save=save, path=path)


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

    case = bp.bench(name + "with gc", f, *params, run_params=run_params)
    run_sample_case(case, save=save, path=path, path_suffix="gc")

    run_params["with_gc"] = False
    case = bp.bench(name + "without gc", f, *params, run_params=run_params)
    run_sample_case(case, save=save, path=path)


if __name__ == "__main__":
    html_sample()
