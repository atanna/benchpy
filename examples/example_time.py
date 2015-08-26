import benchpy as bp
from benchpy.run import noop


def bench_f(f_, *args, run_params_=None, **kwargs):
    if run_params_ is None:
        run_params_ = {}
    bp.bench("f", f_, *args, run_params=run_params_, **kwargs).run()


if __name__ == "__main__":
    run_params = dict(max_batch=10, n_batches=2, n_samples=2)
    f = noop
    run_params_ = dict(run_params)
    run_params_["n_jobs"] = -1
    res = bp.group("benchmark_time",
                   [bp.bench("f", f, run_params=run_params),
                    bp.bench("bench(f)", bench_f, noop, run_params=run_params,
                             run_params_=run_params_)]).run()
    print(res)
    print(res.results[1])