import time
import benchpy as bp
from benchpy import BmException
from bp_samples import exception_sample, circle_list_sample


class TestClass:

    def test_exception(self):
        try:
            exception_sample()
        except Exception as e:
            assert isinstance(e, BmException)

        try:
            circle_list_sample()
        except Exception as e:
            assert isinstance(e, BmException)
            assert isinstance(e, BmException)

    def test_params(self):
        sec = 0.0001
        res = bp.run(
            bp.group("sleep",
                     [bp.bench(time.sleep, sec,
                               run_params=dict(n_samples=2,
                                               max_batch=2,
                                               n_batches=2,
                                               with_gc=True
                                               ),
                               func_name="Sleep_[{}]".format(sec))],
                     with_gc=False),
            with_gc=True
        )
        assert res.results[0].with_gc

        res = bp.run(
            bp.group("sleep",
                     [bp.bench(time.sleep, sec,
                               run_params=dict(n_samples=2,
                                               max_batch=2,
                                               n_batches=2),
                               func_name="Sleep_[{}]".format(sec))],
                     with_gc=False),
            with_gc=True
        )
        assert not res.results[0].with_gc