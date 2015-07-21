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