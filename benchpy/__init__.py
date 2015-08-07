from benchpy.benchtime.my_time import get_time_perf_counter

__all__ = ['BenchException', 'plot_results',
           'bench', 'group', 'run',
           'BenchResult', 'GroupResult',
           'GC_NUM_GENERATIONS',
           'get_time_perf_counter']

class BenchException(Exception):
    pass

from .analyse import BenchResult, GroupResult
from .run import bench, group, run
from .display import plot_results


