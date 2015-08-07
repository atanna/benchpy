from benchpy.benchtime.my_time import get_time_perf_counter

__all__ = ['BmException', 'plot_results',
           'bench', 'group', 'run',
           'BenchRes', 'GroupRes',
           'GC_NUM_GENERATIONS',
           'get_time_perf_counter']

GC_NUM_GENERATIONS = 3
class BmException(Exception):
    pass

try:
    from .analyse import BenchRes, GroupRes
    from .run import bench, group, run
    from .display import plot_results
except:
    from analyse import BenchRes, GroupRes
    from run import bench, group, run
    from display import plot_results

