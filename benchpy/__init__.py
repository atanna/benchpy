__all__ = ['bench', 'group', 'run', 'plot_results']


from benchpy.benchtime.my_time import get_time_perf_counter
from .run import bench, group, run
from .display import plot_results
from .magic import ip