__all__ = ['bench', 'group', 'run', 'plot_results']


from .timed_eval import get_time_perf_counter
from .run import bench, group, run
from .display import plot_results
from .magic import ip