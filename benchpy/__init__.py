__all__ = ['bench', 'group', 'run', 'plot_results',
           "plot_features", "save_info"]


from .timed_eval import get_time_perf_counter
from .run import bench, group, run
from .display import plot_results, plot_features, save_info
from .magic import ip