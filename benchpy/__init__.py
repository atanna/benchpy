__all__ = ['bench', 'group', 'run', 'plot_results',
           "plot_features", "save_info"]

__version__ = "0.1.0"

from .run import bench, group, run
from .display import plot_results, plot_features, save_info
from .magic import ip
