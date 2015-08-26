# -*- coding: utf-8 -*-

import sys
import time

# ``perf_counter`` was introduced in Python3.3.
if sys.version_info[:2] < (3, 3):
    ticker = time.clock
else:
    ticker = time.perf_counter
