# When some filesystems launch, they need to be in a process
# launched by a certain start method. Here, we provide utility
# functions to set a start method for only this plugin's
# subprocesses, without disrupting the start method used in
# other plugins.

import os
import multiprocessing as mp

def KinetoContext():
    # Default to the operating system's start method,
    # and keep the method as a static attribute
    method = mp.get_start_method(False)
    if os.getenv('TORCH_PROFILER_START_METHOD'):
        method = os.getenv('TORCH_PROFILER_START_METHOD')
    return mp.get_context(method)
