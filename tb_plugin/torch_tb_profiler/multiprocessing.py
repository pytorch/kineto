import os
import multiprocessing as mp

valid_settings = ['fork','spawn','forkserver']

def get_start_method():
    start_method = os.getenv('TORCH_PROFILER_START_METHOD')
    if start_method and start_method not in valid_settings:
        raise ValueError("the start_method {} is not valid".format(start_method))
    return start_method
