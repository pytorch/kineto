import os

def get_start_method():
    return os.getenv('TORCH_PROFILER_START_METHOD')
