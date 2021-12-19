# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import logging
import math
import os
import time
from contextlib import contextmanager
from math import pow

from . import consts


def get_logging_level():
    log_level = os.environ.get('TORCH_PROFILER_LOG_LEVEL', 'INFO').upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level


logger = None


def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger(consts.PLUGIN_NAME)
        logger.setLevel(get_logging_level())
    return logger


def is_chrome_trace_file(path):
    return consts.WORKER_PATTERN.match(path)


def href(text, url):
    """"return html formatted hyperlink string

    Note:
        target="_blank" causes this link to be opened in new tab if clicked.
    """
    return f'<a href="{url}" target="_blank">{text}</a>'


class Canonicalizer:
    def __init__(
            self,
            time_metric='us',
            memory_metric='B',
            *,
            input_time_metric='us',
            input_memory_metric='B'):
        # raw timestamp is in microsecond
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/csrc/autograd/profiler_kineto.cpp#L33
        time_metric_to_factor = {
            'us': 1,
            'ms': 1e3,
            's':  1e6,
        }
        # raw memory is in bytes
        memory_metric_to_factor = {
            'B':  pow(1024, 0),
            'KB': pow(1024, 1),
            'MB': pow(1024, 2),
            'GB': pow(1024, 3),
        }

        # canonicalize the memory metric to a string
        self.canonical_time_metrics = {
            'micro': 'us', 'microsecond': 'us', 'us': 'us',
            'milli': 'ms', 'millisecond': 'ms', 'ms': 'ms',
            '':  's',      'second':  's',  's':  's',
        }
        # canonicalize the memory metric to a string
        self.canonical_memory_metrics = {
            '':  'B',  'B':  'B',
            'K': 'KB', 'KB': 'KB',
            'M': 'MB', 'MB': 'MB',
            'G': 'GB', 'GB': 'GB',
        }

        self.time_metric = self.canonical_time_metrics[time_metric]
        self.memory_metric = self.canonical_memory_metrics[memory_metric]

        # scale factor scale input to output
        self.time_factor = time_metric_to_factor[self.canonical_time_metrics[input_time_metric]] /\
            time_metric_to_factor[self.time_metric]
        self.memory_factor = memory_metric_to_factor[self.canonical_memory_metrics[input_memory_metric]] /\
            memory_metric_to_factor[self.memory_metric]

    def convert_time(self, t):
        return self.time_factor * t

    def convert_memory(self, m):
        return self.memory_factor * m


class DisplayRounder:
    """Round a value for display purpose."""

    def __init__(self, ndigits):
        self.ndigits = ndigits
        self.precision = pow(10, -ndigits)

    def __call__(self, v: float):
        _v = abs(v)
        if _v >= self.precision or v == 0:
            return round(v, 2)
        else:
            ndigit = abs(math.floor(math.log10(_v)))
            return round(v, ndigit)


@contextmanager
def timing(description: str, force: bool = False) -> None:
    if force or os.environ.get('TORCH_PROFILER_BENCHMARK', '0') == '1':
        start = time.time()
        yield
        elapsed_time = time.time() - start
        logger.info(f'{description}: {elapsed_time}')
    else:
        yield
