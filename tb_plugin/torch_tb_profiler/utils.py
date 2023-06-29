# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import logging
import math
import os
import time
from contextlib import contextmanager
from math import pow
import numpy as np

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


def _areas_of_triangles(a, bs, c):
    """Calculate areas of triangles from duples of vertex coordinates.

    Uses implicit numpy broadcasting along first axis of ``bs``.

    Returns
    -------
    numpy.array
        Array of areas of shape (len(bs),)
    """
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * abs(
        (a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1])
    )


def lttb_sample(memory_curves, n_out = 10240):
    """
    sample ``memory_curves`` to ``n_out`` points using the LTTB algorithm.

    Parameters
    ----------
    memory_curves : dict(str, list(list(time,allocated,reverved)))
        A dict, key for device (cpu, gpu0, gpu1, ...), 
        value is a list of list of (time,allocated,reverved)
    n_out : int
        Number of data points to downsample to
    
    Returns
    -------
    sumpled memory_curves with at most n_out points.
    """
    sampled_memory_curves = {}
    for key in memory_curves:
        data = memory_curves[key]
        length = len(data)
        if n_out >= length:
            sampled_memory_curves[key] = memory_curves[key]
            continue

        # Split data into bins
        n_bins = n_out - 2
        data = np.array(data)
        data_bins = np.array_split(data[1 : length - 1], n_bins)

        # Prepare output array
        # First and last points are the same as in the input.
        out = np.zeros((n_out, 3))
        out[0] = data[0]
        out[len(out) - 1] = data[length - 1]

        # note that we only need to perform LTTB on (time,allocated)
        # Largest Triangle Three Buckets (LTTB):
        # In each bin, find the point that makes the largest triangle
        # with the point saved in the previous bin
        # and the centroid of the points in the next bin.
        for i in range(len(data_bins)):
            this_bin = data_bins[i]
            if i < n_bins - 1:
                next_bin = data_bins[i + 1]
            else:
                next_bin = data[len(data) - 1 :]
            a = out[i]
            bs = this_bin
            c = next_bin.mean(axis=0)
            areas = _areas_of_triangles(a, bs, c)
            out[i + 1] = bs[np.argmax(areas)]

        sampled_memory_curves[key] = out.tolist()
    return sampled_memory_curves
