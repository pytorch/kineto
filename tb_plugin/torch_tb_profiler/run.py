# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict


class Run(object):
    """ A profiler run. For visualization purpose only.
    May contain profiling results from multiple workers. E.g. distributed scenario.
    """

    def __init__(self, name, run_dir):
        self.name = name
        self.run_dir = run_dir
        self.profiles = OrderedDict()

    @property
    def workers(self):
        return list(self.profiles.keys())

    @property
    def views(self):
        profile = self.get_profile()
        if profile is None:
            return None
        return profile.views

    def add_profile(self, profile):
        self.profiles[profile.worker] = profile

    def get_profile(self, worker=None):
        if len(self.profiles) == 0:
            return None
        if not worker:
            return next(iter(self.profiles.values()))
        return self.profiles.get(worker, None)


class RunProfile(object):
    """ Cooked profiling result for a worker. For visualization purpose only.
    """

    def __init__(self, worker):
        self.worker = worker
        self.views = []
        self.has_runtime = False
        self.has_kernel = False
        self.has_memcpy_or_memset = False
        self.overview = None
        self.operation_pie_by_name = None
        self.operation_table_by_name = None
        self.operation_pie_by_name_input = None
        self.operation_table_by_name_input = None
        self.kernel_op_table = None
        self.kernel_pie = None
        self.kernel_table = None
        self.trace_file_path = None
