# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
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

    def get_spans(self, worker):
        profile = self.profiles.get(worker)
        if profile is None:
            return None

        if isinstance(profile, list):
            return [span for span, p in profile]
        else:
            return None

    def get_views(self, worker, span):
        profile = self.get_profile(worker, span)
        if profile is None:
            return None

        return profile.views

    def add_profile(self, span, profile):
        if span:
            self.profiles.setdefault(profile.worker, []).append((span, profile))
        else:
            self.profiles[profile.worker] = profile

    def get_profile(self, worker, span):
        if not worker:
            raise ValueError("the worker parameter is mandatory")

        if len(self.profiles) == 0:
            return None

        data = self.profiles.get(worker, None)
        if isinstance(data, list):
            for s, p in data:
                if s == span:
                    return p
            else:
                return None
        else:
            return data

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
