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
        # get full worker list and remove the duplicated
        worker_list, _ = zip(*self.profiles.keys())
        worker_list = list(dict.fromkeys(worker_list))
        return worker_list

    def get_spans(self, worker):
        spans = [s for w, s in self.profiles.keys() if w == worker]
        if len(spans) == 1 and spans[0] is None:
            return None
        else:
            return spans

    def get_views(self, worker, span):
        profile = self.get_profile(worker, span)
        if profile is None:
            return None

        return profile.views

    def add_profile(self, profile):
        self.profiles[(profile.worker, profile.span)] = profile

    def get_profile(self, worker, span):
        if not worker:
            raise ValueError("the worker parameter is mandatory")

        if len(self.profiles) == 0:
            return None

        return self.profiles.get((worker, span), None)

class RunProfile(object):
    """ Cooked profiling result for a worker. For visualization purpose only.
    """

    def __init__(self, worker, span):
        self.worker = worker
        self.span = span
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
