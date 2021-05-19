# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from collections import OrderedDict

from . import io


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
        self.has_communication = False
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
        self.gpu_ids = None
        self.gpu_utilization = None
        self.sm_efficency = None
        self.occupancy = None
        self.gpu_util_buckets = None
        self.approximated_sm_efficency_ranges = None

    def get_gpu_metrics(self):
        def build_trace_counter_gpu_util(gpu_id, start_time, counter_value):
            util_json = "{{\"ph\":\"C\", \"name\":\"GPU {} Utilization\", " \
                        "\"pid\":{}, \"ts\":{}, " \
                        "\"args\":{{\"GPU Utilization\":{}}}}}".format(
                gpu_id, gpu_id, start_time, counter_value
            )
            return util_json

        def build_trace_counter_sm_efficiency(gpu_id, start_time, counter_value):
            util_json = "{{\"ph\":\"C\", \"name\":\"GPU {} Est. SM Efficiency\", " \
                        "\"pid\":{}, \"ts\":{}, " \
                        "\"args\":{{\"Est. SM Efficiency\":{}}}}}".format(
                gpu_id, gpu_id, start_time, counter_value
            )
            return util_json

        counter_json_list = []
        for gpu_id, buckets in enumerate(self.gpu_util_buckets):
            for b in buckets:
                json_str = build_trace_counter_gpu_util(gpu_id, b[0], b[1])
                counter_json_list.append(json_str)
        for gpu_id, ranges in enumerate(self.approximated_sm_efficency_ranges):
            for r in ranges:
                efficiency_json_start = build_trace_counter_sm_efficiency(gpu_id, r[0][0], r[1])
                efficiency_json_finish = build_trace_counter_sm_efficiency(gpu_id, r[0][1], 0)
                counter_json_list.append(efficiency_json_start)
                counter_json_list.append(efficiency_json_finish)

        counter_json_str = ", {}".format(", ".join(counter_json_list))
        counter_json_bytes = bytes(counter_json_str, 'utf-8')
        return counter_json_bytes

    def append_gpu_metrics(self, raw_data):
        counter_json_bytes = self.get_gpu_metrics()

        raw_data_without_tail = raw_data[: raw_data.rfind(b']')]
        raw_data = b''.join([raw_data_without_tail, counter_json_bytes, b']}'])

        import gzip
        raw_data = gzip.compress(raw_data, 1)
        return raw_data

class DistributedRunProfile(object):
    """ Profiling all workers in a view.
    """

    def __init__(self):
        self.worker = 'All'
        self.views = []
        self.steps_to_overlap = None
        self.steps_to_wait = None
        self.comm_ops = None
