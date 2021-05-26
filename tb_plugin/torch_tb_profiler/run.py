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
        return sorted(list(self.profiles.keys()))

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

        def add_trace_counter_gpu_util(gpu_id, start_time, counter_value, counter_json_list):
            json_str = build_trace_counter_gpu_util(gpu_id, start_time, counter_value)
            counter_json_list.append(json_str)

        def add_trace_counter_sm_efficiency(gpu_id, start_time, end_time, value, counter_json_list):
            efficiency_json_start = build_trace_counter_sm_efficiency(gpu_id, start_time, value)
            efficiency_json_finish = build_trace_counter_sm_efficiency(gpu_id, end_time, 0)
            counter_json_list.append(efficiency_json_start)
            counter_json_list.append(efficiency_json_finish)

        counter_json_list = []
        for gpu_id, buckets in enumerate(self.gpu_util_buckets):
            if len(buckets) > 0:
                # Adding 1 as baseline. To avoid misleading virtualization when the max value is less than 1.
                add_trace_counter_gpu_util(gpu_id, buckets[0][0], 1, counter_json_list)
                add_trace_counter_gpu_util(gpu_id, buckets[0][0], 0, counter_json_list)
            for b in buckets:
                add_trace_counter_gpu_util(gpu_id, b[0], b[1], counter_json_list)
        for gpu_id, ranges in enumerate(self.approximated_sm_efficency_ranges):
            buckets = self.gpu_util_buckets[gpu_id]
            if len(ranges) > 0 and len(buckets) > 0:
                # Adding 1 as baseline. To avoid misleading virtualization when the max value is less than 1.
                add_trace_counter_sm_efficiency(gpu_id, buckets[0][0], buckets[0][0], 1, counter_json_list)
            for r in ranges:
                add_trace_counter_sm_efficiency(gpu_id, r[0][0], r[0][1], r[1], counter_json_list)

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


    def get_gpu_metrics_data_tooltip(self):
        def get_gpu_metrics_data(profile):
            gpu_metrics_data = []
            has_sm_efficiency = False
            has_occupancy = False
            is_first = True
            for gpu_id in profile.gpu_ids:
                if not is_first:
                    # Append separator line for beautiful to see.
                    gpu_metrics_data.append({"title": "--------",
                                             "value": ""})
                gpu_metrics_data.append({"title": "GPU {}:".format(gpu_id),
                                         "value": ""})
                gpu_metrics_data.append({"title": "GPU Utilization",
                                         "value": "{} %".format(
                                             round(profile.gpu_utilization[gpu_id] * 100, 2))})
                if profile.sm_efficency[gpu_id] > 0.0:
                    gpu_metrics_data.append({"title": "Est. SM Efficiency",
                                             "value": "{} %".format(
                                                 round(profile.sm_efficency[gpu_id] * 100, 2))})
                    has_sm_efficiency = True
                if profile.occupancy[gpu_id] > 0.0:
                    gpu_metrics_data.append({"title": "Est. Achieved Occupancy",
                                             "value": "{} %".format(round(profile.occupancy[gpu_id], 2))})
                    has_occupancy = True
                is_first = False
            return gpu_metrics_data, has_occupancy, has_sm_efficiency

        def get_gpu_metrics_tooltip(has_sm_efficiency, has_occupancy):
            tooltip_summary = "The GPU usage metrics:\n"
            tooltip_gpu_util = \
                "GPU Utilization:\n" \
                "GPU busy time / All steps time. " \
                "GPU busy time is the time during which there is at least one GPU kernel running on it. " \
                "All steps time is the total time of all profiler steps(or called as iterations).\n"
            tooltip_sm_efficiency = \
                "Est. SM Efficiency:\n" \
                "Estimated Stream Multiprocessor Efficiency. " \
                "Est. SM Efficiency of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). " \
                "This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, " \
                "divided by all steps time.\n"
            tooltip_occupancy = \
                "Est. Achieved Occupancy:\n" \
                "Occupancy is the ratio of active threads on an SM " \
                "to the maximum number of active threads supported by the SM. " \
                "The theoretical occupancy of a kernel is upper limit occupancy of this kernel, " \
                "limited by multiple factors such as kernel shape, kernel used resource, " \
                "and the GPU compute capability." \
                "Est. Achieved Occupancy of a kernel, OCC_K = " \
                "min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). " \
                "This overall number is the weighted sum of all kernels OCC_K " \
                "using kernel's execution duration as weight."
            tooltip = "{}\n{}".format(tooltip_summary, tooltip_gpu_util)
            if has_sm_efficiency:
                tooltip += "\n" + tooltip_sm_efficiency
            if has_occupancy:
                tooltip += "\n" + tooltip_occupancy
            return tooltip

        data, has_occupancy, has_sm_efficiency = get_gpu_metrics_data(self)
        tooltip = get_gpu_metrics_tooltip(has_occupancy, has_sm_efficiency)
        return data, tooltip


class DistributedRunProfile(object):
    """ Profiling all workers in a view.
    """

    def __init__(self):
        self.worker = 'All'
        self.views = []
        self.gpu_info = None
        self.steps_to_overlap = None
        self.steps_to_wait = None
        self.comm_ops = None
