# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import List, Optional, Union

from collections import OrderedDict, defaultdict
from math import pow

from . import consts
from .profiler.data import RunProfileData
from .profiler.memory_parser import MemoryParser
from .profiler.node import MemoryMetrics
from .profiler.trace import DeviceType, MemoryEvent

class Run(object):
    """ A profiler run. For visualization purpose only.
    May contain profiling results from multiple workers. E.g. distributed scenario.
    """

    def __init__(self, name, run_dir):
        self.name = name
        self.run_dir = run_dir
        self.profiles = {}

    @property
    def workers(self):
        # get full worker list and remove the duplicated
        worker_list, _ = zip(*self.profiles.keys())
        worker_list = sorted(list(dict.fromkeys(worker_list)))
        return worker_list

    @property
    def views(self):
        view_set = set()
        for profile in self.profiles.values():
            view_set.update(profile.views)
        return sorted(list(view_set), key=lambda x: x.id)

    def get_workers(self, view):
        worker_set = set()
        for profile in self.profiles.values():
            for v in profile.views:
                if v.display_name == view:
                    worker_set.add(profile.worker)
                    break
        return sorted(list(worker_set))

    def get_spans(self, worker=None):
        if worker is not None:
            spans = [s for w, s in self.profiles.keys() if w == worker]
        else:
            spans = [s for _, s in self.profiles.keys()]

        spans = list(set(spans))
        if len(spans) == 1 and spans[0] is None:
            return None
        else:
            return sorted(spans)

    def add_profile(self, profile):
        span = profile.span
        if span is None:
            span = "default"
        else:
            span = str(span)
        self.profiles[(profile.worker, span)] = profile

    def get_profile(self, worker, span) -> "RunProfile":
        if worker is None:
            raise ValueError("the worker parameter is mandatory")

        if len(self.profiles) == 0:
            return None

        return self.profiles.get((worker, span), None)

    def get_profiles(self, *, worker=None, span=None) -> Optional[List["RunProfile"]]:
        # Note: we could not use if span to check it is None or not
        # since the span 0 will be skipped at this case.
        if worker is not None and span is not None:
            return self.profiles.get((worker, span), None)
        elif worker is not None:
            return [p for (w, s), p in self.profiles.items() if worker == w]
        elif span is not None:
            return [p for (w, s), p in self.profiles.items() if span == s]
        else:
            return self.profiles.values()

class RunProfile(object):
    """ Cooked profiling result for a worker. For visualization purpose only.
    """

    def __init__(self, worker, span):
        self.worker = worker
        self.span = span
        self.views = []
        self.has_runtime = False
        self.has_kernel = False
        self.has_communication = False
        self.has_memcpy_or_memset = False
        self.profiler_start_ts = float("inf")
        self.overview = None
        self.operation_pie_by_name = None
        self.operation_table_by_name = None
        self.operation_pie_by_name_input = None
        self.operation_table_by_name_input = None
        self.kernel_op_table = None
        self.kernel_pie = None
        self.kernel_table = None
        self.tc_pie = None
        self.trace_file_path = None
        self.gpu_ids = None
        self.gpu_utilization = None
        self.sm_efficiency = None
        self.occupancy = None
        self.gpu_util_buckets = None
        self.approximated_sm_efficiency_ranges = None
        self.gpu_infos = None

        # for memory stats and curve
        self.tid2tree = None
        self.op_list_groupby_name = None
        self.memory_events : Optional[List[MemoryEvent]] = None

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
        for gpu_id, ranges in enumerate(self.approximated_sm_efficiency_ranges):
            buckets = self.gpu_util_buckets[gpu_id]
            if len(ranges) > 0 and len(buckets) > 0:
                # Adding 1 as baseline. To avoid misleading virtualization when the max value is less than 1.
                add_trace_counter_sm_efficiency(gpu_id, buckets[0][0], buckets[0][0], 1, counter_json_list)
            for r in ranges:
                add_trace_counter_sm_efficiency(gpu_id, r[0], r[1], r[2], counter_json_list)

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
            gpu_info_columns = ["Name", "Memory", "Compute Capability"]
            for gpu_id in profile.gpu_ids:
                if not is_first:
                    # Append separator line for beautiful to see.
                    gpu_metrics_data.append({"title": "<hr/>",
                                             "value": ""})

                gpu_metrics_data.append({"title": "GPU {}:".format(gpu_id),
                                         "value": ""})
                gpu_info = profile.gpu_infos.get(gpu_id, None)
                if gpu_info is not None:
                    for key in gpu_info_columns:
                        if key in gpu_info:
                            gpu_metrics_data.append({"title": key,
                                                     "value": gpu_info[key]})

                gpu_metrics_data.append({"title": "GPU Utilization",
                                         "value": "{} %".format(
                                             round(profile.gpu_utilization[gpu_id] * 100, 2))})
                if profile.sm_efficiency[gpu_id] is not None:
                    gpu_metrics_data.append({"title": "Est. SM Efficiency",
                                             "value": "{} %".format(
                                                 round(profile.sm_efficiency[gpu_id] * 100, 2))})
                    has_sm_efficiency = True
                if profile.occupancy[gpu_id] is not None:
                    gpu_metrics_data.append({"title": "Est. Achieved Occupancy",
                                             "value": "{} %".format(round(profile.occupancy[gpu_id], 2))})
                    has_occupancy = True
                is_first = False
            return gpu_metrics_data, has_occupancy, has_sm_efficiency

        def get_gpu_metrics_tooltip(has_sm_efficiency, has_occupancy):
            tooltip_summary = "The GPU usage metrics:\n"
            tooltip = "{}\n{}".format(tooltip_summary,  consts.TOOLTIP_GPU_UTIL)
            if has_sm_efficiency:
                tooltip += "\n" + consts.TOOLTIP_SM_EFFICIENCY
            if has_occupancy:
                tooltip += "\n" + consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_OVERVIEW
            return tooltip

        data, has_occupancy, has_sm_efficiency = get_gpu_metrics_data(self)
        tooltip = get_gpu_metrics_tooltip(has_occupancy, has_sm_efficiency)
        return data, tooltip

    @staticmethod
    def get_memory_statistics(profile: Union["RunProfile", RunProfileData], start_ts=None, end_ts=None):
        if start_ts is not None and end_ts is not None:
            memory_events = [e for e in profile.memory_events if start_ts <= e.ts <= end_ts]
        elif start_ts is not None:
            memory_events = [e for e in profile.memory_events if start_ts <= e.ts]
        elif end_ts is not None:
            memory_events = [e for e in profile.memory_events if e.ts <= end_ts]
        else:
            memory_events = profile.memory_events
        memory_parser = MemoryParser(profile.tid2tree, profile.op_list_groupby_name, memory_events)
        return memory_parser.get_memory_statistics()

    @staticmethod
    def generate_memory_view(memory_stats):
        data = OrderedDict()
        result = {
            "metadata": {
                "title": "Memory View",
                "default_device": "CPU",
                "search": "Operator Name",
                "sort": "Self Size Increase (KB)"
            },
            "data": data
        }

        columns_names = [
            ("Operator Name", "string", ""),
            ("Calls", "number", "# of calls of the operator."),
            ("Size Increase (KB)", "number", "The memory increase size include all children operators."),
            ("Self Size Increase (KB)", "number", "The memory increase size associated with the operator itself."),
            ("Allocation Count", "number", "The allocation count including all chidren operators."),
            ("Self Allocation Count", "number", "The allocation count belonging to the operator itself."),
            ("Allocation Size (KB)", "number", "The allocation size including all children operators."),
            ("Self Allocation Size (KB)", "number", "The allocation size belonging to the operator itself.\nIt will sum up all allocation bytes without considering the memory free.")
        ]
        for name, memory in sorted(memory_stats.items()):
            table = {}

            # Process columns
            columns = []
            for col_name, col_type, tool_tip in columns_names:
                if tool_tip:
                    columns.append({"type": col_type, "name": col_name, "tooltip": tool_tip})
                else:
                    columns.append({"type": col_type, "name": col_name})
            table["columns"] = columns

            # Process rows
            rows = []
            for op_name, stat in sorted(memory.items()):
                rows.append([
                    op_name,
                    stat[6],
                    round(stat[MemoryMetrics.IncreaseSize] / 1024, 2),
                    round(stat[MemoryMetrics.SelfIncreaseSize] / 1024, 2),
                    stat[MemoryMetrics.AllocationCount],
                    stat[MemoryMetrics.SelfAllocationCount],
                    round(stat[MemoryMetrics.AllocationSize] / 1024, 2),
                    round(stat[MemoryMetrics.SelfAllocationSize] / 1024, 2)
                    ])
            table["rows"] = rows

            data[name] = table
        return result

    @staticmethod
    def get_memory_curve(profile: Union["RunProfile", RunProfileData], time_metric: str = "", memory_metric: str = "G"):
        def get_curves_and_timestamps(memory_events: List[MemoryEvent], time_factor, memory_factor):
            """For example:
            ```py
            {
                "CPU": [# ts, total_allocated, total_reserved
                    [1, 4, 4],
                    [2, 16, 16],
                    [4, 4, 16],
                ],
                "GPU0": ...
            }
            ```"""
            curves = defaultdict(list)
            timestamps = defaultdict(list)
            first_ts = float("inf")
            for e in memory_events:
                curve_container = None
                ts_container = None
                if e.device_type == DeviceType.CPU:
                    curve_container = curves["CPU"]
                    ts_container = timestamps["CPU"]
                elif e.device_type == DeviceType.CUDA:
                    gpuid = f"GPU{e.device_id}"
                    curve_container = curves[gpuid]
                    ts_container = timestamps[gpuid]
                else:
                    raise NotImplementedError("Unknown device type for memory curve")
                first_ts = min(first_ts, e.ts)
                curve_container.append([
                    (e.ts - profile.profiler_start_ts) * time_factor,
                    e.total_allocated * memory_factor,
                    e.total_reserved * memory_factor,
                ])
                ts_container.append(e.ts)
            return curves, timestamps

        # raw timestamp is in microsecond
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/csrc/autograd/profiler_kineto.cpp#L33
        time_metric_to_factor = {
            "micro": 1,    "microsecond": 1,    "us": 1,
            "milli": 1e-3, "millisecond": 1e-3, "ms": 1e-3,
            "":      1e-6, "second":      1e-6, "s":  1e-6,
        }
        # raw memory is in bytes
        memory_metric_to_factor = {
            "":  pow(1024,  0), "B":  pow(1024,  0),
            "K": pow(1024, -1), "KB": pow(1024, -1),
            "M": pow(1024, -2), "MB": pow(1024, -2),
            "G": pow(1024, -3), "GB": pow(1024, -3),
        }

        time_factor = time_metric_to_factor[time_metric]
        memory_factor = memory_metric_to_factor[memory_metric]
        curves, timestamps = get_curves_and_timestamps(profile.memory_events, time_factor, memory_factor)
        return {
            "metadata": {
                "title": "Memory Curves",
                "default_device": "CPU",
            },
            "columns": [
                { "name": "Time", "type": "number", "tooltip": "Time since profiler starts" },
                { "name": "Allocated", "type": "number", "tooltip": "Total memory in use." },
                { "name": "Reserved", "type": "number", "tooltip": "Total reserved memory by allocator, both used and unused." },
            ],
            "rows": curves,
            "ts": timestamps,
        }


class DistributedRunProfile(object):
    """ Profiling all workers in a view.
    """

    def __init__(self, span):
        self.worker = 'All'
        self.span = span
        self.views = []
        self.gpu_info = None
        self.steps_to_overlap = None
        self.steps_to_wait = None
        self.comm_ops = None
