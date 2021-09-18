# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import List, Optional, Union

from collections import defaultdict

from . import consts
from .profiler.data import RunProfileData
from .profiler.memory_parser import MemoryParser, MemoryRecord
from .profiler.node import MemoryMetrics
from .utils import Canonicalizer, DisplayRounder


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
        self.memory_parser: Optional[MemoryParser] = None

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
            has_tc = False
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
                if profile.tc_ratio[gpu_id] is not None:
                    gpu_metrics_data.append({"title": "Kernel Time using Tensor Cores",
                                             "value": "{} %".format(round(profile.tc_ratio[gpu_id] * 100, 2))})
                    has_tc = True
                is_first = False
            return gpu_metrics_data, has_occupancy, has_sm_efficiency, has_tc

        def get_gpu_metrics_tooltip(has_sm_efficiency, has_occupancy, has_tc):
            tooltip_summary = "The GPU usage metrics:\n"
            tooltip = "{}\n{}".format(tooltip_summary,  consts.TOOLTIP_GPU_UTIL)
            if has_sm_efficiency:
                tooltip += "\n" + consts.TOOLTIP_SM_EFFICIENCY
            if has_occupancy:
                tooltip += "\n" + consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_OVERVIEW
            if has_tc:
                tooltip += "\n" + consts.TOOLTIP_TENSOR_CORES
            return tooltip

        data, has_occupancy, has_sm_efficiency, has_tc = get_gpu_metrics_data(self)
        tooltip = get_gpu_metrics_tooltip(has_occupancy, has_sm_efficiency, has_tc)
        return data, tooltip

    @staticmethod
    def _filtered_by_ts(events, start_ts, end_ts):
        """Returns time-ordered events of memory allocation and free"""
        if start_ts is not None and end_ts is not None:
            events = [e for e in events if start_ts <= e.ts and e.ts <= end_ts]
        elif start_ts is not None:
            events = [e for e in events if start_ts <= e.ts]
        elif end_ts is not None:
            events = [e for e in events if e.ts <= end_ts]

        return events

    @staticmethod
    def get_memory_stats(profile: Union["RunProfile", RunProfileData], start_ts=None, end_ts=None, memory_metric="K"):
        cano = Canonicalizer(memory_metric=memory_metric)
        round = DisplayRounder(ndigits=2)

        stats = profile.memory_parser.get_memory_statistics(start_ts=start_ts, end_ts=end_ts)

        result = {
            "metadata": {
                "title": "Memory View",
                "default_device": "CPU",
                "search": "Operator Name",
                "sort": f"Self Size Increase ({cano.memory_metric})"
            },
            "columns": [
                {"name": "Operator Name", "type": "string"},
                {"name": "Calls", "type": "number", "tooltip": "# of calls of the operator."},
                {"name": f"Size Increase ({cano.memory_metric})", "type": "number",
                 "tooltip": "The memory increase size include all children operators."},
                {"name": f"Self Size Increase ({cano.memory_metric})", "type": "number",
                 "tooltip": "The memory increase size associated with the operator itself."},
                {"name": "Allocation Count", "type": "number",
                    "tooltip": "The allocation count including all chidren operators."},
                {"name": "Self Allocation Count", "type": "number",
                 "tooltip": "The allocation count belonging to the operator itself."},
                {"name": f"Allocation Size ({cano.memory_metric})", "type": "number",
                 "tooltip": "The allocation size including all children operators."},
                {"name": f"Self Allocation Size ({cano.memory_metric})", "type": "number",
                 "tooltip": "The allocation size belonging to the operator itself.\nIt will sum up all allocation bytes without considering the memory free."},
            ],
            "rows": {}
        }

        for name in stats:
            these_rows = []
            result["rows"][name] = these_rows

            memory = stats[name]
            for op_name, stat in sorted(memory.items()):
                these_rows.append([
                    op_name,
                    stat[6],
                    round(cano.convert_memory(stat[MemoryMetrics.IncreaseSize])),
                    round(cano.convert_memory(stat[MemoryMetrics.SelfIncreaseSize])),
                    stat[MemoryMetrics.AllocationCount],
                    stat[MemoryMetrics.SelfAllocationCount],
                    round(cano.convert_memory(stat[MemoryMetrics.AllocationSize])),
                    round(cano.convert_memory(stat[MemoryMetrics.SelfAllocationSize])),
                ])

        for dev_name in sorted(stats.keys()):
            if dev_name.startswith("GPU"):
                result["metadata"]["default_device"] = dev_name
                break

        return result

    @staticmethod
    def get_memory_curve(
            profile: Union["RunProfile", RunProfileData],
            time_metric: str = "ms",
            memory_metric: str = "K",
            patch_for_step_plot=True,
        ):
        def get_curves_and_peaks(records: List[MemoryRecord], cano: Canonicalizer):
            """Inputs:
                records: Sorted list of MemoryRecord

            For example:
            ```py
            {
                "CPU": [# Timestamp, Total Allocated, Total Reserved, Device Total Memory, operator
                    [1, 4, 4, 1000000, "aten::add"],
                    [2, 16, 16, 1000000, "aten::empty],
                    [4, 4, 16, 1000000, "..."],
                ],
                "GPU0": ...
            }
            ```"""
            curves = defaultdict(list)
            peaks = defaultdict(float)
            for r in records:
                if r.addr == None:
                    continue
                dev = r.device_name
                ts = r.ts
                ta = r.total_allocated
                tr = r.total_reserved

                if ta != ta or tr != tr: # isnan
                    continue

                curves[dev].append([
                    cano.convert_time(ts - profile.profiler_start_ts),
                    cano.convert_memory(ta),
                    cano.convert_memory(tr),
                ])
                peaks[dev] = max(peaks[dev], ta)

            for dev in curves:
                if len(curves[dev]) == 0:
                    del curves[dev]
                    del peaks[dev]

            return curves, peaks

        # NOTE: this should have been occured in frontend
        def patch_curves_for_step_plot(curves):
            # For example, if a curve is [(0, 0), (1, 1), (2,2)], the line plot
            # is a stright line. Interpolating it as [(0, 0), (1, 0), (1, 1),
            # (2,1) (2,2)], then the line plot will work as step plot.
            new_curves = defaultdict(list)
            for dev, curve in curves.items():
                new_curve = []
                for i, p in enumerate(curve):
                    if i != 0:
                        new_curve.append(p[:1] + new_curve[-1][1:])
                    new_curve.append(p)
                new_curves[dev] = new_curve
            return new_curves

        cano = Canonicalizer(time_metric, memory_metric)

        curves, peaks = get_curves_and_peaks(profile.memory_parser.all_records, cano)
        if patch_for_step_plot:
            curves = patch_curves_for_step_plot(curves)
        peaks_formatted = {}
        totals = {}
        for dev, value in peaks.items():
            peaks_formatted[dev] = "Peak Memory Usage: {:.1f}{}".format(cano.convert_memory(value), cano.memory_metric)
            if dev != "CPU":
                try:
                    totals[dev] = cano.convert_memory(profile.gpu_infos[int(dev[3:])]["Memory Raw"])
                except BaseException:
                    pass

        devices = sorted(list(curves.keys()))
        default_device = "CPU"
        for dev in devices:
            if dev.startswith("GPU"):
                default_device = dev
                break

        return {
            "metadata": {
                "default_device": default_device,
                "devices": devices,
                "peaks": peaks_formatted,
                "totals": totals,
                "first_ts": profile.profiler_start_ts,
                "time_metric": cano.time_metric,
                "memory_metric": cano.memory_metric,
                "time_factor": cano.time_factor,
                "memory_factor": cano.memory_factor,
            },
            "columns": [
                { "name": f"Time ({cano.time_metric})", "type": "number", "tooltip": "Time since profiler starts." },
                { "name": f"Allocated ({cano.memory_metric})", "type": "number", "tooltip": "Total memory in use." },
                { "name": f"Reserved ({cano.memory_metric})", "type": "number", "tooltip": "Total reserved memory by allocator, both used and unused." },
            ],
            "rows": curves,
        }

    @staticmethod
    def get_memory_events(
            p: Union["RunProfile", RunProfileData],
            start_ts=None,
            end_ts=None,
            time_metric: str = "ms",
            memory_metric: str = "K",
        ):
        def get_op_name_or_ctx(record: MemoryRecord):
            name = record.op_name_or_unknown
            if name.startswith("aten::empty") and record.parent_op_name:
                # aten::empty can be treated as the "malloc" in pytorch
                name = f"{record.parent_op_name} ({name})"
            return name

        cano = Canonicalizer(time_metric=time_metric, memory_metric=memory_metric)
        round = DisplayRounder(ndigits=2)

        profiler_start_ts = p.profiler_start_ts
        memory_records = RunProfile._filtered_by_ts(p.memory_parser.all_records, start_ts, end_ts)

        events = defaultdict(list)
        alloc = {}  # allocation events may or may not have paired free event
        free = {}  # free events that does not have paired alloc event
        prev_ts = float("-inf")  # ensure ordered memory records is ordered
        for i, r in enumerate(memory_records):
            if r.addr is None:
                # profile json data prior to pytorch 1.10 do not have addr
                # we should ignore them
                continue
            assert prev_ts < r.ts
            prev_ts = r.ts
            addr = r.addr
            size = r.bytes
            if r.is_allocation:
                # to be matched with a release event
                alloc[addr] = i
            else:
                if addr in alloc:
                    alloc_r = memory_records[alloc[addr]]
                    alloc_ts = alloc_r.ts
                    free_ts = r.ts
                    events[alloc_r.device_name].append([
                        get_op_name_or_ctx(alloc_r),
                        round(cano.convert_memory(-size)),
                        round(cano.convert_time(alloc_ts - profiler_start_ts)),
                        round(cano.convert_time(free_ts - profiler_start_ts)),
                        round(cano.convert_time(free_ts - alloc_ts)),
                    ])
                    del alloc[addr]
                else:
                    assert addr not in free
                    free[addr] = i

        for i in alloc.values():
            r = memory_records[i]
            events[r.device_name].append([
                get_op_name_or_ctx(r),
                round(cano.convert_memory(r.bytes)),
                round(cano.convert_time(r.ts - profiler_start_ts)),
                None,
                None,
            ])

        for i in free.values():
            r = memory_records[i]
            events[r.device_name].append([
                get_op_name_or_ctx(r),
                round(cano.convert_memory(-r.bytes)),
                None,
                round(cano.convert_time(r.ts - profiler_start_ts)),
                None,
            ])

        default_device = "CPU"
        for dev_name in sorted(events.keys()):
            if dev_name.startswith("GPU"):
                default_device = dev_name
                break

        return {
            "metadata": {
                "title": "Memory Events",
                "default_device": default_device,
            },
            "columns": [
                {"name": "Operator", "type": "string", "tooltip": ""},
                {"name": f"Size ({cano.memory_metric})", "type": "number", "tooltip": ""},
                {"name": f"Allocation Time ({cano.time_metric})", "type": "number", "tooltip": ""},
                {"name": f"Release Time ({cano.time_metric})", "type": "number", "tooltip": ""},
                {"name": f"Duration ({cano.time_metric})", "type": "number", "tooltip": ""},
            ],
            "rows": events,  # in the form of { "CPU": [...], "GPU0": [...], ... }
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
