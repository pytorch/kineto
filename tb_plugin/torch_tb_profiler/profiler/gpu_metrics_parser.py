# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from typing import Iterable, List

from .. import consts, utils
from .range_utils import (get_ranges_sum, intersection_ranges_lists,
                          intersection_ranges_lists_with_value, merge_ranges,
                          merge_ranges_with_value)
from .trace import BaseEvent, EventTypes, KernelEvent

logger = utils.get_logger()


# For calculating GPU utilization, and approximated SM efficiency.
class GPUMetricsParser:
    def __init__(self):
        # All gpu ids that used by any kernel.
        self.gpu_ids = set()
        # For calculating GPU utilization.
        self.kernel_ranges_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.gpu_utilization = [None] * consts.MAX_GPU_PER_NODE
        self.gpu_util_timeline_unit_size = 0
        self.gpu_util_timeline_unit_name = ''
        self.gpu_util_buckets = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        # For calculating approximated SM efficiency.
        self.blocks_per_sm_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.avg_approximated_sm_efficiency_per_device = [None] * consts.MAX_GPU_PER_NODE
        self.approximated_sm_efficiency_ranges = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.gpu_sm_efficiency_json = None
        self.blocks_per_sm_count = [0] * consts.MAX_GPU_PER_NODE
        # For calculating averaged occupancy.
        self.occupancy_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.avg_occupancy_per_device = [None] * consts.MAX_GPU_PER_NODE
        self.occupancy_count = [0] * consts.MAX_GPU_PER_NODE

    def calculate_gpu_utilization(self, global_start_time, global_end_time, steps_start_time, steps_end_time):
        # Make bucket_size to 10-power's of us, and number of buckets to (10, 100].
        # 10-power's of us, in order to straight forward for user to understand.
        # If number of buckets are too many, the value of gpu utilization will be either 0 or 1.
        def get_bucket_info(range_micro_seconds):
            max_buckets = 100
            bucket_size = 1
            while range_micro_seconds / bucket_size > max_buckets:
                bucket_size *= 10
            buckets = int(range_micro_seconds / bucket_size)
            unit = bucket_size
            unit_str = 'us'
            if unit >= 1000:
                unit /= 1000
                unit_str = 'ms'
                if unit >= 1000:
                    unit /= 1000
                    unit_str = 's'
            return int(bucket_size), int(buckets), int(unit), unit_str

        gpu_utilization_timeline = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        for gpu_id in self.gpu_ids:
            self.kernel_ranges_per_device[gpu_id] = merge_ranges(self.kernel_ranges_per_device[gpu_id])

            # Top-level number still consider steps, to be consistent with overview's breakdown.
            kernel_ranges_all_steps = intersection_ranges_lists(
                self.kernel_ranges_per_device[gpu_id], [(steps_start_time, steps_end_time)])
            ranges_sum = get_ranges_sum(kernel_ranges_all_steps)
            self.gpu_utilization[gpu_id] = ranges_sum / (steps_end_time - steps_start_time)

            # The timeline will use 'PyTorch Profiler (0)' as start,
            # in order to draw previous step's kernels' gpu utilization.
            bucket_size, buckets, self.gpu_util_timeline_unit_size, self.gpu_util_timeline_unit_name = \
                get_bucket_info(global_end_time - global_start_time)
            buckets_ranges = []
            for i in range(buckets):
                buckets_ranges.append((global_start_time + i * bucket_size,
                                       global_start_time + (i + 1) * bucket_size if i < buckets - 1
                                       else global_end_time))  # The last bucket may be longer.
            gpu_utilization_timeline[gpu_id] = [0] * buckets
            if len(self.kernel_ranges_per_device[gpu_id]) > 0:
                current_range_index = 0
                current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                current_bucket_index = 0
                current_bucket = buckets_ranges[0]
                while (current_range_index < len(self.kernel_ranges_per_device[gpu_id])
                       and current_bucket_index < buckets):
                    if current_bucket[1] <= current_range[0]:
                        current_bucket_index += 1
                        current_bucket = buckets_ranges[current_bucket_index] if current_bucket_index < buckets \
                            else None
                    elif current_bucket[0] >= current_range[1]:
                        current_range_index += 1
                        if current_range_index < len(self.kernel_ranges_per_device[gpu_id]):
                            current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                    else:
                        left_bound = max(current_range[0], current_bucket[0])
                        right_bound = min(current_range[1], current_bucket[1])
                        gpu_utilization_timeline[gpu_id][current_bucket_index] += (right_bound - left_bound)
                        if current_bucket[1] < current_range[1]:
                            current_bucket_index += 1
                            current_bucket = buckets_ranges[current_bucket_index] if current_bucket_index < buckets \
                                else None
                        else:
                            current_range_index += 1
                            if current_range_index < len(self.kernel_ranges_per_device[gpu_id]):
                                current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                for i_bucket in range(buckets):
                    bucket_size = buckets_ranges[i_bucket][1] - buckets_ranges[i_bucket][0]
                    gpu_utilization_timeline[gpu_id][i_bucket] /= bucket_size
                    start_time = buckets_ranges[i_bucket][0]
                    self.gpu_util_buckets[gpu_id].append((start_time, gpu_utilization_timeline[gpu_id][i_bucket]))
                start_time = buckets_ranges[-1][1]
                self.gpu_util_buckets[gpu_id].append((start_time, 0))

        self.kernel_ranges_per_device = None  # Release memory.

    def calculate_approximated_sm_efficiency(self, steps_start_time, steps_end_time):
        def calculate_avg(approximated_sm_efficiency_ranges, total_dur):
            total_weighted_sm_efficiency = 0.0
            for r in approximated_sm_efficiency_ranges:
                dur = r[1] - r[0]
                total_weighted_sm_efficiency += r[2] * dur
            avg_approximated_sm_efficiency = total_weighted_sm_efficiency / total_dur
            return avg_approximated_sm_efficiency

        total_dur = steps_end_time - steps_start_time
        for gpu_id in self.gpu_ids:
            blocks_per_sm_ranges = self.blocks_per_sm_per_device[gpu_id]
            approximated_sm_efficiency_ranges = merge_ranges_with_value(blocks_per_sm_ranges)
            # To be consistent with GPU utilization, here it must also intersect with all steps,
            # in order to remove the kernels out of steps range.
            approximated_sm_efficiency_ranges_all_steps = intersection_ranges_lists_with_value(
                approximated_sm_efficiency_ranges, [(steps_start_time, steps_end_time)])
            if len(approximated_sm_efficiency_ranges_all_steps) > 0:
                avg_approximated_sm_efficiency = calculate_avg(approximated_sm_efficiency_ranges_all_steps, total_dur)
                self.avg_approximated_sm_efficiency_per_device[gpu_id] = avg_approximated_sm_efficiency

            # The timeline still uses all kernels including out of steps scope's.
            if len(approximated_sm_efficiency_ranges) > 0:
                self.approximated_sm_efficiency_ranges[gpu_id] = approximated_sm_efficiency_ranges

        self.blocks_per_sm_per_device = None  # Release memory.

    # Weighted average. Weighted by kernel's time duration.
    def calculate_occupancy(self, steps_start_time, steps_end_time):
        for gpu_id in self.gpu_ids:
            occupancys_on_a_device = self.occupancy_per_device[gpu_id]
            total_time = 0
            total_occupancy = 0.0
            for r in occupancys_on_a_device:
                min_time = max(r[0], steps_start_time)
                max_time = min(r[1], steps_end_time)
                if min_time < max_time:
                    dur = max_time - min_time
                    total_occupancy += r[2] * dur
                    total_time += dur
            if total_time > 0:
                self.avg_occupancy_per_device[gpu_id] = total_occupancy / total_time

    @classmethod
    def parse_events(cls,
                     events: Iterable[BaseEvent],
                     global_start_time: int,
                     global_end_time: int,
                     steps_start_time: int,
                     steps_end_time: int):
        parser = GPUMetricsParser()
        logger.debug('GPU Metrics, parse events')
        for event in events:
            if event.type == EventTypes.KERNEL:
                parser.parse_event(event)

        parser.calculate_gpu_utilization(global_start_time, global_end_time, steps_start_time, steps_end_time)
        parser.calculate_approximated_sm_efficiency(steps_start_time, steps_end_time)
        parser.calculate_occupancy(steps_start_time, steps_end_time)
        return parser

    def parse_event(self, event: KernelEvent):
        ts = event.ts
        dur = event.duration
        gpu_id = event.device_id
        if gpu_id != event.pid:
            logger.warning("pid '{}' is not equal to args.device '{}' on event with ts '{}'".format(
                event.pid, gpu_id, event.ts))
        if gpu_id is not None:
            if gpu_id not in self.gpu_ids:
                self.gpu_ids.add(gpu_id)
            self.kernel_ranges_per_device[gpu_id].append((ts, ts + dur))
            if event.blocks_per_sm is not None:
                if event.blocks_per_sm > 0.0:
                    self.blocks_per_sm_per_device[gpu_id].append((ts, ts + dur, event.blocks_per_sm))
                    self.blocks_per_sm_count[gpu_id] += 1
                else:
                    # Workaround for negative value input.
                    logger.warning('blocks per SM {} with ts {} is not positive!'.format(event.blocks_per_sm, ts))
            if event.occupancy is not None:
                if event.occupancy >= 0.0:
                    self.occupancy_per_device[gpu_id].append((ts, ts + dur, event.occupancy))
                    self.occupancy_count[gpu_id] += 1
                else:
                    # Workaround for negative value input.
                    logger.warning('est. achieved occupancy % {} with ts {} is negative!'.format(event.occupancy, ts))

    def get_gpu_metrics_columns(self):
        columns = []
        if self.has_blocks_per_sm:
            columns.append({'type': 'number', 'name': 'Mean Blocks Per SM',
                            'tooltip': consts.TOOLTIP_BLOCKS_PER_SM})
        if self.has_occupancy:
            columns.append({'type': 'number', 'name': 'Mean Est. Achieved Occupancy (%)',
                            'tooltip': consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_TABLE})
        return columns

    @property
    def has_blocks_per_sm(self):
        return sum(self.blocks_per_sm_count) > 0

    @property
    def has_occupancy(self):
        return sum(self.occupancy_count) > 0

    def get_gpu_metrics(self):
        def build_trace_counter_gpu_util(gpu_id, start_time, counter_value):
            util_json = ("{{\"ph\":\"C\", \"name\":\"GPU {} Utilization\", \"pid\":{}, \"ts\":{}, "
                         "\"args\":{{\"GPU Utilization\":{}}}}}").format(gpu_id, gpu_id, start_time, counter_value)
            return util_json

        def build_trace_counter_sm_efficiency(gpu_id, start_time, counter_value):
            util_json = ("{{\"ph\":\"C\", \"name\":\"GPU {} Est. SM Efficiency\", \"pid\":{}, \"ts\":{}, "
                         "\"args\":{{\"Est. SM Efficiency\":{}}}}}").format(gpu_id, gpu_id, start_time, counter_value)
            return util_json

        def add_trace_counter_gpu_util(gpu_id, start_time, counter_value, counter_json_list: List):
            json_str = build_trace_counter_gpu_util(gpu_id, start_time, counter_value)
            counter_json_list.append(json_str)

        def add_trace_counter_sm_efficiency(gpu_id, start_time, end_time, value, counter_json_list: List):
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

        return counter_json_list

    def get_gpu_metrics_data_tooltip(
            self,
            gpu_infos,
            tc_ratio):
        if not self.gpu_ids:
            return None, None

        has_sm_efficiency = False
        has_occupancy = False
        has_tc = False

        gpu_metrics_data = []
        gpu_info_columns = ['Name', 'Memory', 'Compute Capability']

        def process_gpu(gpu_id: int):
            nonlocal has_sm_efficiency, has_occupancy, has_tc
            gpu_metrics_data.append({'title': 'GPU {}:'.format(gpu_id), 'value': ''})
            gpu_info = gpu_infos.get(gpu_id, None)
            if gpu_info is not None:
                for key in gpu_info_columns:
                    if key in gpu_info:
                        gpu_metrics_data.append({'title': key, 'value': gpu_info[key]})
            else:
                # the legacy chrome tracing file would not have gpu info.
                pass
            gpu_metrics_data.append({'title': 'GPU Utilization', 'value': '{} %'.format(
                round(self.gpu_utilization[gpu_id] * 100, 2))})
            if self.avg_approximated_sm_efficiency_per_device[gpu_id] is not None:
                gpu_metrics_data.append({'title': 'Est. SM Efficiency', 'value': '{} %'.format(
                    round(self.avg_approximated_sm_efficiency_per_device[gpu_id] * 100, 2))})
                has_sm_efficiency = True
            if self.avg_occupancy_per_device[gpu_id] is not None:
                gpu_metrics_data.append({'title': 'Est. Achieved Occupancy', 'value': '{} %'.format(
                    round(self.avg_occupancy_per_device[gpu_id], 2))})
                has_occupancy = True
            if tc_ratio[gpu_id] is not None:
                gpu_metrics_data.append({'title': 'Kernel Time using Tensor Cores', 'value': '{} %'.format(
                    round(tc_ratio[gpu_id] * 100, 2))})
                has_tc = True

        gpu_ids = list(self.gpu_ids)
        process_gpu(gpu_ids[0])
        for idx in range(1, len(gpu_ids)):
            # Append separator line for beautiful to see.
            gpu_metrics_data.append({'title': '<hr/>', 'value': ''})
            process_gpu(gpu_ids[idx])

        tooltip_summary = 'The GPU usage metrics:\n'
        tooltip = '{}\n{}'.format(tooltip_summary,  consts.TOOLTIP_GPU_UTIL)
        if has_sm_efficiency:
            tooltip += '\n' + consts.TOOLTIP_SM_EFFICIENCY
        if has_occupancy:
            tooltip += '\n' + consts.TOOLTIP_OCCUPANCY_COMMON + consts.TOOLTIP_OCCUPANCY_OVERVIEW
        if has_tc:
            tooltip += '\n' + consts.TOOLTIP_TENSOR_CORES

        return gpu_metrics_data, tooltip
