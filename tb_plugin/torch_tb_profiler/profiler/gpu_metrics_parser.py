from .range_lib import *
from .trace import EventTypes
from .. import utils

logger = utils.get_logger()


# For calculating GPU utilization, and approximated SM efficiency.
class GPUMetricsParser(object):
    def __init__(self):
        # All gpu ids that used by any kernel.
        self.gpu_ids = set()
        # For calculating GPU utilization.
        self.kernel_ranges_per_device = []
        self.gpu_utilization = []
        self.gpu_util_timeline_unit_size = 0
        self.gpu_util_timeline_unit_name = ""
        self.gpu_util_json = None
        # For calculating approximated SM efficiency.
        self.blocks_per_sm_per_device = []
        self.avg_approximated_sm_efficency_per_device = []
        self.approximated_sm_efficency_ranges = []
        self.gpu_sm_efficiency_json = None
        # For calculating averaged occupancy.
        self.occupancy_per_device = []
        self.avg_occupancy_per_device = []

    def calculate_gpu_utilization(self, steps_start_time, steps_end_time):
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
            unit_str = "us"
            if unit >= 1000:
                unit /= 1000
                unit_str = "ms"
                if unit >= 1000:
                    unit /= 1000
                    unit_str = "s"
            return int(bucket_size), int(buckets), int(unit), unit_str

        def build_trace_counter(gpu_id, start_time, counter_value):
            util_json = ", {{\"ph\":\"C\", \"name\":\"GPU {} Utilization\", " \
                        "\"pid\":{}, \"ts\":{}, " \
                        "\"args\":{{\"GPU Utilization\":{}}}}}".format(
                gpu_id, gpu_id, start_time, counter_value
            )
            return util_json

        counter_json = ""
        all_steps_range = (steps_start_time, steps_end_time)
        gpu_utilization_timeline = []
        for gpu_id in self.gpu_ids:
            self.kernel_ranges_per_device[gpu_id] = merge_ranges(self.kernel_ranges_per_device[gpu_id])
            self.kernel_ranges_per_device[gpu_id] = intersection_ranges_lists(
                self.kernel_ranges_per_device[gpu_id], [all_steps_range])
            ranges_sum = get_ranges_sum(self.kernel_ranges_per_device[gpu_id])
            self.gpu_utilization.append(ranges_sum / (all_steps_range[1] - all_steps_range[0]))

            bucket_size, buckets, self.gpu_util_timeline_unit_size, self.gpu_util_timeline_unit_name = \
                get_bucket_info(all_steps_range[1] - all_steps_range[0])
            gpu_utilization_timeline.append([0] * buckets)
            if len(self.kernel_ranges_per_device[gpu_id]) > 0:
                current_range_index = 0
                current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                current_bucket_index = 0
                current_bucket = (all_steps_range[0], all_steps_range[0] + bucket_size)
                while current_range_index < len(self.kernel_ranges_per_device[gpu_id]) and current_bucket_index < buckets:
                    if current_bucket[1] <= current_range[0]:
                        current_bucket_index += 1
                        current_bucket = (all_steps_range[0] + current_bucket_index * bucket_size,
                                          all_steps_range[0] + (current_bucket_index + 1) * bucket_size)
                    elif current_bucket[0] >= current_range[1]:
                        current_range_index += 1
                        if current_range_index < len(self.kernel_ranges_per_device[gpu_id]):
                            current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                    else:
                        left_bound = max(current_range[0], current_bucket[0])
                        right_bound = min(current_range[1], current_bucket[1])
                        gpu_utilization_timeline[-1][current_bucket_index] += (right_bound - left_bound)
                        if current_bucket[1] < current_range[1]:
                            current_bucket_index += 1
                            current_bucket = (all_steps_range[0] + current_bucket_index * bucket_size,
                                              all_steps_range[0] + (current_bucket_index + 1) * bucket_size)
                        else:
                            current_range_index += 1
                            if current_range_index < len(self.kernel_ranges_per_device[gpu_id]):
                                current_range = self.kernel_ranges_per_device[gpu_id][current_range_index]
                for i_bucket in range(buckets):
                    gpu_utilization_timeline[-1][i_bucket] /= bucket_size

                for i_bucket in range(buckets):
                    start_time = steps_start_time + i_bucket * bucket_size
                    util_json = build_trace_counter(gpu_id, start_time, gpu_utilization_timeline[-1][i_bucket])
                    counter_json += util_json
                start_time = steps_start_time + buckets * bucket_size
                util_json = build_trace_counter(gpu_id, start_time, 0)
                counter_json += util_json

        self.kernel_ranges_per_device = None  # Release memory.

        counter_json = bytes(counter_json, 'utf-8')
        return counter_json

    def calculate_approximated_sm_efficency(self, steps_start_time, steps_end_time):
        def calculate_avg(approximated_sm_efficency_ranges, total_dur):
            total_weighted_sm_efficiency = 0.0
            for r in approximated_sm_efficency_ranges:
                dur = r[0][1] - r[0][0]
                total_weighted_sm_efficiency += r[1] * dur
            avg_approximated_sm_efficency = total_weighted_sm_efficiency / total_dur
            return avg_approximated_sm_efficency

        total_dur = steps_end_time - steps_start_time
        for gpu_id in self.gpu_ids:
            blocks_per_sm_ranges = self.blocks_per_sm_per_device[gpu_id]
            approximated_sm_efficency_ranges = merge_ranges_with_value(blocks_per_sm_ranges)
            avg_approximated_sm_efficency = calculate_avg(approximated_sm_efficency_ranges, total_dur)
            self.avg_approximated_sm_efficency_per_device[gpu_id] = avg_approximated_sm_efficency

            if avg_approximated_sm_efficency > 0.0:
                self.approximated_sm_efficency_ranges[gpu_id] = approximated_sm_efficency_ranges

        self.blocks_per_sm_per_device = None  # Release memory.

    # Weighted average. Weighted by kernel's time duration.
    def calculate_occupancy(self):
        for gpu_id in self.gpu_ids:
            occupancys_on_a_device = self.occupancy_per_device[gpu_id]
            total_time = 0
            total_occupancy = 0.0
            for r in occupancys_on_a_device:
                dur = r[0][1] - r[0][0]
                total_occupancy += r[1] * dur
                total_time += dur
            avg_occupancy = total_occupancy / total_time
            self.avg_occupancy_per_device[gpu_id] = avg_occupancy

    def parse_events(self, events, steps_start_time, steps_end_time):
        logger.debug("GPU Metrics, parse events")
        for event in events:
            self.parse_event(event)

        self.gpu_util_json = self.calculate_gpu_utilization(steps_start_time, steps_end_time)
        self.gpu_sm_efficiency_json = self.calculate_approximated_sm_efficency(steps_start_time, steps_end_time)
        self.calculate_occupancy()

    def parse_event(self, event):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            gpu_id = event.args.get("device", None)
            if gpu_id is not None:
                if gpu_id not in self.gpu_ids:
                    self.kernel_ranges_per_device.extend([[] for _ in range(gpu_id + 1 - len(self.gpu_ids))])
                    self.blocks_per_sm_per_device.extend([[] for _ in range(gpu_id + 1 - len(self.gpu_ids))])
                    self.occupancy_per_device.extend([[] for _ in range(gpu_id + 1 - len(self.gpu_ids))])
                    self.avg_approximated_sm_efficency_per_device.extend(
                        [None] * (gpu_id + 1 - len(self.gpu_ids)))
                    self.avg_occupancy_per_device.extend(
                        [None] * (gpu_id + 1 - len(self.gpu_ids)))
                    self.approximated_sm_efficency_ranges.extend(
                        [[] for _ in range(gpu_id + 1 - len(self.gpu_ids))])
                    self.gpu_ids.add(gpu_id)
                self.kernel_ranges_per_device[gpu_id].append((ts, ts + dur))
                self.blocks_per_sm_per_device[gpu_id].append(((ts, ts + dur), event.args.get("blocks per SM", 0.0)))
                self.occupancy_per_device[gpu_id].append(((ts, ts + dur),
                                                          event.args.get("theoretical occupancy %", 0.0)))
