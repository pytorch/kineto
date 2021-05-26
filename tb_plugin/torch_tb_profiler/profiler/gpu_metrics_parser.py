from .range_utils import *
from .trace import EventTypes
from .. import utils
from .. import consts

logger = utils.get_logger()


# For calculating GPU utilization, and approximated SM efficiency.
class GPUMetricsParser(object):
    def __init__(self):
        # All gpu ids that used by any kernel.
        self.gpu_ids = set()
        # For calculating GPU utilization.
        self.kernel_ranges_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.gpu_utilization = [None] * consts.MAX_GPU_PER_NODE
        self.gpu_util_timeline_unit_size = 0
        self.gpu_util_timeline_unit_name = ""
        self.gpu_util_buckets = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        # For calculating approximated SM efficiency.
        self.blocks_per_sm_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.avg_approximated_sm_efficency_per_device = [None] * consts.MAX_GPU_PER_NODE
        self.approximated_sm_efficency_ranges = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.gpu_sm_efficiency_json = None
        self.blocks_per_sm_count = 0
        # For calculating averaged occupancy.
        self.occupancy_per_device = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        self.avg_occupancy_per_device = [None] * consts.MAX_GPU_PER_NODE
        self.occupancy_count = 0

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
            unit_str = "us"
            if unit >= 1000:
                unit /= 1000
                unit_str = "ms"
                if unit >= 1000:
                    unit /= 1000
                    unit_str = "s"
            return int(bucket_size), int(buckets), int(unit), unit_str

        gpu_utilization_timeline = [[] for _ in range(consts.MAX_GPU_PER_NODE)]
        for gpu_id in self.gpu_ids:
            self.kernel_ranges_per_device[gpu_id] = merge_ranges(self.kernel_ranges_per_device[gpu_id])

            # Top-level number still consider steps, to be consistent with overview's breakdown.
            kernel_ranges_all_steps = intersection_ranges_lists(
                self.kernel_ranges_per_device[gpu_id], [(steps_start_time, steps_end_time)])
            ranges_sum = get_ranges_sum(kernel_ranges_all_steps)
            self.gpu_utilization[gpu_id] = ranges_sum / (steps_end_time - steps_start_time)

            # The timeline will use "PyTorch Profiler (0)" as start,
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
                while current_range_index < len(self.kernel_ranges_per_device[gpu_id]) and current_bucket_index < buckets:
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
                dur = r[1] - r[0]
                total_occupancy += r[2] * dur
                total_time += dur
            avg_occupancy = total_occupancy / total_time
            self.avg_occupancy_per_device[gpu_id] = avg_occupancy

    def parse_events(self, events, global_start_time, global_end_time, steps_start_time, steps_end_time):
        logger.debug("GPU Metrics, parse events")
        for event in events:
            self.parse_event(event)

        self.calculate_gpu_utilization(global_start_time, global_end_time, steps_start_time, steps_end_time)
        self.calculate_approximated_sm_efficency(steps_start_time, steps_end_time)
        self.calculate_occupancy()

    def parse_event(self, event):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            gpu_id = event.args.get("device", None)
            if gpu_id != event.pid:
                logger.warning("pid '{}' is not equal to args.device '{}' on event with ts '{}'".format(
                    event.pid, gpu_id, event.ts))
            if gpu_id is not None:
                if gpu_id not in self.gpu_ids:
                    self.gpu_ids.add(gpu_id)
                self.kernel_ranges_per_device[gpu_id].append((ts, ts + dur))
                self.blocks_per_sm_per_device[gpu_id].append((ts, ts + dur, event.args.get("blocks per SM", 0.0)))
                self.occupancy_per_device[gpu_id].append((ts, ts + dur,
                                                          event.args.get("est. achieved occupancy %", 0.0)))
                if "blocks per SM" in event.args:
                    self.blocks_per_sm_count += 1
                if "est. achieved occupancy %" in event.args:
                    self.occupancy_count += 1
