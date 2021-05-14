import sys
from enum import IntEnum

from .. import utils
from .trace import EventTypes

logger = utils.get_logger()

ProfileRole = IntEnum('ProfileRole', ['Kernel', 'Memcpy', 'Memset', 'Communication', 'Runtime', 'DataLoader', 'CpuOp', 'Other', 'Total'], start=0)

class StepParser:
    def __init__(self):
        # we could not use [[]] * len here since they all point to same memory
        # https://stackoverflow.com/questions/12791501/python-initializing-a-list-of-lists
        # https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly
        self.role_ranges = [[] for _ in range(ProfileRole.Total - 1)]
        self.steps = []
        self.steps_names = []
        self.min_ts = sys.maxsize
        self.max_ts = -sys.maxsize - 1

    def parse_events(self, events, node_parser):
        for event in events:
            self._parse_event(event, node_parser.communication_data)

        if len(self.steps) == 0:
            self.steps.append((self.min_ts, self.max_ts))
            self.steps_names.append("0")
        self._update_steps_consider_device_side(node_parser.runtime_node_list, node_parser.device_node_list)

        for i in range(len(self.role_ranges)):
            self.role_ranges[i] = merge_ranges(self.role_ranges[i])

    @property
    def has_runtime(self):
        return bool(self.role_ranges[ProfileRole.Runtime])

    @property
    def has_kernel(self):
        return bool(self.role_ranges[ProfileRole.Kernel])

    @property
    def has_communication(self):
        return bool(self.role_ranges[ProfileRole.Communication])

    @property
    def has_memcpy_or_memset(self):
        return bool(self.role_ranges[ProfileRole.Memcpy] or self.role_ranges[ProfileRole.Memset])

    def _parse_event(self, event, communication_data):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        external_id = event.external_id
        if evt_type == EventTypes.KERNEL:
            if external_id in communication_data:
                self.role_ranges[ProfileRole.Communication].append((ts, ts + dur))
                communication_data[external_id].kernel_ranges.append((ts, ts + dur))
                communication_data[external_id].total_time += dur
            else:
                self.role_ranges[ProfileRole.Kernel].append((ts, ts + dur))
        elif evt_type == EventTypes.MEMCPY:
            self.role_ranges[ProfileRole.Memcpy].append((ts, ts + dur))
        elif evt_type == EventTypes.MEMSET:
            self.role_ranges[ProfileRole.Memset].append((ts, ts + dur))
        elif evt_type == EventTypes.RUNTIME:
            self.role_ranges[ProfileRole.Runtime].append((ts, ts + dur))
        elif evt_type == EventTypes.OPERATOR and event.name.startswith("enumerate(DataLoader)#") \
                and event.name.endswith(".__next__"):
            self.role_ranges[ProfileRole.DataLoader].append((ts, ts + dur))
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR]:
            self.role_ranges[ProfileRole.CpuOp].append((ts, ts + dur))

        # Record host side min and max time.
        if evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
            if ts < self.min_ts:
                self.min_ts = ts
            if ts + dur > self.max_ts:
                self.max_ts = ts + dur

    # Update self.steps considering device side events launched by each host side step.
    # Update self.steps_names if some tail steps are removed.
    def _update_steps_consider_device_side(self, runtime_node_list, device_node_list):
        runtime_node_list = sorted(runtime_node_list, key=lambda x: x.start_time)
        # Make sure self.steps is sorted by time.
        self.steps = sorted(self.steps, key=lambda x: x[0])
        # Use similar code with two-way merge to get all runtimes inside each host-side step span,
        # then record each step's min kernel start time and max kernel end time:
        steps_device = [(sys.maxsize, -sys.maxsize - 1)] * len(self.steps)
        steps_matched_device_nodes = [0] * len(self.steps)
        i_step = 0
        i_runtime = 0
        step_device_min_ts = sys.maxsize
        step_device_max_ts = -sys.maxsize - 1
        matched_device_nodes = set()
        while i_step < len(self.steps) and i_runtime < len(runtime_node_list):
            step_host_start_time = self.steps[i_step][0]
            step_host_end_time = self.steps[i_step][1]
            if runtime_node_list[i_runtime].start_time < step_host_start_time:
                # This runtime is ahead of or intersects with this step span. Skip this runtime.
                i_runtime += 1
            elif runtime_node_list[i_runtime].end_time <= step_host_end_time:
                # and runtime_node_list[i_runtime].start_time >= step_host_start_time
                # This runtime is inside this step span. Scan its device_nodes.
                rt = runtime_node_list[i_runtime]
                if rt.device_nodes is not None:
                    for device_node in rt.device_nodes:
                        step_device_min_ts = min(device_node.start_time, step_device_min_ts)
                        step_device_max_ts = max(device_node.end_time, step_device_max_ts)
                        matched_device_nodes.add(device_node)
                        steps_matched_device_nodes[i_step] += 1
                i_runtime += 1
            elif runtime_node_list[i_runtime].start_time < step_host_end_time:
                # and runtime_node_list[i_runtime].end_time > step_host_end_time
                # This runtime intersects with this step span. Skip this runtime.
                i_runtime += 1
            else:
                # runtime_node_list[i_runtime].start_time >= step_host_end_time
                # This runtime starts after this step's end. Record and move forward this step.
                steps_device[i_step] = (step_device_min_ts, step_device_max_ts)
                i_step += 1
                step_device_min_ts = sys.maxsize
                step_device_max_ts = -sys.maxsize - 1
        while i_step < len(self.steps):
            # This step doesn't launch any device side event, just assign it as empty.
            steps_device[i_step] = (step_device_min_ts, step_device_max_ts)
            step_device_min_ts = sys.maxsize
            step_device_max_ts = -sys.maxsize - 1
            i_step += 1
        # Change step time to device side on the condition that any step have device time.
        is_use_gpu = (len(matched_device_nodes) > 0)
        if is_use_gpu:
            prev_step_end_time = self.steps[0][0]
            if steps_device[0][0] != sys.maxsize:  # When step 0 has device event.
                for device_node in device_node_list:
                    if device_node not in matched_device_nodes:
                        # Now this device_node is not launched inside any step span.
                        if device_node.end_time < steps_device[0][0]:
                            prev_step_end_time = max(prev_step_end_time, device_node.end_time)
            for i_step in range(len(self.steps)):
                step_start_time = max(prev_step_end_time, self.steps[i_step][0])
                step_end_time = self.steps[i_step][1]
                if steps_device[i_step][0] == sys.maxsize:  # When step i_step has no device event.
                    # Assign to step_start_time when kernel is behind host step end.
                    step_end_time = max(step_end_time, step_start_time)
                else:
                    step_end_time = max(step_end_time, steps_device[i_step][1])
                    if step_end_time < step_start_time:
                        logger.warning(
                            "Abnormal step_end_time of step {}: [{}, {}]".format(
                                i_step, step_start_time, step_end_time))
                        step_end_time = step_start_time
                self.steps[i_step] = (step_start_time, step_end_time)  # Update step time considering device side.
                prev_step_end_time = step_end_time

        is_remove_tail_steps = True  # TODO: Use tensorboard argument instead.
        if is_use_gpu and len(self.steps) > 1 and is_remove_tail_steps:
            i_step = len(self.steps) - 1
            while i_step >= 0:
                if steps_matched_device_nodes[i_step] > 0:
                    break
                i_step -= 1
            if i_step >= 0:
                keep_steps = i_step + 1
                if i_step > 0 and steps_matched_device_nodes[i_step - 1] * 0.8 > steps_matched_device_nodes[i_step]:
                    keep_steps = i_step
                if keep_steps < len(self.steps):
                    logger.warning(
                        "Remove the last {} steps from overview. "
                        "Because the profiler may fail to capture all the kernels launched by these steps.".format(
                            len(self.steps) - keep_steps
                        ))
                    self.steps = self.steps[:keep_steps]
                    self.steps_names = self.steps_names[:keep_steps]

def merge_ranges(src_ranges, is_sorted=False):
    merged_ranges = []
    if len(src_ranges) > 0:
        if not is_sorted:
            src_ranges.sort(key=lambda x: x[0])
        src_id = 0
        merged_ranges.append(
            (src_ranges[src_id][0], src_ranges[src_id][1]))
        for src_id in range(1, len(src_ranges)):
            dst_id = len(merged_ranges) - 1
            if src_ranges[src_id][1] > merged_ranges[dst_id][1]:
                if src_ranges[src_id][0] <= merged_ranges[dst_id][1]:
                    merged_ranges[dst_id] = (merged_ranges[dst_id][0], src_ranges[src_id][1])
                else:
                    merged_ranges.append(
                        (src_ranges[src_id][0], src_ranges[src_id][1]))
    return merged_ranges
