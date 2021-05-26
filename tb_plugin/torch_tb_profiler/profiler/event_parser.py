# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from enum import IntEnum

from .. import utils
from .communication import generate_communication_nodes
from .node import (CommunicationNode, DeviceNode, OperatorNode,
                   ProfilerStepNode, RuntimeNode)
from .range_utils import merge_ranges
from .trace import EventTypes

logger = utils.get_logger()

CommunicationOpNameSet = ['nccl:broadcast', 'nccl:all_reduce']
ProfileRole = IntEnum('ProfileRole', ['Kernel', 'Memcpy', 'Memset', 'Communication', 'Runtime', 'DataLoader', 'CpuOp', 'Other', 'Total'], start=0)


class NodeContext:
    def __init__(self, tid2list, tid2zero_rt_list, corrid_to_device):
        self.tid2list = tid2list
        self.tid2zero_rt_list = tid2zero_rt_list
        self.corrid_to_device = corrid_to_device

class StepContext:
    def __init__(self, prev_step_end_time, steps_device, steps_matched_device_nodes):
        self.prev_step_end_time = prev_step_end_time
        self.steps_device = steps_device
        self.steps_matched_device_nodes = steps_matched_device_nodes

class NodeParserMixin:
    def __init__(self, *args, **kwargs):
        '''Please refer to https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        to see the reason why we need call super().__init__ like this way
        '''
        super().__init__(*args, **kwargs)

        self.communication_data = {}
        self.device_node_list = []
        self.runtime_node_list = []
        self.used_devices = set()
        self.use_dp = False
        self.use_ddp = False
        self.use_nccl = False

    def parse_nodes(self, events):
        # For OperatorNode and ProfilerStepNode:
        #   Use time interval containing relationship to build father-child correlation,
        #   which is consistent with autograd profiler.
        # For RuntimeNode:
        #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
        #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
        #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
        tid2list = {} # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
        tid2zero_rt_list = {}  # value is a list of RuntimeNode with external_id=0. They will be attached to root nodes.
        corrid_to_device = {}  # value is a list of DeviceNode

        corrid_to_runtime = {}  # value is a RuntimeNode
        externalid_to_runtime = {}  # value is a list of RuntimeNode

        for event in events:
            self._parse_node(event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list)

        for event in events:
            if event.type == EventTypes.KERNEL:
                self._update_communication_node(event)

        # associate CUDA Runtimes with CPU events
        for _, op_list in tid2list.items():
            for op in op_list:
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.warning("{} Runtime with external id {} don't correlate to any operator!".format(
                    len(externalid_to_runtime[ext_id]), ext_id))

        return NodeContext(tid2list, tid2zero_rt_list, corrid_to_device)

    def _update_communication_node(self, event):
        '''Update the communication node by using the TraceEvent instance'''
        external_id = event.external_id
        comm_node = self.communication_data.get(external_id)
        if comm_node:
            ts = event.ts
            dur = event.duration
            comm_node.kernel_ranges.append((ts, ts + dur))
            comm_node.total_time += dur

        return comm_node is not None

    def find_device_steps(self, steps):
        '''return steps associated with device nodes. 
        '''
        runtime_node_list = sorted(self.runtime_node_list, key=lambda x: x.start_time)

        # Use similar code with two-way merge to get all runtimes inside each host-side step span,
        # then record each step's min kernel start time and max kernel end time:
        steps_device = [(sys.maxsize, -sys.maxsize - 1)] * len(steps)
        # where the steps associated with devcie node, if yes, the related array item is larger than 0.
        steps_matched_device_nodes = [0] * len(steps)

        i_step = 0
        i_runtime = 0
        step_device_min_ts = sys.maxsize
        step_device_max_ts = -sys.maxsize - 1
        matched_device_nodes = set()

        while i_step < len(steps) and i_runtime < len(runtime_node_list):
            step_host_start_time = steps[i_step][0]
            step_host_end_time = steps[i_step][1]
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

        while i_step < len(steps):
            # This step doesn't launch any device side event, just assign it as empty.
            steps_device[i_step] = (step_device_min_ts, step_device_max_ts)
            step_device_min_ts = sys.maxsize
            step_device_max_ts = -sys.maxsize - 1
            i_step += 1

        # If there are matched device, find the first step end time before steps_device[0][0]
        prev_step_end_time = None
        if len(matched_device_nodes) > 0:
            prev_step_end_time = steps[0][0]
            if steps_device[0][0] != sys.maxsize:  # When step 0 has device event.
                for device_node in self.device_node_list:
                    if device_node not in matched_device_nodes:
                        # Now this device_node is not launched inside any step span.
                        if device_node.end_time < steps_device[0][0]:
                            prev_step_end_time = max(prev_step_end_time, device_node.end_time)

        return StepContext(prev_step_end_time, steps_device, steps_matched_device_nodes)

    def _parse_node(self, event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list):
        corrid = event.args.get("correlation", None)
        tid = event.tid
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
            self.used_devices.add(event.pid)
            device_node = DeviceNode.create(event)
            if corrid in corrid_to_runtime:
                rt_node = corrid_to_runtime[corrid]  # Don't pop it because it may be used by next kernel.
                if rt_node.device_nodes is None:
                    rt_node.device_nodes = []
                rt_node.device_nodes.append(device_node)

                # Check the external_id
                if rt_node.external_id != device_node.external_id:
                    logger.warning("Runtime and Device-op have same correlation id but with different external id!")
            else:
                corrid_to_device.setdefault(corrid, []).append(device_node)
            self.device_node_list.append(device_node)
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime.setdefault(rt_node.external_id, []).append(rt_node)
            # Some runtimes has external_id 0, which will not be correlated to any operator.
            # So get them and attach them to root node.
            if rt_node.external_id == 0:
                tid2zero_rt_list.setdefault(tid, []).append(rt_node)
            self.runtime_node_list.append(rt_node)

            # check the external_id
            if device_nodes:
                for device_node in device_nodes:
                    if rt_node.external_id != device_node.external_id:
                        logger.warning("Runtime and Device-op have same correlation id but with different external id!")
        elif event.type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
            if event.type == EventTypes.PROFILER_STEP:
                op_node = ProfilerStepNode.create(event, event.input_shape, event.input_type, None)
            else:
                op_node = OperatorNode.create(event, event.input_shape, event.input_type, event.callstack)
            if event.name in CommunicationOpNameSet:
                self.communication_data[op_node.external_id] = CommunicationNode.create(event, op_node.input_shape, op_node.input_type)
                self.use_nccl = True
            if event.name == "DataParallel.forward":
                self.use_dp = True
            if event.name == "DistributedDataParallel.forward":
                self.use_ddp = True
            tid2list.setdefault(tid, []).append(op_node)


class StepParser:
    def __init__(self):
        # we could not use [[]] * len here since they all point to same memory
        # https://stackoverflow.com/questions/12791501/python-initializing-a-list-of-lists
        # https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly
        self.role_ranges = [[] for _ in range(ProfileRole.Total - 1)]
        self.steps = []
        self.steps_names = []
        self.cpu_min_ts = sys.maxsize  # Min time of CPU side events.
        self.cpu_max_ts = -sys.maxsize - 1  # Max time of CPU side events.
        self.global_min_ts = sys.maxsize  # Min time of all events.
        self.global_max_ts = -sys.maxsize - 1  # Max time of all events.
        # The below two form time range for adding gpu utilization to trace view.
        # Use "PyTorch Profiler (0)" as them.
        # If not exists, assign global_min_ts and global_max_ts to them.
        self.global_start_ts = sys.maxsize
        self.global_end_ts = -sys.maxsize - 1

    def parse_steps(self, events, comm_nodes):
        for event in events:
            self._parse_step(event, comm_nodes)
            if event.type == EventTypes.TRACE and event.name == "PyTorch Profiler (0)":
                self.global_start_ts = event.ts
                self.global_end_ts = event.ts + event.duration
        if self.global_start_ts == sys.maxsize:
            self.global_start_ts = self.global_min_ts
        if self.global_end_ts == -sys.maxsize - 1:
            self.global_end_ts = self.global_max_ts

        if len(self.steps) == 0:
            self.steps.append((self.cpu_min_ts, self.cpu_max_ts))
            self.steps_names.append("0")

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

    def _parse_step(self, event, comm_nodes):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            if event.external_id in comm_nodes:
                self.role_ranges[ProfileRole.Communication].append((ts, ts + dur))
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
            self.cpu_min_ts = min(self.cpu_min_ts, ts)
            self.cpu_max_ts = max(self.cpu_max_ts, ts + dur)
        # Record global wise min and max time.
        self.global_min_ts = min(self.global_min_ts, ts)
        self.global_max_ts = max(self.global_max_ts, ts + dur)

    def update_steps_duration(self, context):
        '''Update self.steps considering device side events launched by each host side step.
        Update self.steps_names if some tail steps are removed.'''

        prev_step_end_time = context.prev_step_end_time
        steps_device = context.steps_device
        steps_matched_device_nodes = context.steps_matched_device_nodes

        # Change step time to device side on the condition that any step have device time.
        is_use_gpu = prev_step_end_time is not None
        if is_use_gpu:
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

class EventParser(NodeParserMixin, StepParser):
    def __init__(self):
        super().__init__()

    def parse(self, events):
        node_context = self.parse_nodes(events)
        self.parse_steps(events, self.communication_data)

        # Move the interleaved logic out of each NodeParser and StepParser
        steps_context = self.find_device_steps(self.steps)
        self.update_steps_duration(steps_context)
        return node_context

    def generate_communication_nodes(self):
        return generate_communication_nodes(self.communication_data, self.steps, self.steps_names)
