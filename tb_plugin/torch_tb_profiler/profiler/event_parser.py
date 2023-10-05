# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

from .. import utils
from .communication import generate_communication_nodes
from .node import (CommunicationNode, DeviceNode, ModuleNode, OperatorNode, PLModuleNode, PLProfileNode,
                   ProfilerStepNode, RuntimeNode, create_operator_node)
from .op_tree import OpTreeBuilder
from .range_utils import merge_ranges
from .trace import BaseEvent, DurationEvent, EventTypes, KernelEvent, NcclOpNameSet, GlooOpNameSet

logger = utils.get_logger()

CommLibTypes = IntEnum('CommLibTypes', ['Nccl', 'Gloo'], start=0)


class ProfileRole(IntEnum):
    Kernel = 0
    Memcpy = 1
    Memset = 2
    Communication = 3
    Runtime = 4
    DataLoader = 5
    CpuOp = 6
    Other = 7
    Total = 8


class NodeParserMixin:
    def __init__(self, *args, **kwargs):
        """Please refer to https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way # noqa: E501
        to see the reason why we need call super().__init__ like this way
        """
        super().__init__(*args, **kwargs)

        self.communication_data: Dict[int, CommunicationNode] = {}
        self.device_node_list: List[DeviceNode] = []
        self.runtime_node_list: List[RuntimeNode] = []
        self.used_devices = set()
        self.use_dp = False
        self.use_ddp = False
        self.comm_lib = set()

    def parse_nodes(self, events: Iterable[BaseEvent]):
        # For OperatorNode and ProfilerStepNode:
        #   Use time interval containing relationship to build father-child correlation,
        #   which is consistent with autograd profiler.
        # For RuntimeNode:
        #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
        #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
        #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
        # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
        tid2list: Dict[int, List[OperatorNode]] = defaultdict(list)
        # value is  a list of PLProfileNode. Do not include RuntimeNode
        pl_tid2list: Dict[int, List[PLProfileNode]] = defaultdict(list)
        # value is a list of RuntimeNode with external_id=0. They will be attached to root nodes.
        tid2zero_rt_list: Dict[int, List[RuntimeNode]] = defaultdict(list)
        corrid_to_device: Dict[int, List[DeviceNode]] = defaultdict(list)  # value is a list of DeviceNode

        corrid_to_runtime: Dict[int, RuntimeNode] = {}  # value is a RuntimeNode
        externalid_to_runtime: Dict[int, List[RuntimeNode]] = defaultdict(list)  # value is a list of RuntimeNode

        for event in events:
            if event.type == EventTypes.MEMORY:
                continue
            self._parse_node(
                event,
                corrid_to_device,
                corrid_to_runtime,
                externalid_to_runtime,
                tid2list,
                pl_tid2list,
                tid2zero_rt_list)

        if CommLibTypes.Nccl in self.comm_lib:
            for event in events:
                if event.type == EventTypes.KERNEL:
                    self._update_communication_node(event)

        # associate CUDA Runtimes with CPU events
        for op_list in tid2list.values():
            for op in op_list:
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.warning("{} Runtime with external id {} don't correlate to any operator!".format(
                    len(externalid_to_runtime[ext_id]), ext_id))

        if len(corrid_to_device) > 0:
            node_count_dict = defaultdict(int)
            for nodes in corrid_to_device.values():
                for n in nodes:
                    node_count_dict[n.type] += 1

            logger.debug(("Some events doesn't belongs to any operators: "
                          f"{', '.join([':'.join((k, str(v))) for k, v in node_count_dict.items()])}"))

        staled_device_nodes = []
        for device_nodes in corrid_to_device.values():
            staled_device_nodes.extend([n for n in device_nodes if n.type == EventTypes.KERNEL])

        return tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list

    def _update_communication_node(self, event: KernelEvent):
        """Update the communication node by using the TraceEvent instance"""
        external_id = event.external_id
        comm_node = self.communication_data.get(external_id)
        if comm_node:
            ts = event.ts
            dur = event.duration
            comm_node.kernel_ranges.append((ts, ts + dur))
            comm_node.total_time += dur

        return comm_node is not None

    def _parse_node(self,
                    event: DurationEvent,
                    corrid_to_device: Dict[int, List[DeviceNode]],
                    corrid_to_runtime: Dict[int, RuntimeNode],
                    externalid_to_runtime: Dict[int, List[RuntimeNode]],
                    tid2list: Dict[int, List[OperatorNode]],
                    pl_tid2list: Dict[int, List[PLProfileNode]],
                    tid2zero_rt_list: Dict[int, List[RuntimeNode]]):
        corrid = event.correlation_id
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
                    logger.warning(
                        'Runtime and Device-op have same correlation id %s but with different external id!'
                        ' (runtime external_id, device external_id): (%s, %s)' %
                        (corrid, rt_node.external_id, device_node.external_id))
            else:
                corrid_to_device[corrid].append(device_node)
            self.device_node_list.append(device_node)
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime[rt_node.external_id].append(rt_node)
            # Some runtimes has external_id 0, which will not be correlated to any operator.
            # So get them and attach them to root node.
            if rt_node.external_id == 0:
                tid2zero_rt_list[tid].append(rt_node)
            self.runtime_node_list.append(rt_node)

            # check the external_id
            if device_nodes:
                for device_node in device_nodes:
                    if rt_node.external_id != device_node.external_id:
                        logger.warning(
                            'Runtime and Device-op have same correlation id %s but with different external id!'
                            ' (rt external_id, device external_id): (%s, %s)' %
                            (corrid, rt_node.external_id, device_node.external_id))
        elif event.type in [EventTypes.PYTHON,
                            EventTypes.OPERATOR,
                            EventTypes.PL_MODULE,
                            EventTypes.PROFILER_STEP,
                            EventTypes.MODULE,
                            EventTypes.USER_ANNOTATION]:
            if event.type == EventTypes.PROFILER_STEP:
                op_node = ProfilerStepNode.create(event)
            elif event.type == EventTypes.MODULE:
                op_node = ModuleNode.create(event)
            elif event.type == EventTypes.PL_MODULE:
                op_node = PLModuleNode.create(event)
            else:
                op_node = create_operator_node(event)
            if event.name in NcclOpNameSet or event.name in GlooOpNameSet:
                comm_node = CommunicationNode.create(event)
                if event.name in NcclOpNameSet:
                    self.comm_lib.add(CommLibTypes.Nccl)
                if event.name in GlooOpNameSet:
                    self.comm_lib.add(CommLibTypes.Gloo)
                ts = event.ts
                dur = event.duration
                comm_node.kernel_ranges.append((ts, ts + dur))
                comm_node.total_time = dur
                self.communication_data[op_node.external_id] = comm_node
            if event.name == 'DataParallel.forward':
                self.use_dp = True
            if event.name == 'DistributedDataParallel.forward':
                self.use_ddp = True
            if op_node:
                tid2list[int(tid)].append(op_node)
        elif event.type == EventTypes.PL_PROFILE:
            op_node = PLProfileNode.create(event)
            pl_tid2list[int(tid)].append(op_node)


class StepParser:
    def __init__(self):
        # we could not use [[]] * len here since they all point to same memory
        # https://stackoverflow.com/questions/12791501/python-initializing-a-list-of-lists
        # https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly
        self.role_ranges: List[List[Tuple[int, int]]] = [[] for _ in range(ProfileRole.Total - 1)]
        self.steps: List[Tuple[int, int]] = []
        self.steps_names: List[str] = []
        self.cpu_min_ts = sys.maxsize  # Min time of CPU side events.
        self.cpu_max_ts = -sys.maxsize - 1  # Max time of CPU side events.
        self.global_min_ts = sys.maxsize  # Min time of all events.
        self.global_max_ts = -sys.maxsize - 1  # Max time of all events.
        # The below two form time range for adding gpu utilization to trace view.
        # Use 'PyTorch Profiler (0)' as them.
        # If not exists, assign global_min_ts and global_max_ts to them.
        self.global_start_ts = sys.maxsize
        self.global_end_ts = -sys.maxsize - 1

    def parse_steps(self, events: Iterable[DurationEvent], comm_nodes: Dict[int, CommunicationNode]):
        for event in events:
            if event.type == EventTypes.MEMORY:
                continue

            self._parse_step(event, comm_nodes)
            if event.type == EventTypes.TRACE and event.name == 'PyTorch Profiler (0)':
                self.global_start_ts = event.ts
                self.global_end_ts = event.ts + event.duration
        if self.global_start_ts == sys.maxsize:
            self.global_start_ts = self.global_min_ts
        if self.global_end_ts == -sys.maxsize - 1:
            self.global_end_ts = self.global_max_ts

        if len(self.steps) == 0:
            self.steps.append((self.cpu_min_ts, self.cpu_max_ts))
            self.steps_names.append('0')

        for i in range(len(self.role_ranges)):
            self.role_ranges[i] = merge_ranges(self.role_ranges[i])

    def update_device_steps(self, runtime_node_list: List[RuntimeNode]):
        self._update_steps_duration(*self._find_device_steps(runtime_node_list))

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

    def _parse_step(self, event: DurationEvent, comm_nodes: Dict[int, CommunicationNode]):
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
        elif ((evt_type == EventTypes.OPERATOR or evt_type == EventTypes.USER_ANNOTATION) and (
                (event.name.startswith('enumerate(DataLoader)#') and event.name.endswith('.__next__'))
                or event.name.startswith('enumerate(DataPipe)#'))):
            self.role_ranges[ProfileRole.DataLoader].append((ts, ts + dur))
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.USER_ANNOTATION]:
            if event.name in GlooOpNameSet or event.name in NcclOpNameSet:
                self.role_ranges[ProfileRole.Communication].append((ts, ts + dur))
            else:
                self.role_ranges[ProfileRole.CpuOp].append((ts, ts + dur))

        # Record host side min and max time.
        if evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
            self.cpu_min_ts = min(self.cpu_min_ts, ts)
            self.cpu_max_ts = max(self.cpu_max_ts, ts + dur)
        # Record global wise min and max time.
        self.global_min_ts = min(self.global_min_ts, ts)
        self.global_max_ts = max(self.global_max_ts, ts + dur)

    def _find_device_steps(self, runtime_node_list: List[RuntimeNode]):
        """return steps associated with device nodes.
        """
        runtime_node_list = sorted(runtime_node_list, key=lambda x: x.start_time)

        # Use similar code with two-way merge to get all runtimes inside each host-side step span,
        # then record each step's min kernel start time and max kernel end time:
        steps_device: List[Tuple[int, int]] = [(sys.maxsize, -sys.maxsize - 1)] * len(self.steps)
        # where the steps associated with devcie node, if yes, the related array item is larger than 0.
        steps_matched_device_nodes: List[int] = [0] * len(self.steps)

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

        # If there are matched device, find the first step end time before steps_device[0][0]
        prev_step_end_time: Optional[int] = None
        if len(matched_device_nodes) > 0:
            prev_step_end_time = self.steps[0][0]
            if steps_device[0][0] != sys.maxsize:  # When step 0 has device event.
                for device_node in self.device_node_list:
                    if device_node not in matched_device_nodes:
                        # Now this device_node is not launched inside any step span.
                        if device_node.end_time < steps_device[0][0]:
                            prev_step_end_time = max(prev_step_end_time, device_node.end_time)

        return prev_step_end_time, steps_device, steps_matched_device_nodes

    def _update_steps_duration(self,
                               prev_step_end_time: Optional[int],
                               steps_device: List[Tuple[int, int]],
                               steps_matched_device_nodes: List[int]):
        """Update self.steps considering device side events launched by each host side step.
        Update self.steps_names if some tail steps are removed."""

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
                            'Abnormal step_end_time of step {}: [{}, {}]'.format(
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
                        'Remove the last {} steps from overview. '
                        'Because the profiler may fail to capture all the kernels launched by these steps.'.format(
                            len(self.steps) - keep_steps
                        ))
                    self.steps = self.steps[:keep_steps]
                    self.steps_names = self.steps_names[:keep_steps]


class EventParser(NodeParserMixin, StepParser):
    def __init__(self):
        super().__init__()
        self.comm_node_list: Dict[CommunicationNode] = None

    def parse(self, events: Iterable[BaseEvent], fwd_bwd_map: Dict[int, int]) -> Dict[int, List[OperatorNode]]:
        with utils.timing('EventParser: parse nodes'):
            tid2list, tid2zero_rt_list, staled_device_nodes, pl_tid2list = self.parse_nodes(events)

        with utils.timing('EventParser: build operator tree'):
            builder = OpTreeBuilder()
            tid2tree = builder.build_tree(tid2list, tid2zero_rt_list, staled_device_nodes, fwd_bwd_map=fwd_bwd_map)
            pl_tid2tree = builder.build_tree(pl_tid2list, {}, [], {})

        with utils.timing('EventParser: parse steps times'):
            # Process steps
            self.parse_steps(events, self.communication_data)
            if len(self.comm_lib) > 1:
                logger.warning(
                    'Multiple communication libs are found. To avoid confusing, we disable the distributed view.')
                self.communication_data.clear()

        # Move the interleaved logic out of each NodeParser and StepParser
        self.update_device_steps(self.runtime_node_list)

        self.comm_node_list = generate_communication_nodes(self.communication_data, self.steps, self.steps_names)
        return tid2tree, pl_tid2tree

    @staticmethod
    def print_tree(root):
        class Ctx:
            tid: int = -1
            name_stack: list = []

        ctx = Ctx()

        def print_node_set_prefix(node: OperatorNode):
            header = f'[{ctx.tid}]' + '.'.join(ctx.name_stack[1:])  # omit the CallTreeRoot
            prefix_len = len(ctx.name_stack) * 4 - 4 - 1
            if len(ctx.name_stack) > 1:
                print(header)
                prefix = ' ' * prefix_len
                print(prefix, node.name)
                print(prefix, 'time:', node.start_time, '-->', node.end_time)

        def push(node: OperatorNode):
            ctx.name_stack.append(node.name)

        def pop():
            ctx.name_stack.pop()

        def traverse_opeartor_node(node: OperatorNode):
            print_node_set_prefix(node)

            push(node)
            for n in node.children:
                traverse_opeartor_node(n)
            pop()

        ctx.tid = root.tid
        traverse_opeartor_node(root)
        ctx.tid = -1
