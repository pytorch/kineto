# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from abc import ABC
from collections import defaultdict
from enum import IntEnum

from .. import utils
from .tensor_core import TC_Allowlist, TC_OP_Allowlist
from .trace import EventTypes

logger = utils.get_logger()

ExcludeOpName = ["DataParallel.forward", "DistributedDataParallel.forward"]

class MemoryMetrics(IntEnum):
    SelfIncreaseSize = 0
    SelfAllocationSize = 1
    SelfAllocationCount = 2
    IncreaseSize = 3
    AllocationSize = 4
    AllocationCount = 5

class BaseNode(ABC):
    def __init__(self, name, start_time, end_time, type, tid, external_id):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.tid = tid
        self.external_id = external_id  # For consistency check.

    @staticmethod
    def get_node_argument(event):
        kwargs = {}
        kwargs['name'] = event.name
        kwargs['start_time'] = event.ts
        kwargs['end_time'] = event.ts + event.duration
        kwargs['type'] = event.type
        kwargs['tid'] = event.tid

        external_id = getattr(event, 'external_id', None)
        if external_id is not None:
            kwargs['external_id'] = external_id

        return kwargs


class CommunicationNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, tid, external_id, input_shape, input_type):
        super().__init__(name, start_time, end_time, type, tid, external_id)
        self.input_shape = input_shape
        self.input_type = input_type
        self.kernel_ranges = []
        self.real_time_ranges = []
        self.total_time = 0
        self.real_time = 0
        self.step_name = None

    @classmethod
    def create(cls, event):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=event.input_shape, input_type=event.input_type, **kwargs)


class HostNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, tid, external_id, device_duration=0):
        super().__init__(name, start_time, end_time, type, tid, external_id)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream?


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    def __init__(self, name, start_time, end_time, type, tid, external_id=None, device_duration=0,
            children=None, runtimes=None, input_shape=None, input_type=None, callstack=None, self_host_duration=0, self_device_duration=0):
        super().__init__(name, start_time, end_time, type, tid,  external_id, device_duration)
        self.children = [] if children is None else children # OperatorNode and ProfilerStepNode.
        self.runtimes = [] if runtimes is None else runtimes # RuntimeNode
        self.input_shape = input_shape
        self.input_type = input_type
        self.callstack = callstack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration
        self.memory_records = []
        # self.parent_node = None
        self.tc_eligible = self.name in TC_OP_Allowlist
        self.tc_self_duration = 0  # Time of TC kernels launched by this op excluding its children operators.
        self.tc_total_duration = 0  # Time of TC kernels launched by this op including its children operators.

    def add_memory_record(self, record):
        self.memory_records.append(record)

    def get_memory_metrics(self, start_ts, end_ts):
        metrics_count = len([e.name for e in MemoryMetrics if e.name.startswith("Self")])
        memory_metrics = defaultdict(lambda: [0] * metrics_count)
        for record in self.memory_records:
            if start_ts is not None and record.ts < start_ts:
                continue
            if end_ts is not None and record.ts > end_ts:
                continue
            name = record.device_name
            if name is None:
                continue

            memory_metrics[name][MemoryMetrics.SelfIncreaseSize] += record.bytes
            if record.bytes > 0:
                memory_metrics[name][MemoryMetrics.SelfAllocationSize] += record.bytes
                memory_metrics[name][MemoryMetrics.SelfAllocationCount] += 1

        return memory_metrics

    def fill_stats(self):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        for child in self.children:
            child.fill_stats()
        for rt in self.runtimes:
            rt.fill_stats(self)

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= (child.end_time - child.start_time)
            self.tc_total_duration += child.tc_total_duration
            # Mark TC eligible as True if any child operator is TC eligible.
            if self.type == EventTypes.OPERATOR and not self.tc_eligible and child.tc_eligible:
                self.tc_eligible = True
        for rt in self.runtimes:
            # From PyTorch 1.8 RC1, cpu_self_time does not include runtime's time.
            # So here we keep consistent with it.
            self.self_host_duration -= (rt.end_time - rt.start_time)
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration
            self.tc_self_duration += rt.tc_duration
            self.tc_total_duration += rt.tc_duration
            if self.type == EventTypes.OPERATOR and not self.tc_eligible and rt.tc_duration > 0:
                logger.warning("New Tensor Cores eligible operator found: '{}'!".format(self.name))
                self.tc_eligible = True

    def replace_time_by_children(self):
            self.start_time = next((child.start_time for child in self.children if child.start_time is not None), None)
            self.end_time = next((child.end_time for child in reversed(self.children) if child.end_time is not None), None)

    def get_operator_and_kernels(self):
        ops = []
        kernels = []
        for child in self.children:
            child_ops, child_kernels = child.get_operator_and_kernels()
            ops.extend(child_ops)
            kernels.extend(child_kernels)
        for rt in self.runtimes:
            kernels.extend(rt.get_kernels())

        if is_operator_node(self):
            ops.append(self)

        return ops, kernels

    @classmethod
    def create(cls, event):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=event.input_shape, input_type=event.input_type, callstack=event.callstack, **kwargs)


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(self, name, start_time, end_time, type, tid, external_id=None, device_duration=0,
            device_nodes=None):
        super().__init__(name, start_time, end_time, type, tid, external_id, device_duration)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = device_nodes
        self.tc_duration = 0  # Time summarization of all its launched kernels.

    def fill_stats(self, op_node=None):
        if self.device_nodes:
            for device_node in self.device_nodes:
                if op_node:
                    device_node.op_name = op_node.name
                    device_node.op_tc_eligible = op_node.tc_eligible
                device_duration = device_node.end_time - device_node.start_time
                self.device_duration += device_duration
                self.tc_duration += device_duration if device_node.tc_used else 0

    def get_kernels(self):
        return [n for n in self.device_nodes if n.type == EventTypes.KERNEL] if self.device_nodes else []

    @classmethod
    def create(cls, event, device_nodes):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class DeviceNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, tid, external_id=None,
                 blocks_per_sm=None, occupancy=None,
                 grid=None, block=None, regs_per_thread=None, shared_memory=None, tc_used=False, device_id=None):
        super().__init__(name, start_time, end_time, type, tid, external_id)
        self.op_tc_eligible = False
        self.op_name = None
        self.blocks_per_sm = blocks_per_sm
        self.occupancy = occupancy
        self.grid = grid
        self.block = block
        self.regs_per_thread = regs_per_thread
        self.shared_memory = shared_memory
        self.tc_used = self.name in TC_Allowlist
        self.device_id = device_id

    @classmethod
    def create(cls, event):
        kwargs = BaseNode.get_node_argument(event)
        if event.type == EventTypes.KERNEL:
            kwargs["blocks_per_sm"] = event.blocks_per_sm
            kwargs["occupancy"] = event.occupancy
            kwargs["grid"] = event.grid
            kwargs["block"] = event.block
            kwargs["regs_per_thread"] = event.regs_per_thread
            kwargs["shared_memory"] = event.shared_memory
            kwargs["device_id"] = event.device_id
        return cls(**kwargs)

def is_operator_node(node):
    if type(node) is OperatorNode and node.type == EventTypes.OPERATOR \
        and not (node.name.startswith("enumerate(DataLoader)#") and node.name.endswith(".__next__")) \
        and not node.name.startswith("Optimizer.") and node.name not in ExcludeOpName:
        return True
    else:
        return False
