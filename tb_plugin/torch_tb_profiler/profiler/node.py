# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from abc import ABC
from enum import IntEnum

MemoryMetrics = IntEnum('MemoryMetrics', ['SelfIncreaseSize', 'SelfAllocationSize', 'SelfAllocationCount', 'IncreaseSize', 'AllocationSize', 'AllocationCount', 'Total'], start=0)

from .trace import EventTypes


class BaseNode(ABC):
    def __init__(self, name, start_time, end_time, type, external_id):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.external_id = external_id  # For consistency check.

    @staticmethod
    def get_node_argument(event):
        kwargs = {}
        kwargs['name'] = event.name
        kwargs['start_time'] = event.ts
        kwargs['end_time'] = event.ts + event.duration
        kwargs['type'] = event.type

        if event.external_id is not None:
            kwargs['external_id'] = event.external_id

        return kwargs


class CommunicationNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, external_id, input_shape, input_type):
        super().__init__(name, start_time, end_time, type, external_id)
        self.input_shape = input_shape
        self.input_type = input_type
        self.kernel_ranges = []
        self.total_time = 0
        self.real_time = 0
        self.step_name = None

    @classmethod
    def create(cls, event, input_shape, input_type):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=input_shape, input_type=input_type, **kwargs)


class HostNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, external_id, device_duration=0):
        super().__init__(name, start_time, end_time, type, external_id)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream?


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    def __init__(self, name, start_time, end_time, type, external_id=None, device_duration=0,
            children=None, runtimes=None, input_shape=None, input_type=None, call_stack=None, self_host_duration=0, self_device_duration=0):
        super().__init__(name, start_time, end_time, type, external_id, device_duration)
        self.children = [] if children is None else children # OperatorNode and ProfilerStepNode.
        self.runtimes = [] if runtimes is None else runtimes # RuntimeNode
        self.input_shape = input_shape
        self.input_type = input_type
        self.call_stack = call_stack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration
        self.memory_records = []

    def add_memory_record(self, record):
        self.memory_records.append(record)

    def get_memroy_metrics(self, device_type, device_id):
        memory_metrics = [
            self.get_memory_increase(device_type, device_id),
            self.get_memory_allocation_size(device_type, device_id),
            self.get_memory_allocation_count(device_type, device_id)
        ]
        return memory_metrics

    def get_memory_allocation_count(self, device_type, device_id):
        count = 0
        for record in self.memory_records:
            if record.device_type == device_type and record.device_id == device_id and record.bytes > 0:
                count += 1
        return count

    def get_memory_allocation_size(self, device_type, device_id):
        return self.get_memory_stats(device_type, device_id, False)

    def get_memory_increase(self, device_type, device_id):
        return self.get_memory_stats(device_type, device_id, True)

    def get_memory_stats(self, device_type, device_id, include_negative):
        size = 0
        for record in self.memory_records:
            if record.device_type == device_type and record.device_id == device_id:
                if record.bytes > 0 or include_negative:
                    size += record.bytes
        return size

    def fill_stats(self):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        for child in self.children:
            child.fill_stats()
        for rt in self.runtimes:
            rt.fill_stats()
            rt.update_device_op_node(self)

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= (child.end_time - child.start_time)
        for rt in self.runtimes:
            # From PyTorch 1.8 RC1, cpu_self_time does not include runtime's time.
            # So here we keep consistent with it.
            self.self_host_duration -= (rt.end_time - rt.start_time)
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration

    def replace_time_by_children(self):
            self.start_time = next((child.start_time for child in self.children if child.start_time is not None), None)
            self.end_time = next((child.end_time for child in reversed(self.children) if child.end_time is not None), None)

    @classmethod
    def create(cls, event, input_shape, input_type, call_stack):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=input_shape, input_type=input_type, call_stack=call_stack, **kwargs)


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(self, name, start_time, end_time, type, external_id=None, device_duration=0,
            device_nodes=None):
        super().__init__(name, start_time, end_time, type, external_id, device_duration)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = device_nodes

    def fill_stats(self):
        if self.device_nodes:
            for device_node in self.device_nodes:
                self.device_duration += device_node.end_time - device_node.start_time

    def update_device_op_node(self, node):
        if self.device_nodes:
            for device_node in self.device_nodes:
                device_node.op_node = node

    @classmethod
    def create(cls, event, device_nodes):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class DeviceNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, external_id=None,
            op_node=None, blocks_per_sm=None, occupancy=None):
        super().__init__(name, start_time, end_time, type, external_id)
        self.op_node = op_node  # The cpu operator that launched it.
        self.blocks_per_sm = blocks_per_sm
        self.occupancy = occupancy

    @classmethod
    def create(cls, event):
        kwargs = DeviceNode.get_node_argument(event)
        return cls(**kwargs)

    @staticmethod
    def get_node_argument(event):
        kwargs = BaseNode.get_node_argument(event)
        if event.type == EventTypes.KERNEL:
            kwargs["blocks_per_sm"] = event.args.get("blocks per SM", 0)
            kwargs["occupancy"] = event.args.get("est. achieved occupancy %", 0)
        return kwargs

def is_operator_node(node):
    if type(node) is OperatorNode and node.type == EventTypes.OPERATOR \
        and not (node.name.startswith("enumerate(DataLoader)#") and node.name.endswith(".__next__")) \
        and not node.name.startswith("Optimizer."):
        return True
    else:
        return False
