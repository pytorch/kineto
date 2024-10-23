# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from abc import ABC
from typing import List, Optional, Tuple

from .. import utils
from .tensor_core import TC_Allowlist, TC_OP_Allowlist
from .trace import (DurationEvent, EventTypes, KernelEvent, ModuleEvent,
                    OperatorEvent, PLProfileEvent, NcclOpNameSet, GlooOpNameSet)

logger = utils.get_logger()

ExcludeOpName = ['DataParallel.forward', 'DistributedDataParallel.forward']


class BaseNode(ABC):
    def __init__(self, name: str, start_time: int, end_time: int, type: str, tid: int,
                 external_id: Optional[int] = None):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.tid = tid
        self.external_id = external_id  # For consistency check.

    @staticmethod
    def get_node_argument(event: DurationEvent):
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

    @property
    def duration(self) -> int:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return 0


class CommunicationNode(BaseNode):
    def __init__(self, input_shape: List[List[int]], input_type: List[str], **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.input_type = input_type
        self.kernel_ranges: List[Tuple[int, int]] = []
        self.real_time_ranges: List[Tuple[int, int]] = []
        self.total_time: int = 0
        self.real_time: int = 0
        self.step_name: str = None

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=event.input_shape, input_type=event.input_type, **kwargs)


class HostNode(BaseNode):
    def __init__(self, device_duration: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream? # noqa: E501


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    def __init__(self, children=None, runtimes=None, input_shape: Optional[List[List[int]]] = None,
                 input_type: Optional[List[str]] = None, callstack: Optional[str] = None,
                 self_host_duration: int = 0, self_device_duration: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.children: List[OperatorNode] = [] if children is None else children  # OperatorNode and ProfilerStepNode.
        self.runtimes: List[RuntimeNode] = [] if runtimes is None else runtimes  # RuntimeNode
        self.input_shape = input_shape
        self.input_type = input_type
        self.callstack = callstack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration
        # self.parent_node = None
        self.tc_eligible = self.name in TC_OP_Allowlist
        self.tc_self_duration = 0  # Time of TC kernels launched by this op excluding its children operators.
        self.tc_total_duration = 0  # Time of TC kernels launched by this op including its children operators.

    def fill_stats(self):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.runtimes.sort(key=lambda x: (x.start_time, -x.end_time)
                           if x.start_time and x.end_time else (sys.maxsize, -sys.maxsize - 1))

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
            if rt.end_time is not None and rt.start_time is not None:
                self.self_host_duration -= (rt.end_time - rt.start_time)
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration
            self.tc_self_duration += rt.tc_duration
            self.tc_total_duration += rt.tc_duration
            if self.type == EventTypes.OPERATOR and not self.tc_eligible and rt.tc_duration > 0:
                logger.warning("New Tensor Cores eligible operator found: '{}'!".format(self.name))
                self.tc_eligible = True

    def get_operator_and_kernels(self):
        ops: List[OperatorNode] = []
        kernels: List[DeviceNode] = []
        for child in self.children:
            child_ops, child_kernels = child.get_operator_and_kernels()
            ops.extend(child_ops)
            kernels.extend(child_kernels)
        for rt in self.runtimes:
            kernels.extend(list(rt.get_kernels()))

        if is_operator_node(self):
            ops.append(self)

        return ops, kernels

    @classmethod
    def create(cls, event: OperatorEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=event.input_shape, input_type=event.input_type, callstack=event.callstack, **kwargs)


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ModuleNode(OperatorNode):
    def __init__(self, module_id: int, python_id: int, python_parent_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id
        self.python_id = python_id
        self.python_parent_id = python_parent_id

    def fill_stats(self):
        super().fill_stats()
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: ModuleEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs['module_id'] = event.module_id
        kwargs['python_id'] = event.python_id
        kwargs['python_parent_id'] = event.python_parent_id
        # From the time being, the ModuleNode always have external_id to 0.
        # As the result, we need reset the external_id to None to ignore adding the runtime nodes for ModuleNode
        kwargs.pop('external_id', None)
        return cls(**kwargs)


class BackwardNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fill_stats(self):
        """Override the timestamps and duration for BackwardNode only
        """
        self.children.sort(key=lambda x: (x.start_time, -x.end_time))
        self.start_time = self.children[0].start_time
        self.end_time = self.children[-1].end_time

        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= (child.end_time - child.start_time)
            self.tc_total_duration += child.tc_total_duration
            # Mark TC eligible as True if any child operator is TC eligible.
            if not self.tc_eligible and child.tc_eligible:
                self.tc_eligible = True


class PLProfileNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        return cls(**kwargs)


class PLModuleNode(OperatorNode):
    def __init__(self, module_id: int, **kwargs):
        super().__init__(**kwargs)
        self.module_id = module_id

    def fill_stats(self):
        super().fill_stats()
        self.self_device_duration += get_chilren_self_device_time(self)

    @classmethod
    def create(cls, event: PLProfileEvent):
        kwargs = BaseNode.get_node_argument(event)
        kwargs['module_id'] = event.module_id
        return cls(**kwargs)


class DataLoaderNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OptimizerNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(self, device_nodes: Optional[List['DeviceNode']] = None, **kwargs):
        super().__init__(**kwargs)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = sorted(device_nodes, key=lambda x: (x.start_time, -x.end_time)) if device_nodes else None
        self.tc_duration: int = 0  # Time summarization of all its launched kernels.

    def fill_stats(self, op_node: OperatorNode = None):
        if self.device_nodes:
            for device_node in self.device_nodes:
                if op_node:
                    device_node.op_name = op_node.name
                    device_node.op_tc_eligible = op_node.tc_eligible
                device_duration = device_node.end_time - device_node.start_time
                self.device_duration += device_duration
                self.tc_duration += device_duration if device_node.tc_used else 0

    def get_kernels(self):
        if self.device_nodes:
            for d in self.device_nodes:
                if d.type == EventTypes.KERNEL:
                    yield d

    @classmethod
    def create(cls, event, device_nodes: Optional[List['DeviceNode']]):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class DeviceNode(BaseNode):
    def __init__(self,
                 blocks_per_sm: Optional[float] = None,
                 occupancy: int = None,
                 grid: Optional[List[int]] = None,
                 block: Optional[List[int]] = None,
                 regs_per_thread: int = None,
                 shared_memory: int = None,
                 device_id: int = None, **kwargs):
        super().__init__(**kwargs)
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
    def create(cls, event: KernelEvent):
        kwargs = BaseNode.get_node_argument(event)
        if event.type == EventTypes.KERNEL:
            kwargs['blocks_per_sm'] = event.blocks_per_sm
            kwargs['occupancy'] = event.occupancy
            kwargs['grid'] = event.grid
            kwargs['block'] = event.block
            kwargs['regs_per_thread'] = event.regs_per_thread
            kwargs['shared_memory'] = event.shared_memory
            kwargs['device_id'] = event.device_id
        return cls(**kwargs)


def create_operator_node(event: OperatorEvent):
    if (event.name.startswith('enumerate(DataLoader)#') and event.name.endswith('.__next__')
            or event.name.startswith('enumerate(DataPipe)#')):
        return DataLoaderNode.create(event)
    elif event.name.startswith('Optimizer.step'):
        return OptimizerNode.create(event)
    elif event.type == EventTypes.USER_ANNOTATION:
        if event.name in GlooOpNameSet or event.name in NcclOpNameSet:
            return OperatorNode.create(event)
        else:
            return None
    else:
        return OperatorNode.create(event)


def is_operator_node(node: BaseNode):
    return bool(type(node) is OperatorNode and node.type == EventTypes.OPERATOR and node.name not in ExcludeOpName
                and not node.name.startswith("Optimizer."))  # exclude Optimizer.zero_grad


def get_chilren_self_device_time(node):
    self_device_duration = 0
    for child in node.children:
        if is_operator_node(child):
            self_device_duration += child.device_duration
    return self_device_duration
