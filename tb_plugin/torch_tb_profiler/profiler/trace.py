# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from enum import IntEnum
from typing import Dict, Optional

from .. import utils

__all__ = ['EventTypes', 'create_event']

logger = utils.get_logger()

NcclOpNameSet = ['nccl:broadcast', 'nccl:reduce', 'nccl:all_reduce', 'nccl:all_gather', 'nccl:reduce_scatter']
GlooOpNameSet = ['gloo:broadcast', 'gloo:reduce', 'gloo:all_reduce', 'gloo:all_gather', 'gloo:reduce_scatter']

class DeviceType(IntEnum):
    CPU = 0
    CUDA = 1


class EventTypes:
    TRACE = 'Trace'
    OPERATOR = 'Operator'
    PROFILER_STEP = 'ProfilerStep'
    RUNTIME = 'Runtime'
    KERNEL = 'Kernel'
    MEMCPY = 'Memcpy'
    MEMSET = 'Memset'
    PYTHON = 'Python'
    MEMORY = 'Memory'
    PYTHON_FUNCTION = 'python_function'
    MODULE = 'Module'
    PL_PROFILE = 'pl_profile'
    PL_MODULE = 'pl_module'
    USER_ANNOTATION = 'user_annotation'


EventTypeMap = {
    'trace': EventTypes.TRACE,
    'cpu_op': EventTypes.OPERATOR,
    'operator': EventTypes.OPERATOR,
    'runtime': EventTypes.RUNTIME,
    'kernel': EventTypes.KERNEL,
    'memcpy': EventTypes.MEMCPY,
    'gpu_memcpy': EventTypes.MEMCPY,
    'memset': EventTypes.MEMSET,
    'gpu_memset': EventTypes.MEMSET,
    'python': EventTypes.PYTHON,
    'memory': EventTypes.MEMORY,
    'python_function': EventTypes.PYTHON_FUNCTION,
    'user_annotation': EventTypes.USER_ANNOTATION,
    'gpu_user_annotation': EventTypes.USER_ANNOTATION
}


class BaseEvent:
    def __init__(self, type, data):
        self.type: str = type
        self.name: str = data.get('name')
        self.ts: int = data.get('ts')
        self.pid: int = data.get('pid')
        self.tid: int = data.get('tid')
        self.args: Dict = data.get('args', {})


class DurationEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.category: str = data.get('cat', '')
        self.duration: int = data.get('dur')

        extern_id: Optional[int] = self.args.get('external id')
        if extern_id is None:
            extern_id = self.args.get('External id')
        self.external_id = extern_id
        self.correlation_id: Optional[int] = self.args.get('correlation')


class KernelEvent(DurationEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.occupancy = self.args.get('est. achieved occupancy %')
        self.blocks_per_sm = self.args.get('blocks per SM')
        self.grid = self.args.get('grid')
        self.block = self.args.get('block')
        self.regs_per_thread = self.args.get('registers per thread')
        self.shared_memory = self.args.get('shared memory')
        self.device_id = self.args.get('device')


class OperatorEvent(DurationEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.callstack = self.args.get('Call stack')
        self.input_type = self.args.get('Input type')

        shape = self.args.get('Input Dims')
        if shape is None:
            # Setting shape to '[]' other None is to align with autograd result
            shape = self.args.get('Input dims', [])
        self.input_shape = shape


class ProfilerStepEvent(OperatorEvent):
    def __init__(self, data):
        super().__init__(EventTypes.PROFILER_STEP, data)
        # torch.profiler.profile.step will invoke record_function with name like 'ProfilerStep#5'
        self.step: int = int(self.name.split('#')[1])


class MemoryEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.scope: str = data.get('s', '')
        self.device_id: int = self.args.get('Device Id')
        dtype = self.args.get('Device Type')
        if dtype is not None:
            try:
                dtype = DeviceType(dtype)
            except ValueError:
                dtype = None

        self.device_type: DeviceType = dtype

    @property
    def addr(self):
        return self.args.get('Addr')

    @property
    def bytes(self):
        return self.args.get('Bytes', 0)

    @property
    def total_allocated(self):
        return self.args.get('Total Allocated', float('nan'))

    @property
    def total_reserved(self):
        return self.args.get('Total Reserved', float('nan'))


class PythonFunctionEvent(DurationEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.python_id: int = self.args.get('Python id')
        self.python_parent_id: int = self.args.get('Python parent id')


class ModuleEvent(PythonFunctionEvent):
    def __init__(self, data):
        super().__init__(EventTypes.MODULE, data)
        self.module_id: int = self.args.get('Python module id')


class PLProfileEvent(DurationEvent):
    def __init__(self, data):
        super().__init__(EventTypes.PL_PROFILE, data)
        self.name = self.name.replace('[pl][profile]', '')


class PLModuleEvent(DurationEvent):
    def __init__(self, data):
        super().__init__(EventTypes.PL_MODULE, data)
        self.module_id = 0  # just to be compatible with ModuleEvent processing
        self.name = self.name.replace('[pl][module]', '')
        # self.shape = self.name[:self.name.rfind(']')+1]
        # self.name = self.name[self.name.rfind(']')+1:]
        self.module_type = self.name[:self.name.find(': ')]
        self.name = self.name[self.name.find(': ')+2:]


def create_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
    try:
        type = event.get('ph')
        if type == 'X':
            return create_trace_event(event, is_pytorch_lightning)
        elif type == 'i' and event.get('name') == '[memory]':
            return MemoryEvent(EventTypes.MEMORY, event)
        else:
            return None
    except Exception as ex:
        logger.warning('Failed to parse profile event. Exception=%s. Event=%s', ex, event, exc_info=True)
        raise


def create_trace_event(event, is_pytorch_lightning) -> Optional[BaseEvent]:
    category = event.get('cat')
    event_type = EventTypeMap.get(category.lower())
    if event_type == EventTypes.USER_ANNOTATION:
        name = event.get('name')
        if name and name.startswith('ProfilerStep#'):
            return ProfilerStepEvent(event)
        return OperatorEvent(event_type, event)
    elif event_type == EventTypes.OPERATOR:
        name = event.get('name')
        if name and name.startswith('ProfilerStep#'):
            return ProfilerStepEvent(event)
        if is_pytorch_lightning:
            if name and name.startswith('[pl][profile]'):
                return PLProfileEvent(event)
            elif name and name.startswith('[pl][module]'):
                return PLModuleEvent(event)
        return OperatorEvent(event_type, event)
    elif event_type == EventTypes.PYTHON:
        return OperatorEvent(event_type, event)
    elif event_type == EventTypes.KERNEL:
        return KernelEvent(event_type, event)
    elif event_type == EventTypes.PYTHON_FUNCTION:
        if is_pytorch_lightning:
            return None
        args = event.get('args')
        if args and args.get('Python module id') is not None:
            return ModuleEvent(event)
        else:
            return PythonFunctionEvent(event_type, event)
    elif event_type is not None:
        return DurationEvent(event_type, event)
    return None


def create_association_events(events) -> Dict[int, int]:
    forward_map = {}
    backward_map = {}

    result = {}
    for e in events:
        ph = e.get('ph')
        id = e['id']
        ts = e['ts']
        if ph == 's':
            forward_map[id] = ts
        elif ph == 'f':
            backward_map[id] = ts

    for id, ts in forward_map.items():
        backward_ts = backward_map.get(id)
        if backward_ts is not None:
            result[ts] = backward_ts

    return result
