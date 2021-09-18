# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from enum import IntEnum

from .. import utils

__all__ = ["EventTypes", "create_event"]

logger = utils.get_logger()

class DeviceType(IntEnum):
    CPU = 0
    CUDA = 1

class EventTypes(object):
    TRACE = "Trace"
    OPERATOR = "Operator"
    PROFILER_STEP = "ProfilerStep"
    RUNTIME = "Runtime"
    KERNEL = "Kernel"
    MEMCPY = "Memcpy"
    MEMSET = "Memset"
    PYTHON = "Python"
    MEMORY = "Memory"

EventTypeMap = {
    "Trace" : EventTypes.TRACE,
    "cpu_op" : EventTypes.OPERATOR,
    "Operator" : EventTypes.OPERATOR,
    "Runtime" : EventTypes.RUNTIME,
    "Kernel" : EventTypes.KERNEL,
    "Memcpy" : EventTypes.MEMCPY,
    "gpu_memcpy" : EventTypes.MEMCPY,
    "Memset" : EventTypes.MEMSET,
    "gpu_memset" : EventTypes.MEMSET,
    "Python" : EventTypes.PYTHON,
    "Memory" : EventTypes.MEMORY
}

class BaseEvent(object):
    def __init__(self, type, data):
        self.type = type
        self.name = data.get("name")
        self.ts = data.get("ts")
        self.pid = data.get("pid")
        self.tid = data.get("tid")
        self.args = data.get("args", {})

class DurationEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.category = data.get("cat", "")
        self.duration = data.get("dur")

        extern_id = self.args.get("external id")
        if extern_id is None:
            extern_id = self.args.get("External id")
        self.external_id = extern_id
        self.correlation_id = self.args.get("correlation")

class KernelEvent(DurationEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.occupancy = self.args.get("est. achieved occupancy %")
        self.blocks_per_sm = self.args.get("blocks per SM")
        self.grid = self.args.get("grid")
        self.block = self.args.get("block")
        self.regs_per_thread = self.args.get("registers per thread")
        self.shared_memory = self.args.get("shared memory")
        self.device_id = self.args.get("device")


class OperatorEvent(DurationEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.callstack = self.args.get("Call stack", "")
        self.input_type = self.args.get("Input type")

        shape = self.args.get("Input Dims")
        if shape is None:
            # Setting shape to '[]' other None is to align with autograd result
            shape = self.args.get("Input dims", [])
        self.input_shape = shape

class ProfilerStepEvent(OperatorEvent):
    def __init__(self, data):
        super().__init__(EventTypes.PROFILER_STEP, data)
        # torch.profiler.profile.step will invoke record_function with name like "ProfilerStep#5"
        self.step = int(self.name.split("#")[1])

class MemoryEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.scope = data.get("s", "")
        self.device_id = self.args.get("Device Id")
        dtype = self.args.get("Device Type")
        if dtype is not None:
            try:
                dtype = DeviceType(dtype)
            except ValueError:
                dtype = None

        self.device_type = dtype

    @property
    def addr(self):
        return self.args.get("Addr")

    @property
    def bytes(self):
        return self.args.get("Bytes", 0)

    @property
    def total_allocated(self):
        return self.args.get("Total Allocated", float("nan"))

    @property
    def total_reserved(self):
        return self.args.get("Total Reserved", float("nan"))

def create_event(event):
    try:
        type = event.get("ph")
        if type == "X":
            return create_trace_event(event)
        elif type == "i" and event.get("name") == "[memory]":
            return MemoryEvent(EventTypes.MEMORY, event)
        else:
            return None
    except Exception as ex:
        logger.warning("Failed to parse profile event. Exception=%s. Event=%s", ex, event, exc_info=True)
        raise

def create_trace_event(event):
    category = event.get("cat")
    event_type = EventTypeMap.get(category)
    if event_type == EventTypes.OPERATOR:
        name = event.get("name")
        if name and name.startswith("ProfilerStep#"):
            return ProfilerStepEvent(event)
        else:
            return OperatorEvent(event_type, event)
    elif event_type == EventTypes.PYTHON:
        return OperatorEvent(event_type, event)
    elif event_type == EventTypes.KERNEL:
        return KernelEvent(event_type, event)
    elif event_type is not None:
        return DurationEvent(event_type, event)
    else:
        return None
