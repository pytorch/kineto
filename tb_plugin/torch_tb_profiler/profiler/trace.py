# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from enum import IntEnum

from .. import utils

__all__ = ["EventTypes", "create_event"]

logger = utils.get_logger()

DeviceType = IntEnum('DeviceType', ['CPU', 'CUDA'], start=0)

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

Supported_EventTypes = [v for k, v in vars(EventTypes).items() if not k.startswith("_") and v != EventTypes.PROFILER_STEP]

class BaseEvent(object):
    def __init__(self, type, data):
        self.type = type
        self.name = data.get("name")
        self.ts = data.get("ts")
        self.pid = data.get("pid")
        self.tid = data.get("tid")
        self.args = data.get("args", {})

class TraceEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.category = data.get("cat", "")
        self.duration = data.get("dur")

    @property
    def external_id(self):
        extern_id = self.args.get("external id")
        if extern_id is None:
            extern_id = self.args.get("External id")

        return extern_id

    @property
    def callstack(self):
        return self.args.get("Call stack", "")

    @property
    def input_shape(self):
        shape = self.args.get("Input Dims")
        if shape is None:
            shape = self.args.get("Input dims")

        return shape

    @property
    def input_type(self):
        return self.args.get("Input type")

class ProfilerStepEvent(TraceEvent):
    def __init__(self, data):
        super().__init__(EventTypes.PROFILER_STEP, data)
        # torch.profiler.profile.step will invoke record_function with name like "ProfilerStep#5"
        self.step = int(self.name.split("#")[1])

class MemoryEvent(BaseEvent):
    def __init__(self, type, data):
        super().__init__(type, data)
        self.scope = data.get("s", "")

    @property
    def device_type(self):
        dtype = self.args.get("Device Type")
        if dtype is None:
            return None

        try:
            return DeviceType(dtype)
        except ValueError:
            return None

    @property
    def device_id(self):
        return self.args.get("Device Id")

    @property
    def bytes(self):
        return self.args.get("Bytes", 0)

def create_event(event):
    try:
        type = event.get("ph")
        if type == "X":
            return create_trace_event(event)
        elif type == "i" and event.get('s') == 't':
            return MemoryEvent(EventTypes.MEMORY, event)
        else:
            return None
    except Exception as ex:
        logger.warning("Failed to parse profile event. Exception=%s. Event=%s", ex, event, exc_info=True)
        raise

def create_trace_event(event):
    category = event.get("cat")
    if category == "Operator":
        name = event.get("name")
        if name and name.startswith("ProfilerStep#"):
            return ProfilerStepEvent(event)

    if category in Supported_EventTypes:
        return TraceEvent(category, event)
    else:
        return None
