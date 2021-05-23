# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from .. import utils

__all__ = ["EventTypes", "create_event"]

logger = utils.get_logger()


class EventTypes(object):
    TRACE = "Trace"
    OPERATOR = "Operator"
    PROFILER_STEP = "ProfilerStep"
    RUNTIME = "Runtime"
    KERNEL = "Kernel"
    MEMCPY = "Memcpy"
    MEMSET = "Memset"
    PYTHON = "Python"

Supported_EventTypes = [v for k, v in vars(EventTypes).items() if not k.startswith("_") and v != EventTypes.PROFILER_STEP]

class TraceEvent(object):
    def __init__(self, type, data):
        self.type = type
        self.category = data.get("cat", "")
        self.name = data.get("name")
        self.ts = data.get("ts")
        self.duration = data.get("dur")
        self.pid = data.get("pid")
        self.tid = data.get("tid")
        self.args = data.get("args", {})

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


def create_event(event):
    try:
        type = event.get("ph")
        if type != "X":
            return None

        category = event.get("cat")
        if category == "Operator":
            name = event.get("name")
            if name and name.startswith("ProfilerStep#"):
                return ProfilerStepEvent(event)

        if category in Supported_EventTypes:
            return TraceEvent(category, event)
        else:
            return None
    except Exception as ex:
        logger.warning("Failed to parse profile event. Exception=%s. Event=%s", ex, event, exc_info=True)
        raise
