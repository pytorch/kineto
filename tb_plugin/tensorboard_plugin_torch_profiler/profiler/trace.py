# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import utils

__all__ = ["EventTypes", "get_event_parser"]

logger = utils.get_logger()


class EventTypes(object):
    NET = "NetEvent"
    OPERATOR = "OperatorEvent"
    PROFILER_STEP = "ProfilerStepEvent"
    RUNTIME = "RuntimeEvent"
    KERNEL = "KernelEvent"
    MEMCPY = "MemcpyEvent"
    MEMSET = "MemsetEvent"
    PYTHON = "PythonEvent"


class TraceEvent(object):
    def __init__(self, type, data):
        self.type = type
        self.category = data.get("cat", "")
        self.name = data.get("name", None)
        self.ts = data.get("ts", None)
        self.duration = data.get("dur", None)
        self.pid = data.get("pid", None)
        self.tid = data.get("tid", None)
        self.args = data.get("args", None)

    def to_dict(self):
        return vars(self)


class NetEvent(TraceEvent):
    def __init__(self, data):
        super(NetEvent, self).__init__(EventTypes.NET, data)


class OperatorEvent(TraceEvent):
    def __init__(self, data):
        super(OperatorEvent, self).__init__(EventTypes.OPERATOR, data)


class ProfilerStepEvent(TraceEvent):
    def __init__(self, data):
        super(ProfilerStepEvent, self).__init__(EventTypes.PROFILER_STEP, data)
        # In torch.profiler, next_step will invoke record_function with name like "ProfilerStep#5"
        self.step = int(self.name.split("#")[1])


class RuntimeEvent(TraceEvent):
    def __init__(self, data):
        super(RuntimeEvent, self).__init__(EventTypes.RUNTIME, data)


class KernelEvent(TraceEvent):
    def __init__(self, data):
        super(KernelEvent, self).__init__(EventTypes.KERNEL, data)


class MemcpyEvent(TraceEvent):
    def __init__(self, data):
        super(MemcpyEvent, self).__init__(EventTypes.MEMCPY, data)


class MemsetEvent(TraceEvent):
    def __init__(self, data):
        super(MemsetEvent, self).__init__(EventTypes.MEMSET, data)


class PythonEvent(TraceEvent):
    def __init__(self, data):
        super(PythonEvent, self).__init__(EventTypes.PYTHON, data)


class EventParser(object):
    def __init__(self):
        self._handlers = {
            "X": {
                "Net": NetEvent,
                "Operator": self._parse_operator_event,
                "Runtime": RuntimeEvent,
                "Kernel": KernelEvent,
                "Memcpy": MemcpyEvent,
                "Memset": MemsetEvent,
                "Python": PythonEvent,
            }
        }

    def _get_handler(self, type=None, category=None):
        handlers = self._handlers.get(type, None)
        if handlers is None:
            return None
        return handlers.get(category, None)

    def parse(self, event):
        try:
            type = event.get("ph", None)
            category = event.get("cat", None)
            handler = self._get_handler(type, category)
            if handler is None:
                return None
            return handler(event)
        except Exception as ex:
            logger.warning("Failed to parse profile event. Exception=%s. Event=%s", ex, event, exc_info=True)
            raise ex

    def _parse_operator_event(self, event):
        name = event.get("name")
        if name.startswith("ProfilerStep#"):
            return ProfilerStepEvent(event)
        return OperatorEvent(event)


def get_event_parser(version=None):
    return EventParser()
