# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import utils

__all__ = ["EventTypes", "create_event"]

logger = utils.get_logger()


class EventTypes(object):
    NET = "Net"
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
        self.name = data.get("name", None)
        self.ts = data.get("ts", None)
        self.duration = data.get("dur", None)
        self.pid = data.get("pid", None)
        self.tid = data.get("tid", None)
        self.args = data.get("args", {})


class ProfilerStepEvent(TraceEvent):
    def __init__(self, data):
        super(ProfilerStepEvent, self).__init__(EventTypes.PROFILER_STEP, data)
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
            if name.startswith("ProfilerStep#"):
                return ProfilerStepEvent(event)

        if category in Supported_EventTypes:
            return TraceEvent(category, event)
        else:
            return None
    except Exception as ex:
        logger.warning("Failed to parse profile event. Exception=%s. Event=%s", ex, event, exc_info=True)
        raise ex
