# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

PLUGIN_NAME = "pytorch_profiler"

TRACE_FILE_SUFFIX = ".pt.trace.json"
TRACE_GZIP_FILE_SUFFIX = ".pt.trace.json.gz"

MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10

View = namedtuple("View", "id, name, display_name")
OVERALL_VIEW = View(1, "overall", "Overview")
OP_VIEW = View(2, "operator", "Operator")
KERNEL_VIEW = View(3, "kernel", "Kernel")
TRACE_VIEW = View(4, "trace", "Trace")
