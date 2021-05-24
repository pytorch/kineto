# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import re
from collections import namedtuple

PLUGIN_NAME = "pytorch_profiler"

WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        # TODO: uncomment the following line when we need supprort multiple steps
        # (?:\.\d+)? # optional timestamp like 1619499959628
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""", re.X)  # optional .gz extension

NODE_PROCESS_PATTERN = re.compile(r"""^(.*)_(\d+)""")
MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10
MAX_GPU_PER_NODE = 64

View = namedtuple("View", "id, name, display_name")
OVERALL_VIEW = View(1, "overall", "Overview")
OP_VIEW = View(2, "operator", "Operator")
KERNEL_VIEW = View(3, "kernel", "Kernel")
TRACE_VIEW = View(4, "trace", "Trace")
DISTRIBUTED_VIEW = View(5, "distributed", "Distributed")
