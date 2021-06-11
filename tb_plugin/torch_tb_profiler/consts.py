# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import re
from collections import namedtuple

PLUGIN_NAME = "pytorch_profiler"

WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        (\.\d+)? # optional timestamp like 1619499959628 used as span name
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
MEMORY_VIEW = View(6, "memory", "Memory")

TOOLTIP_GPU_UTIL = \
    "GPU Utilization:\n" \
    "GPU busy time / All steps time. " \
    "GPU busy time is the time during which there is at least one GPU kernel running on it. " \
    "All steps time is the total time of all profiler steps(or called as iterations).\n"
TOOLTIP_SM_EFFICIENCY = \
    "Est. SM Efficiency:\n" \
    "Estimated Stream Multiprocessor Efficiency. " \
    "Est. SM Efficiency of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). " \
    "This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, " \
    "divided by all steps time.\n"
TOOLTIP_OCCUPANCY = \
    "Est. Achieved Occupancy:\n" \
    "Occupancy is the ratio of active threads on an SM " \
    "to the maximum number of active threads supported by the SM. " \
    "The theoretical occupancy of a kernel is upper limit occupancy of this kernel, " \
    "limited by multiple factors such as kernel shape, kernel used resource, " \
    "and the GPU compute capability." \
    "Est. Achieved Occupancy of a kernel, OCC_K = " \
    "min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). " \
    "This overall number is the weighted sum of all kernels OCC_K " \
    "using kernel's execution duration as weight."
TOOLTIP_BLOCKS_PER_SM = \
    "Blocks Per SM:\n" \
    "min(blocks of this kernel / SM number of this GPU). " \
    "If this number is less than 1, it indicates the GPU multiprocessors are not fully utilized."
