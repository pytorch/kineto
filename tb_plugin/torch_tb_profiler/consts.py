# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import re
from collections import namedtuple

PLUGIN_NAME = 'pytorch_profiler'

WORKER_PATTERN = re.compile(r"""^(.*?) # worker name
        (\.\d+)? # optional timestamp like 1619499959628 used as span name
        \.pt\.trace\.json # the ending suffix
        (?:\.gz)?$""", re.X)  # optional .gz extension

NODE_PROCESS_PATTERN = re.compile(r"""^(.*)_(\d+)""")
MONITOR_RUN_REFRESH_INTERNAL_IN_SECONDS = 10
MAX_GPU_PER_NODE = 64

View = namedtuple('View', 'id, name, display_name')
OVERALL_VIEW = View(1, 'overall', 'Overview')
OP_VIEW = View(2, 'operator', 'Operator')
KERNEL_VIEW = View(3, 'kernel', 'Kernel')
TRACE_VIEW = View(4, 'trace', 'Trace')
DISTRIBUTED_VIEW = View(5, 'distributed', 'Distributed')
MEMORY_VIEW = View(6, 'memory', 'Memory')
MODULE_VIEW = View(7, 'module', 'Module')
LIGHTNING_VIEW = View(8, 'lightning', 'Lightning')

TOOLTIP_GPU_UTIL = \
    'GPU Utilization:\n' \
    'GPU busy time / All steps time. The higher, the better. ' \
    'GPU busy time is the time during which there is at least one GPU kernel running on it. ' \
    'All steps time is the total time of all profiler steps(or called as iterations).\n'
TOOLTIP_SM_EFFICIENCY = \
    'Est. SM Efficiency:\n' \
    'Estimated Stream Multiprocessor Efficiency. The higher, the better. ' \
    'This metric of a kernel, SM_Eff_K = min(blocks of this kernel / SM number of this GPU, 100%). ' \
    "This overall number is the sum of all kernels' SM_Eff_K weighted by kernel's execution duration, " \
    'divided by all steps time.\n'
TOOLTIP_OCCUPANCY_COMMON = \
    'Est. Achieved Occupancy:\n' \
    'For most cases such as memory bandwidth bounded kernels, the higher the better. ' \
    'Occupancy is the ratio of active warps on an SM ' \
    'to the maximum number of active warps supported by the SM. ' \
    'The theoretical occupancy of a kernel is upper limit occupancy of this kernel, ' \
    'limited by multiple factors such as kernel shape, kernel used resource, ' \
    'and the GPU compute capability.\n' \
    'Est. Achieved Occupancy of a kernel, OCC_K = ' \
    'min(threads of the kernel / SM number / max threads per SM, theoretical occupancy of the kernel). '
TOOLTIP_OCCUPANCY_OVERVIEW = \
    "This overall number is the weighted average of all kernels' OCC_K " \
    "using kernel's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.\n'
TOOLTIP_TENSOR_CORES = \
    'Kernel using Tensor Cores:\n' \
    'Total GPU Time for Tensor Core kernels / Total GPU Time for all kernels.\n'
TOOLTIP_OCCUPANCY_TABLE = \
    "This \"Mean\" number is the weighted average of all calls' OCC_K of the kernel, " \
    "using each call's execution duration as weight. " \
    'It shows fine-grained low-level GPU utilization.'
TOOLTIP_BLOCKS_PER_SM = \
    'Blocks Per SM = blocks of this kernel / SM number of this GPU.\n' \
    'If this number is less than 1, it indicates the GPU multiprocessors are not fully utilized.\n' \
    '\"Mean Blocks per SM\" is the weighted average of all calls of this kernel, ' \
    "using each call's execution duration as weight."
TOOLTIP_OP_TC_ELIGIBLE = \
    'Whether this operator is eligible to use Tensor Cores.'
TOOLTIP_OP_TC_SELF = \
    'Time of self-kernels with Tensor Cores / Time of self-kernels.'
TOOLTIP_OP_TC_TOTAL = \
    'Time of kernels with Tensor Cores / Time of kernels.'
TOOLTIP_KERNEL_USES_TC = \
    'Whether this kernel uses Tensor Cores.'
TOOLTIP_KERNEL_OP_TC_ELIGIBLE = \
    'Whether the operator launched this kernel is eligible to use Tensor Cores.'
