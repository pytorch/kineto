# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys
from collections import defaultdict
from typing import Dict, List

from .. import utils

logger = utils.get_logger()


class OperatorAgg:
    def __init__(self, op):
        self.name = op.name
        self.input_shape = str(op.input_shape)  # Optional

        self.callstacks = set()  # Optional
        self.calls = 0
        self.host_duration = 0
        self.device_duration = 0
        self.self_host_duration = 0
        self.self_device_duration = 0
        self.tc_eligible = op.tc_eligible
        self.tc_self_duration = 0
        self.tc_total_duration = 0
        # TODO: Think about adding these avgs to UI.

    @property
    def tc_self_ratio(self):
        return self.tc_self_duration / self.self_device_duration if self.self_device_duration > 0 else 0

    @property
    def tc_total_ratio(self):
        return self.tc_total_duration / self.device_duration if self.device_duration > 0 else 0


def aggregate_ops(op_list, keys_func):
    def aggregate(key_to_agg, key, op):
        if key not in key_to_agg:
            key_to_agg[key] = OperatorAgg(op)
        agg = key_to_agg[key]
        agg.callstacks.add(op.callstack)
        agg.calls += 1
        agg.host_duration += op.end_time - op.start_time
        agg.device_duration += op.device_duration
        agg.self_host_duration += op.self_host_duration
        agg.self_device_duration += op.self_device_duration
        agg.tc_self_duration += op.tc_self_duration
        agg.tc_total_duration += op.tc_total_duration
        return agg

    agg_dicts = [{} for _ in range(len(keys_func))]
    for op in op_list:
        for i, key_func in enumerate(keys_func):
            key = key_func(op)
            aggregate(agg_dicts[i], key, op)

    return agg_dicts


class KernelAggByNameOp:
    def __init__(self, kernel, op_name):
        self.name = kernel.name
        self.op_name = op_name
        self.grid = kernel.grid
        self.block = kernel.block
        self.regs_per_thread = kernel.regs_per_thread
        self.shared_memory = kernel.shared_memory

        self.calls = 0
        self.total_duration = 0
        self.min_duration = sys.maxsize
        self.max_duration = 0
        self.blocks_per_sm = 0.0
        self.occupancy = 0.0
        self.tc_used = kernel.tc_used
        self.op_tc_eligible = kernel.op_tc_eligible

    @property
    def avg_duration(self):
        return self.total_duration / self.calls

    @property
    def avg_blocks_per_sm(self):
        return self.blocks_per_sm / self.total_duration if self.total_duration > 0 else 0

    @property
    def avg_occupancy(self):
        return self.occupancy / self.total_duration if self.total_duration > 0 else 0


def aggregate_kernels(kernel_list):
    name_op_to_agg = {}
    for kernel in kernel_list:
        dur = kernel.end_time - kernel.start_time
        op_name = "N/A" if kernel.op_name is None else kernel.op_name
        key = "###".join((kernel.name, op_name,
                            str(kernel.grid), str(kernel.block),
                            str(kernel.regs_per_thread or '0'), str(kernel.shared_memory or '0')))
        if key not in name_op_to_agg:
            name_op_to_agg[key] = KernelAggByNameOp(kernel, op_name)
        agg = name_op_to_agg[key]
        agg.calls += 1
        agg.total_duration += dur
        agg.min_duration = min(agg.min_duration, dur)
        agg.max_duration = max(agg.max_duration, dur)
        agg.blocks_per_sm += float(kernel.blocks_per_sm or 0) * dur
        agg.occupancy += float(kernel.occupancy or 0) * dur

    kernel_list_groupby_name_op = list(name_op_to_agg.values())
    return kernel_list_groupby_name_op


class ModuleAggregator:

    def __init__(self):
        self.op_list_groupby_name: List[OperatorAgg] = None  # For Operator-view.
        self.op_list_groupby_name_input: List[OperatorAgg] = None  # For Operator-view.
        self.kernel_list_groupby_name_op: Dict[str, KernelAggByNameOp] = None  # For Kernel-view.
        self.stack_lists_group_by_name: Dict[str, List[OperatorAgg]] = None
        self.stack_lists_group_by_name_input: Dict[str, List[OperatorAgg]] = None
        self.ops = None

    def aggregate(self, tid2tree):
        # get the operators and kernels recursively by traverse the node tree root.
        ops = []
        kernels = []
        for root in tid2tree.values():
            root_ops, root_kernels = root.get_operator_and_kernels()
            ops.extend(root_ops)
            kernels.extend(root_kernels)

        # aggregate both kernels and operators
        self.kernel_list_groupby_name_op = aggregate_kernels(kernels)

        keys = [
            lambda x: x.name,
            lambda x: x.name + "###" + str(x.input_shape),
            lambda x: x.name + "###" + str(x.callstack),
            lambda x: x.name + "###" + str(x.input_shape) + "###" + str(x.callstack)
        ]
        agg_result = aggregate_ops(ops, keys)
        stack_lists_group_by_name = defaultdict(list)
        stack_lists_group_by_name_input = defaultdict(list)
        for agg in agg_result[2].values():
            assert (len(agg.callstacks) == 1)
            if list(agg.callstacks)[0]:
                stack_lists_group_by_name[agg.name].append(agg)
        for agg in agg_result[3].values():
            assert (len(agg.callstacks) == 1)
            if list(agg.callstacks)[0]:
                key = agg.name + "###" + str(agg.input_shape)
                stack_lists_group_by_name_input[key].append(agg)

        self.op_list_groupby_name = list(agg_result[0].values())
        self.op_list_groupby_name_input = list(agg_result[1].values())
        self.stack_lists_group_by_name = stack_lists_group_by_name
        self.stack_lists_group_by_name_input = stack_lists_group_by_name_input
        self.ops = ops
