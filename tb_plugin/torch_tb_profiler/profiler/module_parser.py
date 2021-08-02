# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys
from collections import defaultdict
from typing import Dict, List

from .. import utils
from .node import OperatorNode, RuntimeNode
from .trace import EventTypes

logger = utils.get_logger()


class OperatorAgg:
    def __init__(self, op):
        self.name = op.name
        self.input_shape = str(op.input_shape)  # Optional

        self.call_stacks = set()  # Optional
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
        agg.call_stacks.add(op.call_stack)
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
        self.op_tc_eligible = kernel.op_node_ref().tc_eligible if kernel.op_node_ref is not None else False

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
        op_name = "N/A" if kernel.op_node_ref is None else kernel.op_node_ref().name
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


class ModuleParser:
    def __init__(self):
        self.tid2tree = {}

    def build_tree(self, context):
        tid2list = context.tid2list
        tid2zero_rt_list = context.tid2zero_rt_list
        corrid_to_device = context.corrid_to_device

        staled_device_nodes = []
        for _, device_nodes in corrid_to_device.items():
             staled_device_nodes.extend([n for n in device_nodes if n.type == EventTypes.KERNEL])

        for tid, op_list in tid2list.items():
            zero_rt_list = tid2zero_rt_list[tid] if tid in tid2zero_rt_list else []
            # Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
            op_list.sort(key=lambda x: (x.start_time, -x.end_time))
            main_tid = any([op.name.startswith("ProfilerStep#") for op in op_list])
            if main_tid:
                # only append the staled device nodes into main thread
                root_node = self._build_tree(op_list, zero_rt_list, tid, staled_device_nodes)
            else:
                root_node = self._build_tree(op_list, zero_rt_list, tid, [])
            self.tid2tree[int(tid)] = root_node

    def _build_tree(self, host_node_list, zero_rt_list, tid, device_nodes):
        '''host_node_list: list of OperatorNode and ProfilerStepNode.
        zero_rt_list: list of RuntimeNode with external_id=0.'''

        def build_tree_relationship(host_node_list, zero_rt_list, device_nodes):
            dummpy_rt = []
            if device_nodes:
                dummpy_rt.append(RuntimeNode("dummy", 0, 0, EventTypes.RUNTIME, 0, None, 0, device_nodes))
                dummpy_rt[0].fill_stats()
            node_stack = []
            root_node = OperatorNode(
                name="CallTreeRoot",
                start_time=-sys.maxsize - 1,
                end_time=sys.maxsize,
                type=EventTypes.PYTHON,
                tid=tid,
                runtimes=zero_rt_list + dummpy_rt) # Give the list of RuntimeNode with external_id=0 to root node.
            node_stack.append(root_node)
            for node in host_node_list:
                while True:  # break loop when the node is inserted.
                    tail_node = node_stack[-1]
                    if node.start_time < tail_node.end_time:
                        if node.end_time <= tail_node.end_time:
                            tail_node.children.append(node)
                            # node.parent_node = weakref.ref(tail_node)
                            node_stack.append(node)
                        else:
                            logger.error("Error in input data: ranges on the same thread should not intersect!"
                                         "Father:({},{},{}) Child:({},{},{})".format(
                                tail_node.name, tail_node.start_time, tail_node.end_time,
                                node.name, node.start_time, node.end_time
                            ))
                        break
                    else:
                        node_stack.pop()
            return root_node

        # Merge the consecutive calls to same function into one.
        # Just follow the same pattern in torch/autograd/profiler.py,
        # EventList._remove_dup_nodes
        # TODO: Replace recursive by for loop, in case of too deep callstack.
        def remove_dup_nodes(node):
            if node.type == EventTypes.RUNTIME:
                return
            if len(node.children) == 1:
                child = node.children[0]
                if node.name == child.name and node.type == EventTypes.OPERATOR and child.type == EventTypes.OPERATOR:
                    node.children = child.children
                    node.runtimes = child.runtimes  # Keep consistent with autograd profiler.
                    remove_dup_nodes(node)  # This node may have to merge with child's child.
            for child in node.children:
                remove_dup_nodes(child)

        root_node = build_tree_relationship(host_node_list, zero_rt_list, device_nodes)
        remove_dup_nodes(root_node)
        root_node.replace_time_by_children()
        root_node.fill_stats()
        return root_node

class ModuleAggregator:

    def __init__(self):
        self.op_list_groupby_name: List[OperatorAgg] = None  # For Operator-view.
        self.op_list_groupby_name_input: List[OperatorAgg] = None  # For Operator-view.
        self.kernel_list_groupby_name_op: Dict[str, KernelAggByNameOp] = None  # For Kernel-view.
        self.stack_lists_group_by_name: Dict[str, List[OperatorAgg]] = None
        self.stack_lists_group_by_name_input: Dict[str, List[OperatorAgg]] = None

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
            lambda x: x.name + "###" + str(x.call_stack),
            lambda x: x.name + "###" + str(x.input_shape) + "###" + str(x.call_stack)
        ]
        agg_result = aggregate_ops(ops, keys)
        stack_lists_group_by_name = defaultdict(list)
        stack_lists_group_by_name_input = defaultdict(list)
        for agg in agg_result[2].values():
            assert (len(agg.call_stacks) == 1)
            if list(agg.call_stacks)[0]:
                stack_lists_group_by_name[agg.name].append(agg)
        for agg in agg_result[3].values():
            assert (len(agg.call_stacks) == 1)
            if list(agg.call_stacks)[0]:
                key = agg.name + "###" + str(agg.input_shape)
                stack_lists_group_by_name_input[key].append(agg)

        self.op_list_groupby_name = list(agg_result[0].values())
        self.op_list_groupby_name_input = list(agg_result[1].values())
        self.stack_lists_group_by_name = stack_lists_group_by_name
        self.stack_lists_group_by_name_input = stack_lists_group_by_name_input
