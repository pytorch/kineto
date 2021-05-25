# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys

from .. import utils
from .node import OperatorNode
from .trace import EventTypes

logger = utils.get_logger()


class OperatorAgg:
    def __init__(self):
        self.name = None
        self.input_shape = None  # Optional
        self.call_stacks = set()  # Optional
        self.calls = 0
        self.host_duration = 0
        self.device_duration = 0
        self.self_host_duration = 0
        self.self_device_duration = 0
        # TODO: Think about adding these avgs to UI.

    @property
    def avg_host_duration(self):
        return self.host_duration / self.calls

    @property
    def avg_device_duration(self):
        return  self.device_duration / self.calls


class KernelAggByNameOp:
    def __init__(self):
        self.name = None
        self.op_name = None
        self.calls = 0
        self.total_duration = 0
        self.min_duration = sys.maxsize
        self.max_duration = 0
        self.blocks_per_sm = 0
        self.occupancy = 0

    @property
    def avg_duration(self):
        return self.total_duration / self.calls

    @property
    def avg_blocks_per_sm(self):
        return self.blocks_per_sm / self.total_duration if self.total_duration > 0 else 0

    @property
    def avg_occupancy(self):
        return self.occupancy / self.total_duration if self.total_duration > 0 else 0


class ModuleParser:
    def __init__(self):
        self.tid2tree = {}
        self.cpp_op_list = []  # For Operator-view.
        self.kernel_list = []  # For Kernel-view.
        self.op_list_groupby_name = []  # For Operator-view.
        self.op_list_groupby_name_input = []  # For Operator-view.
        self.kernel_list_groupby_name_op = {}  # For Kernel-view.

    # host_node_list: list of OperatorNode and ProfilerStepNode.
    # zero_rt_list: list of RuntimeNode with external_id=0.
    def _build_tree(self, host_node_list, zero_rt_list):

        def build_tree_relationship(host_node_list, zero_rt_list):
            node_stack = []
            root_node = OperatorNode(
                name="CallTreeRoot",
                start_time=-sys.maxsize - 1,
                end_time=sys.maxsize,
                type=EventTypes.PYTHON,
                runtimes=zero_rt_list) # Give the list of RuntimeNode with external_id=0 to root node.
            node_stack.append(root_node)
            for node in host_node_list:
                while True:  # break loop when the node is inserted.
                    tail_node = node_stack[-1]
                    if node.start_time < tail_node.end_time:
                        if node.end_time <= tail_node.end_time:
                            tail_node.children.append(node)
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

        def traverse_node(node):
            if node.type != EventTypes.RUNTIME:
                for child in node.children:
                    traverse_node(child)
                for rt in node.runtimes:
                    traverse_node(rt)

            if type(node) is OperatorNode and node.type == EventTypes.OPERATOR \
                    and not (node.name.startswith("enumerate(DataLoader)#") and node.name.endswith(".__next__")) \
                    and not node.name.startswith("Optimizer."):
                self.cpp_op_list.append(node)
            if node.type == EventTypes.RUNTIME and node.device_nodes is not None:
                self.kernel_list.extend([n for n in node.device_nodes if n.type == EventTypes.KERNEL])

        root_node = build_tree_relationship(host_node_list, zero_rt_list)
        remove_dup_nodes(root_node)
        root_node.replace_time_by_children()
        root_node.fill_stats()
        traverse_node(root_node)
        return root_node

    def aggregate(self, context):

        def parse_ops(cpp_op_list):
            def aggregate(key_to_agg, key, op):
                if key not in key_to_agg:
                    key_to_agg[key] = OperatorAgg()
                agg = key_to_agg[key]
                agg.name = op.name
                agg.input_shape = str(op.input_shape)
                agg.call_stacks.add(op.call_stack)
                agg.calls += 1
                agg.host_duration += op.end_time - op.start_time
                agg.device_duration += op.device_duration
                agg.self_host_duration += op.self_host_duration
                agg.self_device_duration += op.self_device_duration
                return agg

            name_to_agg = {}
            name_input_to_agg = {}
            name_stack_to_agg = {}
            name_input_stack_to_agg = {}
            for op in cpp_op_list:
                aggregate(name_to_agg, op.name, op)
                aggregate(name_input_to_agg, op.name + "###" + str(op.input_shape), op)
                aggregate(name_stack_to_agg, op.name + "###" + str(op.call_stack), op)
                aggregate(name_input_stack_to_agg, op.name + "###" + str(op.input_shape) + "###" + str(op.call_stack), op)

            op_list_groupby_name = list(name_to_agg.values())
            op_list_groupby_name_input = list(name_input_to_agg.values())
            stack_lists_group_by_name = dict()
            stack_lists_group_by_name_input = dict()
            for agg in name_stack_to_agg.values():
                assert (len(agg.call_stacks) == 1)
                if list(agg.call_stacks)[0]:
                    stack_lists_group_by_name.setdefault(agg.name, []).append(agg)
            for agg in name_input_stack_to_agg.values():
                assert (len(agg.call_stacks) == 1)
                if list(agg.call_stacks)[0]:
                    key = agg.name + "###" + str(agg.input_shape)
                    stack_lists_group_by_name_input.setdefault(key, []).append(agg)

            return op_list_groupby_name, op_list_groupby_name_input, stack_lists_group_by_name, stack_lists_group_by_name_input

        def parse_kernels(kernel_list):
            name_op_to_agg = {}
            for kernel in kernel_list:
                dur = kernel.end_time - kernel.start_time
                # remove 0 duration kernels in case of divided by zero.
                if dur > 0:
                    op_name = "N/A" if kernel.op_node is None else kernel.op_node.name
                    key = kernel.name + "###" + op_name
                    if key not in name_op_to_agg:
                        name_op_to_agg[key] = KernelAggByNameOp()
                    agg = name_op_to_agg[key]
                    agg.name = kernel.name
                    agg.op_name = op_name
                    agg.calls += 1
                    agg.total_duration += dur
                    agg.min_duration = min(agg.min_duration, dur)
                    agg.max_duration = max(agg.max_duration, dur)
                    agg.blocks_per_sm += kernel.blocks_per_sm * dur
                    agg.occupancy += kernel.occupancy * dur

            kernel_list_groupby_name_op = list(name_op_to_agg.values())
            return kernel_list_groupby_name_op

        tid2list = context.tid2list
        tid2zero_rt_list = context.tid2zero_rt_list
        corrid_to_device = context.corrid_to_device

        # Kernel that not owned by any operator should also be shown in kernel view
        # when group by "Kernel Name + Op Name".
        for _, device_nodes in corrid_to_device.items():
            self.kernel_list.extend([n for n in device_nodes if n.type == EventTypes.KERNEL])

        for tid, op_list in tid2list.items():
            zero_rt_list = tid2zero_rt_list[tid] if tid in tid2zero_rt_list else []
            # Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
            op_list.sort(key=lambda x: (x.start_time, -x.end_time))
            root_node = self._build_tree(op_list, zero_rt_list)
            self.tid2tree[tid] = root_node
        self.op_list_groupby_name, self.op_list_groupby_name_input, self.stack_lists_group_by_name, self.stack_lists_group_by_name_input = parse_ops(self.cpp_op_list)
        self.kernel_list_groupby_name_op = parse_kernels(self.kernel_list)
