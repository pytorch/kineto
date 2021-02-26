# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import sys

from .trace import EventTypes
from .. import utils

logger = utils.get_logger()


class BaseNode:
    def __init__(self):
        self.name = None
        self.start_time = None
        self.end_time = None
        self.type = None
        self.external_id = None  # For consistency check.


class HostNode(BaseNode):
    def __init__(self):
        super(HostNode, self).__init__()
        self.device_duration = 0  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream?


class OperatorNode(HostNode):
    def __init__(self):
        super(OperatorNode, self).__init__()
        self.children = []  # OperatorNode and ProfilerStepNode.
        self.runtimes = []  # RuntimeNode
        self.input_shape = None
        self.self_host_duration = 0
        self.self_device_duration = 0

    def fill_stats(self):
        self.self_host_duration = self.end_time - self.start_time
        for child in self.children:
            self.device_duration += child.device_duration
            self.self_host_duration -= (child.end_time - child.start_time)
        for rt in self.runtimes:
            # From PyTorch 1.8 RC1, cpu_self_time does not include runtime's time.
            # So here we keep consistent with it.
            self.self_host_duration -= (rt.end_time - rt.start_time)
            self.device_duration += rt.device_duration
            self.self_device_duration += rt.device_duration


class ProfilerStepNode(OperatorNode):
    def __init__(self):
        super(ProfilerStepNode, self).__init__()


class RuntimeNode(HostNode):
    def __init__(self):
        super(RuntimeNode, self).__init__()
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = None

    def fill_stats(self):
        if self.device_nodes is None:
            return
        for device_node in self.device_nodes:
            self.device_duration += device_node.end_time - device_node.start_time


class DeviceNode(BaseNode):
    def __init__(self):
        super(DeviceNode, self).__init__()
        self.op_node = None  # The cpu operator that launched it.


class OperatorAgg:
    def __init__(self):
        self.name = None
        self.input_shape = None  # Optional
        self.calls = 0
        self.host_duration = 0
        self.device_duration = 0
        self.self_host_duration = 0
        self.self_device_duration = 0
        # TODO: Think about adding these avgs to UI.
        self.avg_host_duration = 0
        self.avg_device_duration = 0

    def average(self):
        self.avg_host_duration = self.host_duration / self.calls
        self.avg_device_duration = self.device_duration / self.calls


class KernelAggByNameOp:
    def __init__(self):
        self.name = None
        self.op_name = None
        self.calls = 0
        self.total_duration = 0
        self.avg_duration = 0
        self.min_duration = sys.maxsize
        self.max_duration = 0

    def average(self):
        self.avg_duration = self.total_duration / self.calls


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
            root_node = OperatorNode()
            root_node.start_time = -sys.maxsize - 1
            root_node.end_time = sys.maxsize
            root_node.runtimes = zero_rt_list  # Give the list of RuntimeNode with external_id=0 to root node.
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
            root_node.name = "CallTreeRoot"
            root_node.type = EventTypes.PYTHON
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

        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        def fill_stats(node):
            if node.type != EventTypes.RUNTIME:
                for child in node.children:
                    fill_stats(child)
                for rt in node.runtimes:
                    fill_stats(rt)
                    if rt.device_nodes is not None:
                        for device_node in rt.device_nodes:
                            device_node.op_node = node

            if node.name == "CallTreeRoot":
                node.start_time = node.end_time = None
                for i in range(len(node.children)):
                    if node.children[i].start_time is not None:
                        node.start_time = node.children[i].start_time
                        break
                for i in range(len(node.children) - 1, -1, -1):
                    if node.children[i].end_time is not None:
                        node.end_time = node.children[i].end_time
                        break

            node.fill_stats()
            if type(node) is OperatorNode and node.type == EventTypes.OPERATOR \
                    and not (node.name.startswith("enumerate(DataLoader)#") and node.name.endswith(".__next__")) \
                    and not node.name.startswith("Optimizer."):
                self.cpp_op_list.append(node)
            if node.type == EventTypes.RUNTIME and node.device_nodes is not None:
                self.kernel_list.extend([n for n in node.device_nodes if n.type == EventTypes.KERNEL])

        root_node = build_tree_relationship(host_node_list, zero_rt_list)
        remove_dup_nodes(root_node)
        fill_stats(root_node)
        return root_node

    def parse_events(self, events):

        def parse_event(event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list):

            def build_node(node, event):
                node.name = event.name
                node.start_time = event.ts
                node.end_time = event.ts + event.duration
                node.type = event.type
                if "external id" in event.args:
                    node.external_id = event.args["external id"]
                elif "External id" in event.args:
                    node.external_id = event.args["External id"]

            corrid = event.args["correlation"] if "correlation" in event.args else None
            input_shape = event.args["Input dims"] if "Input dims" in event.args else None
            tid = event.tid
            if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
                device_node = DeviceNode()
                build_node(device_node, event)
                if corrid in corrid_to_runtime:
                    rt_node = corrid_to_runtime[corrid]  # Don't pop it because it may be used by next kernel.
                    if rt_node.device_nodes is None:
                        rt_node.device_nodes = [device_node]
                    else:
                        rt_node.device_nodes.append(device_node)
                    if rt_node.external_id != device_node.external_id:
                        logger.warning(
                            "Runtime and Device-op have same correlation id but with different external id!"
                        )
                else:
                    if corrid not in corrid_to_device:
                        corrid_to_device[corrid] = [device_node]
                    else:
                        corrid_to_device[corrid].append(device_node)
            elif event.type == EventTypes.RUNTIME:
                rt_node = RuntimeNode()
                build_node(rt_node, event)
                corrid_to_runtime[corrid] = rt_node
                if corrid in corrid_to_device:
                    rt_node.device_nodes = []
                    rt_node.device_nodes.extend(corrid_to_device[corrid])
                    for device_node in corrid_to_device[corrid]:
                        if rt_node.external_id != device_node.external_id:
                            logger.warning(
                                "Runtime and Device-op have same correlation id but with different external id!"
                            )
                    del corrid_to_device[corrid]
                if rt_node.external_id in externalid_to_runtime:
                    externalid_to_runtime[rt_node.external_id].append(rt_node)
                else:
                    externalid_to_runtime[rt_node.external_id] = [rt_node]
                # Some runtimes has external_id 0, which will not be correlated to any operator.
                # So get them and attach them to root node.
                if rt_node.external_id == 0:
                    if tid not in tid2zero_rt_list:
                        tid2zero_rt_list[tid] = []
                    tid2zero_rt_list[tid].append(rt_node)
            elif event.type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
                if event.type == EventTypes.PROFILER_STEP:
                    op_node = ProfilerStepNode()
                else:
                    op_node = OperatorNode()
                build_node(op_node, event)
                op_node.input_shape = input_shape
                if tid not in tid2list:
                    tid2list[tid] = []
                tid2list[tid].append(op_node)

        def parse_ops(cpp_op_list):
            def aggregate(key_to_agg, key, op):
                if key not in key_to_agg:
                    key_to_agg[key] = OperatorAgg()
                agg = key_to_agg[key]
                agg.name = op.name
                agg.input_shape = str(op.input_shape)
                agg.calls += 1
                agg.host_duration += op.end_time - op.start_time
                agg.device_duration += op.device_duration
                agg.self_host_duration += op.self_host_duration
                agg.self_device_duration += op.self_device_duration
                return agg

            name_to_agg = {}
            for op in cpp_op_list:
                agg = aggregate(name_to_agg, op.name, op)
            for _, agg in name_to_agg.items():
                agg.average()
            op_list_groupby_name = list(name_to_agg.values())

            name_input_to_agg = {}
            for op in cpp_op_list:
                name_input = op.name + "###" + str(op.input_shape)
                agg = aggregate(name_input_to_agg, name_input, op)
            for _, agg in name_input_to_agg.items():
                agg.average()
            op_list_groupby_name_input = list(name_input_to_agg.values())

            return op_list_groupby_name, op_list_groupby_name_input

        def parse_kernels(kernel_list):
            name_op_to_agg = {}
            for kernel in kernel_list:
                op_name = "N/A" if kernel.op_node is None else kernel.op_node.name
                key = kernel.name + "###" + op_name
                if key not in name_op_to_agg:
                    name_op_to_agg[key] = KernelAggByNameOp()
                agg = name_op_to_agg[key]
                agg.name = kernel.name
                agg.op_name = op_name
                agg.calls += 1
                dur = kernel.end_time - kernel.start_time
                agg.total_duration += dur
                agg.min_duration = min(agg.min_duration, dur)
                agg.max_duration = max(agg.max_duration, dur)
            for _, agg in name_op_to_agg.items():
                agg.average()
            kernel_list_groupby_name_op = list(name_op_to_agg.values())

            return kernel_list_groupby_name_op

        # For OperatorNode and ProfilerStepNode:
        #   Use time interval containing relationship to build father-child correlation,
        #   which is consistent with autograd profiler.
        # For RuntimeNode:
        #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
        #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
        #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
        tid2list = {}  # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
        tid2zero_rt_list = {}  # value is a list of RuntimeNode with external_id=0. They will be attached to root nodes.
        corrid_to_device = {}  # value is a list of DeviceNode
        corrid_to_runtime = {}  # value is a RuntimeNode
        externalid_to_runtime = {}  # value is a list of RuntimeNode
        for event in events:
            parse_event(event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list)
        # Kernel that not owned by any operator should also be shown in kernel view
        # when group by "Kernel Name + Op Name".
        for _, device_nodes in corrid_to_device.items():
            self.kernel_list.extend([n for n in device_nodes if n.type == EventTypes.KERNEL])
        # associate CUDA Runtimes with CPU events
        for _, op_list in tid2list.items():
            for op in op_list:
                if op.external_id in externalid_to_runtime:
                    op.runtimes.extend(externalid_to_runtime[op.external_id])
                    del externalid_to_runtime[op.external_id]
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.warning("{} Runtime with external id {} don't correlate to any operator!".format(
                    len(externalid_to_runtime[ext_id]), ext_id))
        for tid, op_list in tid2list.items():
            zero_rt_list = tid2zero_rt_list[tid] if tid in tid2zero_rt_list else []
            # Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
            op_list.sort(key=lambda x: (x.start_time, -x.end_time))
            root_node = self._build_tree(op_list, zero_rt_list)
            self.tid2tree[tid] = root_node
        self.op_list_groupby_name, self.op_list_groupby_name_input = parse_ops(self.cpp_op_list)
        self.kernel_list_groupby_name_op = parse_kernels(self.kernel_list)
