# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import sys
from abc import ABC

from .trace import EventTypes
from .. import utils

logger = utils.get_logger()


class BaseNode(ABC):
    def __init__(self, name, start_time, end_time, type, external_id):
        self.name = name
        self.start_time = start_time
        self.end_time = end_time
        self.type = type
        self.external_id = external_id  # For consistency check.

    @staticmethod
    def get_node_argument(event):
        kwargs = {}
        kwargs['name'] = event.name
        kwargs['start_time'] = event.ts
        kwargs['end_time'] = event.ts + event.duration
        kwargs['type'] = event.type

        if "external id" in event.args:
            kwargs['external_id'] = event.args["external id"]
        elif "External id" in event.args:
            kwargs['external_id'] = event.args["External id"]
        return kwargs


class HostNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, external_id, device_duration=0):
        super().__init__(name, start_time, end_time, type, external_id)
        self.device_duration = device_duration  # Total time of Kernel, GPU Memcpy, GPU Memset. TODO: parallel multi-stream?


class OperatorNode(HostNode):
    # Don't use [] as default parameters
    # https://stackoverflow.com/questions/1132941/least-astonishment-and-the-mutable-default-argument?page=1&tab=votes#tab-top
    # https://web.archive.org/web/20200221224620/http://effbot.org/zone/default-values.htm
    def __init__(self, name, start_time, end_time, type, external_id=None, device_duration=0,
            children=None, runtimes=None, input_shape=None, call_stack=None, self_host_duration=0, self_device_duration=0):
        super().__init__(name, start_time, end_time, type, external_id, device_duration)
        self.children = [] if children is None else children # OperatorNode and ProfilerStepNode.
        self.runtimes = [] if runtimes is None else runtimes # RuntimeNode
        self.input_shape = input_shape
        self.call_stack = call_stack
        self.self_host_duration = self_host_duration
        self.self_device_duration = self_device_duration

    def fill_stats(self):
        # TODO: Replace recursive by using a stack, in case of too deep callstack.
        for child in self.children:
            child.fill_stats()
        for rt in self.runtimes:
            rt.fill_stats()
            rt.update_device_op_node(self)

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

    def replace_time_by_children(self):
            self.start_time = next((child.start_time for child in self.children if child.start_time is not None), None)
            self.end_time = next((child.end_time for child in reversed(self.children) if child.end_time is not None), None)

    @classmethod
    def create(cls, event, input_shape, call_stack):
        kwargs = BaseNode.get_node_argument(event)
        return cls(input_shape=input_shape, call_stack=call_stack, **kwargs)


class ProfilerStepNode(OperatorNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RuntimeNode(HostNode):
    def __init__(self, name, start_time, end_time, type, external_id=None, device_duration=0,
            device_nodes=None):
        super().__init__(name, start_time, end_time, type, external_id, device_duration)
        # One runtime could trigger more than one kernel, such as cudaLaunchCooperativeKernelMultiDevice.
        self.device_nodes = device_nodes

    def fill_stats(self):
        if self.device_nodes:
            for device_node in self.device_nodes:
                self.device_duration += device_node.end_time - device_node.start_time

    def update_device_op_node(self, node):
        if self.device_nodes:
            for device_node in self.device_nodes:
                device_node.op_node = node

    @classmethod
    def create(cls, event, device_nodes):
        kwargs = BaseNode.get_node_argument(event)
        return cls(device_nodes=device_nodes, **kwargs)


class DeviceNode(BaseNode):
    def __init__(self, name, start_time, end_time, type, external_id=None,
            op_node=None):
        super().__init__(name, start_time, end_time, type, external_id)
        self.op_node = op_node  # The cpu operator that launched it.

    @classmethod
    def create(cls, event):
        kwargs = BaseNode.get_node_argument(event)
        return cls(**kwargs)


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

    @property
    def avg_duration(self):
        return self.total_duration / self.calls


class ModuleParser:
    def __init__(self):
        self.tid2tree = {}
        self.cpp_op_list = []  # For Operator-view.
        self.kernel_list = []  # For Kernel-view.
        self.op_list_groupby_name = []  # For Operator-view.
        self.op_list_groupby_name_input = []  # For Operator-view.
        self.kernel_list_groupby_name_op = {}  # For Kernel-view.
        self.runtime_node_list = []  # For Overall-view.
        self.device_node_list = []  # For Overall-view.

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

    def parse_events(self, events):

        def parse_event(event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list):
            corrid = event.args.get("correlation", None)
            input_shape = event.args.get("Input dims", None)
            call_stack = event.args.get("Call stack", "")
            tid = event.tid
            if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
                device_node = DeviceNode.create(event)
                if corrid in corrid_to_runtime:
                    rt_node = corrid_to_runtime[corrid]  # Don't pop it because it may be used by next kernel.
                    if rt_node.device_nodes is None:
                        rt_node.device_nodes = []
                    rt_node.device_nodes.append(device_node)

                    # Check the external_id
                    if rt_node.external_id != device_node.external_id:
                        logger.warning("Runtime and Device-op have same correlation id but with different external id!")
                else:
                    corrid_to_device.setdefault(corrid, []).append(device_node)
                self.device_node_list.append(device_node)
            elif event.type == EventTypes.RUNTIME:
                device_nodes = corrid_to_device.pop(corrid, None)
                rt_node = RuntimeNode.create(event, device_nodes)
                corrid_to_runtime[corrid] = rt_node
                externalid_to_runtime.setdefault(rt_node.external_id, []).append(rt_node)
                # Some runtimes has external_id 0, which will not be correlated to any operator.
                # So get them and attach them to root node.
                if rt_node.external_id == 0:
                    tid2zero_rt_list.setdefault(tid, []).append(rt_node)
                self.runtime_node_list.append(rt_node)

                # check the external_id
                if device_nodes:
                    for device_node in device_nodes:
                        if rt_node.external_id != device_node.external_id:
                            logger.warning("Runtime and Device-op have same correlation id but with different external id!")
            elif event.type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
                if event.type == EventTypes.PROFILER_STEP:
                    op_node = ProfilerStepNode.create(event, input_shape, None)
                else:
                    op_node = OperatorNode.create(event, input_shape, call_stack)
                tid2list.setdefault(tid, []).append(op_node)

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
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
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
        self.op_list_groupby_name, self.op_list_groupby_name_input, self.stack_lists_group_by_name, self.stack_lists_group_by_name_input = parse_ops(self.cpp_op_list)
        self.kernel_list_groupby_name_op = parse_kernels(self.kernel_list)
