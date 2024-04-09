# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .. import utils
from .node import (BackwardNode, DeviceNode, ModuleNode, OperatorNode,
                   ProfilerStepNode, RuntimeNode, is_operator_node)
from .trace import EventTypes

logger = utils.get_logger()


class OpTreeBuilder:
    BACKWARD_ROOT_PREFIX = 'autograd::engine::evaluate_function:'
    BACKWARD_ACCUMULATE_GRAD = 'autograd::engine::evaluate_function: torch::autograd::AccumulateGrad'

    def __init__(self):
        self.main_tid: int = None
        self.tid2tree: Dict[int, OperatorNode] = None

    def build_tree(self,
                   tid2list: Dict[int, List[OperatorNode]],
                   tid2zero_rt_list: Dict[int, List[RuntimeNode]],
                   staled_device_nodes: List[DeviceNode],
                   fwd_bwd_map: Dict[int, int]):
        """Construct the BackwardNode and replace the original backward nodes
        """
        self.tid2tree = self._build_tree(tid2list, tid2zero_rt_list, staled_device_nodes)

        # if could not find any forward/backward association, skip the processing
        if not fwd_bwd_map:
            logger.debug('there is no any forward backward association, skip processing backward correlation.')
            return self.tid2tree

        self._set_main_tid()

        modules, backward_nodes = self._get_modules()
        if not modules or not backward_nodes:
            return self.tid2tree

        _, ts2parent = OpTreeBuilder._get_node_parents(backward_nodes)
        agg_nodes = OpTreeBuilder._group_backward_nodes(backward_nodes)
        fwd_bwd_root = self._get_backward_roots(fwd_bwd_map, ts2parent, agg_nodes)
        if len(agg_nodes) > 0:
            logger.warning('some nodes cannot find forward nodes')

        backward_modules: List[BackwardNode] = []
        for module in modules:
            OpTreeBuilder._build_backward_module(module, None, fwd_bwd_root, backward_modules)
        OpTreeBuilder._insert_backward_modules(self.tid2tree[self.main_tid], backward_modules)
        self.tid2tree = {tid: root for tid, root in self.tid2tree.items() if len(root.children) > 0}

        return self.tid2tree

    def _build_tree(self, tid2list: Dict[int, List[OperatorNode]], tid2zero_rt_list, staled_device_nodes):
        tid2tree = {}

        for tid, op_list in tid2list.items():
            zero_rt_list = tid2zero_rt_list[tid] if tid in tid2zero_rt_list else []
            # Note that when 2 start_time are equal, the one with bigger end_time should be ahead of the other.
            op_list.sort(key=lambda x: (x.start_time, -x.end_time))
            main_tid = any([op.name.startswith('ProfilerStep#') for op in op_list])
            if main_tid:
                # only append the staled device nodes into main thread
                self.main_tid = op_list[0].tid
                root_node = self._build_tree_internal(op_list, zero_rt_list, tid, staled_device_nodes)
            else:
                root_node = self._build_tree_internal(op_list, zero_rt_list, tid, [])
            tid2tree[int(tid)] = root_node

        return tid2tree

    def _set_main_tid(self):
        if self.main_tid is None and self.tid2tree:
            if len(self.tid2tree) == 1:
                self.main_tid = next(iter(self.tid2tree))
            else:
                # there are multiple tids
                backward_tid = self._find_backward_tid()
                tid2len = {
                    tid: root.end_time - root.start_time for tid, root in self.tid2tree.items()
                    if tid != backward_tid or backward_tid is None
                }
                # get the maximum length as the main thread
                self.main_tid = max(tid2len, key=tid2len.get)

    def _find_backward_tid(self):
        for root in self.tid2tree.values():
            for child in root.children:
                if child.name.startswith(OpTreeBuilder.BACKWARD_ROOT_PREFIX):
                    return child.tid

        return None

    def _build_tree_internal(self, host_node_list, zero_rt_list, tid, staled_device_nodes):
        """host_node_list: list of OperatorNode and ProfilerStepNode.
        zero_rt_list: list of RuntimeNode with external_id=0."""

        def build_tree_relationship(host_node_list: Iterable[OperatorNode], zero_rt_list, staled_device_nodes):
            dummpy_rt: List[RuntimeNode] = []
            if staled_device_nodes:
                # Note: Although kernels of this dummy runtime is put under main thread's tree,
                # we don't know which thread launches them.
                # TODO: Don't make belonging thread assumption on future usage if we need special handling
                dummpy_rt.append(RuntimeNode(
                    name='dummy',
                    start_time=None,
                    end_time=None,
                    type=EventTypes.RUNTIME,
                    tid=0,
                    device_nodes=staled_device_nodes))
                dummpy_rt[0].fill_stats()
            node_stack: List[OperatorNode] = []
            root_node = OperatorNode(
                name='CallTreeRoot',
                start_time=-sys.maxsize - 1,
                end_time=sys.maxsize,
                type=EventTypes.PYTHON,
                tid=tid,
                runtimes=zero_rt_list + dummpy_rt)  # Give the list of RuntimeNode with external_id=0 to root node.
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
                            logger.error('Error in input data: ranges on the same thread should not intersect!'
                                         'Father:({},{},{}) Child:({},{},{})'
                                         .format(tail_node.name, tail_node.start_time, tail_node.end_time,
                                                 node.name, node.start_time, node.end_time))
                        break
                    else:
                        node_stack.pop()
            return root_node

        # Merge the consecutive calls to same function into one.
        # Just follow the same pattern in torch/autograd/profiler.py,
        # EventList._remove_dup_nodes
        # TODO: Replace recursive by for loop, in case of too deep callstack.
        def remove_dup_nodes(node: OperatorNode):
            if node.type == EventTypes.RUNTIME:
                return
            if len(node.children) == 1:
                child = node.children[0]
                if node.name == child.name and node.type == EventTypes.OPERATOR and child.type == EventTypes.OPERATOR:
                    node.children = child.children
                    node.runtimes = child.runtimes  # Keep consistent with autograd profiler.
                    remove_dup_nodes(node)  # This node may have to merge with child's child.
                    return

            for child in node.children:
                remove_dup_nodes(child)

        root_node = build_tree_relationship(host_node_list, zero_rt_list, staled_device_nodes)
        remove_dup_nodes(root_node)
        root_node.fill_stats()

        # replace the root_node start_time/end_time
        root_node.start_time = next((child.start_time for child in root_node.children
                                     if child.start_time is not None), None)
        root_node.end_time = next((child.end_time for child in reversed(root_node.children)
                                   if child.end_time is not None), None)
        return root_node

    def _get_modules(self) -> Tuple[List[ModuleNode], List[OperatorNode]]:
        """Get the ModuleNodes and backward root nodes
        If there are any ModuleNodes, the backward roots will be removed from the tree
        so that later a new BackwardNode will be replaced.
        """
        modules: List[ModuleNode] = []
        backward_nodes: Dict[OperatorNode, List[OperatorNode]] = defaultdict(list)

        def traverse_node(parent, node: OperatorNode):
            if isinstance(node, ModuleNode):
                modules.append(node)
            elif isinstance(node, ProfilerStepNode):
                for child in node.children:
                    traverse_node(node, child)
            else:
                if node.name.startswith(OpTreeBuilder.BACKWARD_ROOT_PREFIX):
                    backward_nodes[parent].append(node)
                else:
                    pass

        for root in self.tid2tree.values():
            for child in root.children:
                traverse_node(root, child)

        if modules:
            backward_nodes_flatten: List[OperatorNode] = []
            # only remove the backward nodes when the module information exist
            for p, nodes in backward_nodes.items():
                p.children = [child for child in p.children if child not in nodes]
                backward_nodes_flatten.extend(nodes)

            return modules, backward_nodes_flatten
        else:
            return None, None

    @staticmethod
    def _get_node_parents(nodes: Iterable[OperatorNode]):
        """Get the child->parent relationship for these nodes"""
        ts_to_node: Dict[int, OperatorNode] = {}
        ts_to_parent: Dict[int, OperatorNode] = {}

        def traverse_node(node: OperatorNode):
            if node.start_time not in ts_to_node:
                ts_to_node[node.start_time] = node
            for child in node.children:
                if child.start_time not in ts_to_parent:
                    ts_to_parent[child.start_time] = node
                traverse_node(child)

        for node in nodes:
            traverse_node(node)
        return ts_to_node, ts_to_parent

    @staticmethod
    def _group_backward_nodes(nodes: Iterable[OperatorNode]) -> Dict[OperatorNode, List[OperatorNode]]:
        """All nodes are backward nodes startswith autograd::engine::evaluate_function.
        If one node's name is autograd::engine::evaluate_function: torch::autograd::AccumulateGrad,
        it should be grouped with previous normal backward node. Otherwise, a new backward node should be started
        """
        grouped_bwd_nodes: List[List[OperatorNode]] = []
        for node in nodes:
            if node.name == OpTreeBuilder.BACKWARD_ACCUMULATE_GRAD:
                grouped_bwd_nodes[-1].append(node)
            else:
                grouped_bwd_nodes.append([node])

        # return the root backward node -> aggregated backward nodes array
        # if there is no any AccumulateGrad accompanied with it, then the key:value is itself.
        return {nodes[0]: nodes for nodes in grouped_bwd_nodes}

    @staticmethod
    def _get_backward_roots(fwd_bwd_map: Dict[int, int],
                            ts2parent: Dict[int, OperatorNode],
                            backward_nodes: Dict[OperatorNode, List[OperatorNode]]) -> Dict[int, List[OperatorNode]]:
        if not fwd_bwd_map:
            return None

        fwd_to_bwdroot: Dict[int, List[OperatorNode]] = {}
        for fwd, bwd in fwd_bwd_map.items():
            parent = ts2parent.get(bwd)
            while parent is not None and not parent.name.startswith(OpTreeBuilder.BACKWARD_ROOT_PREFIX):
                parent = ts2parent.get(parent.start_time)

            if parent:
                fwd_to_bwdroot[fwd] = backward_nodes.pop(parent)
            else:
                logger.warning('parent is None for', bwd)

        return fwd_to_bwdroot

    def _build_backward_module(node: ModuleNode,
                               parent: Optional[BackwardNode],
                               fwd_bwd_map: Dict[int, List[OperatorNode]],
                               result: List[BackwardNode]):
        """Construct the backward module from root (node argument) and
        insert it into result array if there is no any parent associated with it.
        """
        if not fwd_bwd_map:
            logger.warning('The forward backward map is empty. The backward construction is skipped.')
            return

        if isinstance(node, ModuleNode):
            backward_node = BackwardNode(name=node.name + '.backward', start_time=node.start_time,
                                         end_time=node.end_time, type='backward', tid=node.tid)
            if parent is None:
                result.append(backward_node)
            else:
                parent.children.append(backward_node)
            parent = backward_node

        for child in node.children:
            if parent:
                if is_operator_node(child):
                    bwd_ops = fwd_bwd_map.get(child.start_time)
                    if bwd_ops:
                        parent.children.extend(bwd_ops)

            OpTreeBuilder._build_backward_module(child, parent, fwd_bwd_map, result)

        if isinstance(node, ModuleNode) and parent and parent.children:
            parent.fill_stats()
            parent.tid = parent.children[0].tid

    @staticmethod
    def _insert_backward_modules(root: OperatorNode, backward_modules: List[BackwardNode]):
        backward_modules.sort(key=lambda x: (x.start_time, -x.end_time))

        # each item is (parent_node, child_index) that it is visiting.
        node_stack = []
        module_index = 0
        child_index = 0
        current_node = root

        staled_modules = []

        while module_index < len(backward_modules):
            module = backward_modules[module_index]
            if current_node is None:
                # ignore all remaining modules
                staled_modules.append(module)
                module_index += 1
                continue

            if module.end_time < current_node.start_time:
                staled_modules.append(module)
                module_index += 1
                continue
            elif module.start_time > current_node.end_time:
                if node_stack:
                    # pop parent node and update the child_index accordingly.
                    current_node, child_index = node_stack.pop()
                    child_index += 1
                else:
                    # if there is not item in stack, set it to None
                    current_node = None
                continue

            while child_index < len(current_node.children):
                if module.end_time < current_node.children[child_index].start_time:
                    # if current module is before next child,
                    # we will break the search and keep the child_index not change.
                    # As the result, the module will be treated as child of 'current_node'
                    # So that next time we can continue from here.
                    # there is no any child contains the record.timestamp
                    # child_find is False at this case.
                    break
                elif module.start_time >= current_node.children[child_index].end_time:
                    child_index += 1
                else:
                    # current children contains the record
                    node_stack.append((current_node, child_index))
                    current_node = current_node.children[child_index]
                    child_index = 0

            # when code execute here, it means the current_node will be the parent of backward module
            # Add the module into current_node
            current_node.children.insert(child_index, module)
            # since the children number is increased by 1, we need increment the child_index.
            child_index += 1
            module_index += 1
