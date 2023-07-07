# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import sys
from typing import Generator, List, Union

from ..node import (BackwardNode, DataLoaderNode, ModuleNode, OperatorNode,
                    OptimizerNode, ProfilerStepNode)
from .contract import DiffStats, OpStats
from .operator import Operator, Operators, create_operator

INDENT = '    '
RUN_NODE_TYPES = (BackwardNode, DataLoaderNode, ModuleNode, OptimizerNode, ProfilerStepNode)


class DiffNode:
    def __init__(self, left: Operator, right: Operator):
        self.left: Operator = left
        self.right: Operator = right
        self.children: List[DiffNode] = []

    def build_tree(self):
        '''build the children from the left_node and right_node'''
        if not isinstance(self.left, Operators) or not isinstance(self.right, Operators):
            # TODO: do we need calculate the stats or not?
            return

        if isinstance(self.left.op_nodes, OperatorNode) and isinstance(self.right.op_nodes, OperatorNode):
            # simple node match.
            diff_nodes = list(DiffNode.compare_operator_nodes(
                self.left.op_nodes.children, self.right.op_nodes.children))
            if diff_nodes:
                self.children.extend(diff_nodes)
        elif isinstance(self.left.op_nodes, list) and isinstance(self.right.op_nodes, list):
            # compare two list
            diff_nodes = list(DiffNode.compare_operator_nodes(self.left.op_nodes, self.right.op_nodes))
            if diff_nodes:
                self.children.extend(diff_nodes)
        else:
            # one single item and one list
            pass
            # TODO: do we need statistic the stats for both operator and kernel here?

    @staticmethod
    def create_node(
            left: Union[OperatorNode, List[OperatorNode]],
            right: Union[OperatorNode, List[OperatorNode]]) -> 'DiffNode':
        if isinstance(left, list) and len(left) == 1:
            left = left[0]
        if isinstance(right, list) and len(right) == 1:
            right = right[0]

        node = DiffNode(create_operator(left), create_operator(right))
        node.build_tree()
        return node

    @staticmethod
    def compare_operator_nodes(
            left_nodes: List[OperatorNode],
            right_nodes: List[OperatorNode]) -> Generator['DiffNode', None, None]:
        '''Given two OperatorNode lists, find the DataLoader/Module/Backward/Optimizer node and create the child list DiffNode
        '''
        right_keys = [(type(r), r.name) for r in right_nodes]

        # find matching points in the two list
        matched_paris = []
        key_index = 0
        for i, left_node in enumerate(left_nodes):
            if not isinstance(left_node, RUN_NODE_TYPES):
                # only handle DataLoader/Module/Backward/Optimizer nodes
                continue

            for j in range(key_index, len(right_keys)):
                if right_keys[j] == (type(left_node), left_node.name):
                    matched_paris.append((i, j))
                    key_index = j + 1
                    break

        if not matched_paris:
            # there is not any matching points.
            return

        # split the two list by the matching points
        l_iter = 0
        r_iter = 0

        for (l, r) in matched_paris:
            left_child = left_nodes[l_iter:l]
            right_child = right_nodes[r_iter:r]
            if left_child or right_child:
                yield DiffNode.create_node(left_child, right_child)

            yield DiffNode.create_node(left_nodes[l], right_nodes[r])
            l_iter = l + 1
            r_iter = r + 1
            # TODO: fill unknown nodes in case of the start_time of next node and current
            # end time is bigger than threshold.
            # Or do we need move the logic into frondend for visualization?

        # process the remaining nodes
        left_remaining = left_nodes[l_iter:]
        right_remaining = right_nodes[r_iter:]
        if left_remaining or right_remaining:
            yield DiffNode.create_node(left_remaining, right_remaining)


def compare_op_tree(left: OperatorNode, right: OperatorNode) -> DiffNode:
    '''Create the diff tree from two root node
       TODO: need handle the different threads case
             need add runtimes besides of children?
    '''
    left_children = list(get_tree_operators(left))
    right_children = list(get_tree_operators(right))
    return DiffNode.create_node(left_children, right_children)


def get_tree_operators(root: OperatorNode) -> Generator[OperatorNode, None, None]:
    '''Get the operators by the root operators by excluding the ProfilerStepNode
    '''
    profiler_nodes = [c for c in root.children if isinstance(c, ProfilerStepNode)]
    if not profiler_nodes:
        # there is no ProfilerStepNode at all
        yield from root.children
    else:
        yield from (child for p in profiler_nodes for child in p.children)


def diff_summary(node: DiffNode) -> DiffStats:
    if not node:
        return None

    left = OpStats(
        node.left.name,
        node.left.duration,
        node.left.device_duration,
        node.left.total_duration,
        list(node.left.aggregate_ops()))
    right = OpStats(
        node.right.name,
        node.right.duration,
        node.right.device_duration,
        node.right.total_duration,
        list(node.right.aggregate_ops()))

    stats = DiffStats(left, right)
    for child in node.children:
        stats.children.append(diff_summary(child))

    return stats


def print_node(node: Union[DiffNode, DiffStats], level: int, index: int, file=sys.stdout):
    file.write(f'{INDENT * level}level {level}, index {index}:\n')
    file.write(f'{INDENT * (level + 1)}left : {node.left}\n')
    file.write(f'{INDENT * (level + 1)}right: {node.right}\n')
    for i, child in enumerate(node.children):
        print_node(child, level + 1, i, file=file)


def print_ops(op: Operators, prefix: str = INDENT, file=sys.stdout):
    if isinstance(op.op_nodes, list):
        for n in op.op_nodes:
            file.write(f'{prefix}{n.name}\n')
    else:
        file.write(f'{prefix}{op.op_nodes.name}\n')
