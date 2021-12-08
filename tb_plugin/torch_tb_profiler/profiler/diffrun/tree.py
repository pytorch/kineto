import sys
from typing import List, Union

from ..node import (BackwardNode, DataLoaderNode, ModuleNode, OperatorNode,
                    OptimizerNode, ProfilerStepNode)
from .operator import Operator, Operators, create_operator

INDENT = "    "
RUN_NODE_TYPES = (BackwardNode, DataLoaderNode, ModuleNode, OptimizerNode, ProfilerStepNode)


class DiffNode:
    def __init__(self, left: Operator, right: Operator):
        self.left = left
        self.right = right
        self.children = []

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
    def create_node(left: Union[OperatorNode, List[OperatorNode]],
                    right: Union[OperatorNode, List[OperatorNode]]) -> 'DiffNode':
        '''Create the diff tree from two root node
        TODO: need handle the different threads case
        TODO: need add runtimes besides of children?
        '''
        node = DiffNode(create_operator(left), create_operator(right))
        node.build_tree()
        return node

    @staticmethod
    def compare_operator_nodes(left_nodes: List[OperatorNode], right_nodes: List[OperatorNode]) -> List['DiffNode']:
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

        # split the two list by the matching points

        # construct the result
        if not matched_paris:
            # there is not any matching points.
            return

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
            # TODO: fill unknown nodes in case of the starttime of next node and current
            # end time is bigger than threshold.
            # Or do we need move the logic into frondend for visualization?

        # process the remaining nodes
        left_remaining = left_nodes[l_iter:]
        right_remaining = right_nodes[r_iter:]
        if left_remaining or right_remaining:
            yield DiffNode.create_node(left_remaining, right_remaining)


def create_diff_tree(left: OperatorNode, right: OperatorNode) -> DiffNode:
    '''Create the diff tree from two root node
       TODO: need handle the different threads case
       TODO: need add runtimes besides of children?
    '''
    return DiffNode.create_node(left.children, right.children)


def print_node(node: DiffNode, level: int, index: int, file=sys.stdout):
    file.write(f"{INDENT * level}level {level}, index {index}:\n")
    file.write(f"{INDENT * (level + 1)}left : {node.left}\n")
    file.write(f"{INDENT * (level + 1)}right: {node.right}\n")
    for i, child in enumerate(node.children):
        print_node(child, level + 1, i, file=file)
