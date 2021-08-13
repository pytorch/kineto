import sys
from abc import ABCMeta
from typing import List

from ..node import (BackwardNode, DataLoaderNode, ModuleNode, OperatorNode,
                    OptimizerNode, ProfilerStepNode)

run_node_types = (OptimizerNode, DataLoaderNode, BackwardNode, ModuleNode, ProfilerStepNode)


class LogicalNode(metaclass=ABCMeta):
    def __init__(self, name, start, end) -> None:
        self.name = name
        self.start_time = start
        self.end_time = end

    def __str__(self) -> str:
        return f"{self.name}"


class BlankNode(LogicalNode):
    def __init__(self) -> None:
        super().__init__('blank', None, None)


class UnknownNode(LogicalNode):
    def __init__(self, start, end) -> None:
        super().__init__('unknown', start, end)


class OperatorRunNode(LogicalNode):
    def __init__(self, nodes) -> None:
        if nodes is None:
            raise ValueError("the operator node is None")
        if isinstance(nodes, OperatorNode):
            # TODO: do we need using device time?
            # super().__init__(nodes.name, nodes.device_start_time, nodes.device_end_time)
            super().__init__(nodes.name, nodes.start_time, nodes.end_time)
        elif isinstance(nodes, list):
            # TODO: do we need using device time?
            # super().__init__("multiple nodes", nodes[0].device_start_time, nodes[-1].device_end_time)
            super().__init__("multiple nodes", nodes[0].start_time, nodes[-1].end_time)
        self.op_nodes = nodes

    def __str__(self) -> str:
        if isinstance(self.op_nodes, list):
            return f"{self.name}: {len(self.op_nodes)}: {self.op_nodes[0].name}"
        else:
            return f"{self.name}: {self.op_nodes.__class__.__name__}"


def create_diff_node(op_nodes):
    if op_nodes:
        return OperatorRunNode(op_nodes)
    else:
        return BlankNode()


class DiffNode:
    INDENT = "    "

    def __init__(self, left: LogicalNode, right: LogicalNode, level: int = 0) -> None:
        self.level = level
        self.left_node = left
        self.right_node = right
        # The tree children. Only single OperatorNode will be traversed recursively.
        self.children: List[DiffNode] = []
        self._build_tree()

        # find two threads: ProfilerStep#, Backward, then align each backward with profiler step node

    def _build_tree(self):
        '''build the children from the left_node and right_node'''
        if not isinstance(self.left_node, OperatorRunNode) or not isinstance(self.right_node, OperatorRunNode):
            # TODO: do we need calculate the stats or not?
            return

        if isinstance(self.left_node.op_nodes, OperatorNode) and isinstance(self.right_node.op_nodes, OperatorNode):
            # simple node match.
            diff_nodes = compare_operator_nodes(
                self.left_node.op_nodes.children, self.right_node.op_nodes.children, self.level + 1)
            if diff_nodes:
                self.children.extend(diff_nodes)
        elif isinstance(self.left_node.op_nodes, list) and isinstance(self.right_node.op_nodes, list):
            # compare two list
            diff_nodes = compare_operator_nodes(self.left_node.op_nodes, self.right_node.op_nodes, self.level + 1)
            if diff_nodes:
                self.children.extend(diff_nodes)
        else:
            # one single item and one list
            pass
            # TODO: do we need statistic the stats for both operator and kernel here?

    def print(self, indent="", index=0, file=sys.stdout):
        file.write(f"{indent}level {self.level}, index {index}:\n")
        file.write(f"{indent + DiffNode.INDENT}left : {self.left_node}\n")
        file.write(f"{indent + DiffNode.INDENT}right: {self.right_node}\n")
        for i, child in enumerate(self.children):
            child.print(indent + DiffNode.INDENT, i, file=file)


def compare_operator_nodes(left_nodes: List[OperatorNode], right_nodes: List[OperatorNode], level) -> List[DiffNode]:
    '''Given two OperatorNode lists, find the DataLoader/Module/Backward/Optimizer node and create the child list DiffNode
    '''
    right_keys = [(type(r), r.name) for r in right_nodes]

    # find matching points in the two list
    matched_paris = []
    key_index = 0
    for i, left_node in enumerate(left_nodes):
        if not isinstance(left_node, run_node_types):
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
        return None

    result = []

    l_iter = 0
    r_iter = 0

    for (l, r) in matched_paris:
        left_child = left_nodes[l_iter:l]
        right_child = right_nodes[r_iter:r]
        if left_child or right_child:
            diff_child = DiffNode(create_diff_node(left_child), create_diff_node(right_child), level)
            result.append(diff_child)

        result.append(DiffNode(create_diff_node(left_nodes[l]), create_diff_node(right_nodes[r]), level))
        l_iter = l + 1
        r_iter = r + 1
        # TODO: fill unknown nodes in case of the starttime of next node and current
        # end time is bigger than threshold.
        # Or do we need move the logic into frondend for visualization?

    # process the remaining nodes
    left_remaining = left_nodes[l_iter:]
    right_remaining = right_nodes[r_iter:]
    if left_remaining or right_remaining:
        diff_child = DiffNode(create_diff_node(left_remaining), create_diff_node(right_remaining), level)
        result.append(diff_child)

    return result


def create_diff_tree(left_root: OperatorNode, right_root: OperatorNode) -> DiffNode:
    '''Create the diff tree from two root node
       TODO: need handle the different threads case
       TODO: need add runtimes besides of children?
    '''
    return DiffNode(create_diff_node(left_root.children), create_diff_node(right_root.children))
