from abc import ABCMeta
from typing import List, Union

from ..node import OperatorNode


class Operator(metaclass=ABCMeta):
    def __init__(self, name, start, end) -> None:
        self.name = name
        self.start_time = start
        self.end_time = end

    def __str__(self) -> str:
        return f"{self.name}"


class BlankOp(Operator):
    def __init__(self) -> None:
        super().__init__('blank', None, None)


class UnknownOp(Operator):
    def __init__(self, start, end) -> None:
        super().__init__('unknown', start, end)


class Operators(Operator):
    def __init__(self, nodes: Union[OperatorNode, List[OperatorNode]]):
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


def create_operator(op_nodes: Union[OperatorNode, List[OperatorNode]]) -> Operator:
    if op_nodes:
        return Operators(op_nodes)
    else:
        return BlankOp()
