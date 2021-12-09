from abc import ABCMeta
from typing import List, Union

from ..node import OperatorNode


class Operator(metaclass=ABCMeta):
    def __init__(self, name) -> None:
        self.name: str = name

    def __str__(self) -> str:
        return f"{self.name}: {self.duration}"

    @property
    def duration(self):
        return 0

    @property
    def device_duration(self):
        return 0

    @property
    def total_duration(self):
        return self.device_duration or self.duration


class BlankOp(Operator):
    def __init__(self) -> None:
        super().__init__('blank')


class UnknownOp(Operator):
    def __init__(self, device_duration, duration) -> None:
        super().__init__('unknown')
        self.device_duration = device_duration
        self.duration = duration

    @property
    def duration(self):
        return self.duration

    @property
    def device_duration(self):
        return self.device_duration


class Operators(Operator):
    def __init__(self, nodes: Union[OperatorNode, List[OperatorNode]]):
        if nodes is None:
            raise ValueError("the operator node is None")
        if isinstance(nodes, OperatorNode):
            super().__init__(nodes.name)
        elif isinstance(nodes, list):
            super().__init__("multiple nodes")

        self.op_nodes: Union[OperatorNode, List[OperatorNode]] = nodes

    @property
    def duration(self):
        if isinstance(self.op_nodes, list):
            return sum(n.duration for n in self.op_nodes)
        else:
            return self.op_nodes.duration

    @property
    def device_duration(self):
        if isinstance(self.op_nodes, list):
            return sum(n.device_duration for n in self.op_nodes)
        else:
            return self.op_nodes.device_duration

    @property
    def total_duration(self):
        if isinstance(self.op_nodes, list):
            d = 0
            for n in self.op_nodes:
                if n.device_duration > 0:
                    d += n.device_duration
                else:
                    d += n.duration
            return d
        else:
            return self.op_nodes.device_duration or self.op_nodes.duration

    def __str__(self) -> str:
        if isinstance(self.op_nodes, list):
            return f"{self.name}: {len(self.op_nodes)}: {self.op_nodes[0].name}: {self.total_duration}"
        else:
            return f"{self.name}: {self.op_nodes.__class__.__name__}: {self.total_duration}"


def create_operator(op_nodes: Union[OperatorNode, List[OperatorNode]]) -> Operator:
    if op_nodes:
        return Operators(op_nodes)
    else:
        return BlankOp()
