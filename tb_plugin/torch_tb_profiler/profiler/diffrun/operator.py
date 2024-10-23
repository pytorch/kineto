# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from abc import ABCMeta
from typing import List, Tuple, Union

from ..node import DeviceNode, OperatorNode
from ..op_agg import aggregate_ops
from .contract import OpAgg


class Operator(metaclass=ABCMeta):
    def __init__(self, name) -> None:
        self.name: str = name

    def __str__(self) -> str:
        return f'{self.name}: {self.duration}'

    @property
    def duration(self) -> int:
        return 0

    @property
    def device_duration(self) -> int:
        return 0

    @property
    def total_duration(self):
        return self.device_duration or self.duration

    def aggregate_ops(self):
        ops, _ = self.get_operators_and_kernels()
        agg_result = aggregate_ops(ops, [lambda x: x.name])[0]
        for agg in agg_result.values():
            yield OpAgg(
                agg.name,
                agg.calls,
                agg.host_duration,
                agg.device_duration,
                agg.self_host_duration,
                agg.self_device_duration)

    def get_operators_and_kernels(self) -> Tuple[List[OperatorNode], List[DeviceNode]]:
        return [], []


class BlankOp(Operator):
    def __init__(self) -> None:
        super().__init__('Blank')


class UnknownOp(Operator):
    def __init__(self, device_duration: int, duration: int) -> None:
        super().__init__('Unknown')
        self.device_duration = device_duration
        self.duration = duration

    @property
    def duration(self) -> int:
        return self.duration

    @property
    def device_duration(self) -> int:
        return self.device_duration


class Operators(Operator):
    def __init__(self, nodes: Union[OperatorNode, List[OperatorNode]]):
        if not nodes:
            raise ValueError('the operator node is None or empty')
        if isinstance(nodes, OperatorNode):
            super().__init__(nodes.name)
        elif isinstance(nodes, list):
            super().__init__('CompositeNodes')

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
            return sum(n.device_duration or n.duration for n in self.op_nodes)
        else:
            return self.op_nodes.device_duration or self.op_nodes.duration

    def __str__(self) -> str:
        if isinstance(self.op_nodes, list):
            return f'{self.name}: {len(self.op_nodes)}: {self.op_nodes[0].name}: {self.total_duration}'
        else:
            return f'{self.name}: {self.op_nodes.__class__.__name__}: {self.total_duration}'

    def get_operators_and_kernels(self) -> Tuple[List[OperatorNode], List[DeviceNode]]:
        if isinstance(self.op_nodes, list):
            nodes = self.op_nodes
        else:
            nodes = [self.op_nodes]

        ops: List[OperatorNode] = []
        kernels: List[DeviceNode] = []
        for n in nodes:
            o, k = n.get_operator_and_kernels()
            ops.extend(o)
            kernels.extend(k)
        return ops, kernels


def create_operator(op_nodes: Union[OperatorNode, List[OperatorNode]]) -> Operator:
    if op_nodes:
        return Operators(op_nodes)
    else:
        return BlankOp()
