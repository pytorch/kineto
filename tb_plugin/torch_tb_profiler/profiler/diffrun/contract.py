# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from collections import namedtuple
from typing import Dict, List

OpAgg = namedtuple('OpAgg', [
    'name',
    'calls',
    'host_duration',
    'device_duration',
    'self_host_duration',
    'self_device_duration'])


class OpStats:
    def __init__(self,
                 name,
                 duration,
                 device_duration,
                 total_duration,
                 aggs: List[OpAgg]):
        self.name = name
        self.duration = duration
        self.device_duration = device_duration
        self.total_duration = total_duration
        self.op_aggs = aggs

    def __str__(self) -> str:
        return f'{self.name}: {self.duration}/{self.device_duration}/{len(self.op_aggs)}'


class DiffStats:
    def __init__(self, left: OpStats, right: OpStats):
        self.left = left
        self.right = right
        self.children: List[DiffStats] = []

    def flatten_diff_tree(self) -> Dict[str, 'DiffStats']:
        result: Dict[str, DiffStats] = {}

        def traverse(node: DiffStats, path: str):
            result[path] = node
            for i, child in enumerate(node.children):
                traverse(child, f'{path}-{i}')

        traverse(self, '0')
        return result

    def to_dict(self):
        d = {
            'left': {
                'name': self.left.name,
                'duration': self.left.duration,
                'device_duration': self.left.device_duration,
                'total_duration': self.left.total_duration,
                'aggs': []
            },
            'right': {
                'name': self.right.name,
                'duration': self.right.duration,
                'device_duration': self.right.device_duration,
                'total_duration': self.right.total_duration,
                'aggs': []
            }
        }

        for agg in self.left.op_aggs:
            d['left']['aggs'].append(agg._asdict())

        for agg in self.right.op_aggs:
            d['right']['aggs'].append(agg._asdict())

        return d

    def get_diff_tree_summary(self):
        def traverse_node_recursive(node: DiffStats):
            d = node.to_dict()

            d['children'] = []
            for c in node.children:
                d['children'].append(traverse_node_recursive(c))

            return d

        return traverse_node_recursive(self)

    def get_diff_node_summary(self, path: str):
        def traverse_node(node: DiffStats, path: str):
            d = node.to_dict()
            d['path'] = path
            return d

        d = traverse_node(self, path)
        d['children'] = []
        for i, c in enumerate(self.children):
            d['children'].append(traverse_node(c, f'{path}-{i}'))

        return d
