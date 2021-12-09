# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from collections import namedtuple
from typing import List

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
        return f"{self.name}: {self.duration}/{self.device_duration}/{len(self.op_aggs)}"


class DiffStats:
    def __init__(self, left: OpStats, right: OpStats):
        self.left = left
        self.right = right
        self.children: List[DiffStats] = []
