# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Iterable, List, Optional, Tuple

from .. import utils
from .node import OperatorNode, is_operator_node
from .op_agg import aggregate_ops
from .trace import DeviceType, MemoryEvent

logger = utils.get_logger()


class MemoryMetrics(IntEnum):
    SelfIncreaseSize = 0
    SelfAllocationSize = 1
    SelfAllocationCount = 2
    IncreaseSize = 3
    AllocationSize = 4
    AllocationCount = 5


class MemoryRecord:
    def __init__(self, scope: str, pid: int, tid: int, ts: int,
                 device_type: DeviceType, device_id: int,
                 address: int, bytes: int, total_allocated: float, total_reserved: float):
        self.scope = scope
        self.tid = tid
        self.pid = pid
        self.ts = ts
        self.device_type = device_type
        self.device_id = device_id
        self.addr = address
        self.bytes = bytes
        self.total_allocated = total_allocated
        self.total_reserved = total_reserved
        self.op_name: Optional[str] = None
        self.parent_op_name: Optional[str] = None

    @property
    def device_name(self):
        if self.device_type == DeviceType.CPU:
            return 'CPU'
        elif self.device_type == DeviceType.CUDA:
            return 'GPU{}'.format(self.device_id)
        else:
            return None

    @property
    def is_allocation(self):
        return self.bytes > 0

    @property
    def op_name_or_unknown(self):
        return self.op_name if self.op_name else '<unknown>'

    @classmethod
    def from_event(cls, event: MemoryEvent):
        return cls(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id,
                   event.addr, event.bytes, event.total_allocated, event.total_reserved)

    def __repr__(self) -> str:
        return f"<{'+' if self.bytes>0 else ''}{self.bytes}B, addr: {self.addr}, ts: {self.ts}>"


class MemorySnapshot:
    def __init__(self, memory_records: Iterable[MemoryRecord],
                 op_memory_table: Dict[OperatorNode, List[MemoryRecord]],
                 processed_nodes: Dict[OperatorNode, int]) -> None:
        self.memory_records = memory_records
        self.op_memory_table = op_memory_table
        # the visited node times from parent to child
        # troubleshooting issue purpose.
        self.processed_node = processed_nodes
        self.unreached_node = defaultdict(list)

    def get_peak_memory(self) -> Dict[Tuple[DeviceType, int], int]:
        peaks = defaultdict(int)
        for r in self.memory_records:
            if r.total_allocated == r.total_allocated:  # !isnan
                peaks[(r.device_type, r.device_id)] = max(peaks[(r.device_type, r.device_id)], r.total_allocated)
        return peaks

    def get_memory_statistics(self,
                              tid2tree: Dict[int, OperatorNode],
                              start_ts=None, end_ts=None) -> Dict[str, Dict[str, List[int]]]:
        metric_length = len(MemoryMetrics)
        self_metric_length = metric_length // 2

        def dict_factory():
            return defaultdict(lambda: [0] * metric_length)

        # traverse outputs
        op_list: List[OperatorNode] = []
        # two level keys dictionary
        # first keyed by node, then keyed by device (CPU/GPU0/GPU1/etc.)
        memory_metrics_keyed_by_node: Dict[OperatorNode, Dict[str, List[int]]] = defaultdict(dict_factory)

        def traverse_node_memory(node: OperatorNode):
            if start_ts is not None and node.end_time < start_ts:
                return
            if end_ts is not None and node.start_time > end_ts:
                return

            is_op = is_operator_node(node)
            if is_op:
                op_list.append(node)

            if node not in self.processed_node:
                self.unreached_node[tid].append(node)
                # since the node has not been visited for insert memory records, just ignore all childrens
                return
            elif is_op:
                node_memory_metrics = self.get_memory_metrics(node, start_ts, end_ts)
                for device, metrics in node_memory_metrics.items():
                    # device is name of device like: CPU/GPU0
                    # metrics is an arrary [SelfIncreaseSize, SelfAllocationSize, SelfAllocationCount]
                    for i, value in enumerate(metrics):
                        memory_metrics_keyed_by_node[node][device][i] = value
                        memory_metrics_keyed_by_node[node][device][i + self_metric_length] += value
            else:
                logger.debug('node {}:{} is not operator node, will skip its self metrics processing'.format(
                    node.name, node.start_time))

            # recursive the children nodes
            for child in node.children:
                traverse_node_memory(child)
                # sum up the child metrics
                for device, metrics in memory_metrics_keyed_by_node[child].items():
                    for i in range(self_metric_length, metric_length):
                        memory_metrics_keyed_by_node[node][device][i] += metrics[i]

        for tid, root in tid2tree.items():
            for child in root.children:
                traverse_node_memory(child)

        # keyed first by device name like CPU/GPU0 etc, then keyed by operator name.
        # the value is array [items indexed by MemoryMetrics]
        memory_metrics_keyed_by_nodename: Dict[str, Dict[str, List[int]]] = defaultdict(dict_factory)
        # node: the instance, device_keyed_metrics: dictionary keyed by device name like CPU/GPU0
        for node, device_keyed_metrics in memory_metrics_keyed_by_node.items():
            if not is_operator_node(node):
                # skip the node like Optimizer.step, DataLoader, ProfilerStep#1 etc.
                continue

            for device, metrics in device_keyed_metrics.items():
                for i, metric in enumerate(metrics):
                    memory_metrics_keyed_by_nodename[device][node.name][i] += metric

        # get the op_calls dictionary from module parser result.
        op_calls: Dict[str, int] = defaultdict(int)
        agg_result = aggregate_ops(op_list, [lambda op: op.name])
        for op_name, op_agg in agg_result[0].items():
            op_calls[op_name] += op_agg.calls

        result: Dict[str, Dict[str, List[int]]] = defaultdict(defaultdict)
        for device, node_metrics in memory_metrics_keyed_by_nodename.items():
            for node, values in node_metrics.items():
                if any(values):
                    result[device][node] = values + [op_calls[node]]

        return result

    def get_memory_metrics(self, op: OperatorNode, start_ts, end_ts):
        metrics_count = len([e.name for e in MemoryMetrics if e.name.startswith('Self')])
        memory_metrics: Dict[str, List[int]] = defaultdict(lambda: [0] * metrics_count)
        for record in self.op_memory_table[op]:
            if start_ts is not None and record.ts < start_ts:
                continue
            if end_ts is not None and record.ts > end_ts:
                continue
            name = record.device_name
            if name is None:
                continue

            memory_metrics[name][MemoryMetrics.SelfIncreaseSize] += record.bytes
            if record.bytes > 0:
                memory_metrics[name][MemoryMetrics.SelfAllocationSize] += record.bytes
                memory_metrics[name][MemoryMetrics.SelfAllocationCount] += 1

        return memory_metrics


class MemoryParser:
    def __init__(self, memory_events: Iterable[MemoryEvent]):
        # statistics purpose
        self.staled_records: List[MemoryRecord] = []
        self.processed_records: List[MemoryRecord] = []
        self.memory_records: List[MemoryRecord] = [MemoryRecord.from_event(e) for e in memory_events]

    def find_memory_nodes(self, tid2tree: Dict[int, OperatorNode]) -> MemorySnapshot:
        records_by_tid: Dict[int, List[MemoryRecord]] = defaultdict(list)
        for r in self.memory_records:
            records_by_tid[r.tid].append(r)

        op_memory_table: Dict[OperatorNode, List[MemoryRecord]] = defaultdict(list)
        processed_node = defaultdict(int)

        tree_height = 0
        for tid, records in records_by_tid.items():
            if not records:
                continue

            # each item is (parent_node, child_index) that it is visiting.
            node_stack: List[Tuple[OperatorNode, int]] = []

            record_index = 0
            current_node: OperatorNode = tid2tree.get(tid)
            child_index = 0

            if current_node:
                processed_node[current_node] += 1

            while record_index < len(records):
                """In the loop, one pass will process one record. The basic logic is:
                It will search from the node that last visited since both the records and tree is ordered already
                1. it current node contains the records, then find the exactly child which just embrace it.
                2. otherwise, find the parent node and set the child_index, so that the parent node could continue from previous visited node. # noqa: E501
                3. if there is not any node contains the records, then all remaining records will be ignored.
                """
                record = records[record_index]

                if len(node_stack) > tree_height:
                    tree_height = len(node_stack)

                if current_node is None or current_node.start_time is None or current_node.end_time is None:
                    # 3. Ignore all remaining records.
                    logger.debug(
                        'could not find the node for tid %d, timestamp: %d, record index: %d, total records: %d' % (
                            record.tid, record.ts, record_index, len(records)))
                    self.staled_records.append(records[record_index])
                    record_index += 1
                    continue

                if record.ts < current_node.start_time:
                    # this should only happens for root node.
                    logger.debug('record timestamp %d is less that the start time of %s' %
                                 (record.ts, current_node.name))
                    # This record has no chance to be appended to following tree node.
                    self.staled_records.append(record)
                    record_index += 1
                    continue
                elif record.ts >= current_node.end_time:
                    # 2. pop parent node and update the child_index accordingly.
                    if len(node_stack) > 0:
                        current_node, child_index = node_stack.pop()
                        child_index += 1
                    else:
                        # if there is not item in stack, set it to None
                        current_node = None
                    continue

                # 1. find the real node embrace the record.
                # Find the node which contains the records from top to downmost.
                while child_index < len(current_node.children):
                    if record.ts < current_node.children[child_index].start_time:
                        # if current record timestamp is less than the current child's startime,
                        # we will break the search and keep the child_index not change. So that next time
                        # we can continue from here.
                        # there is no any child contains the record.timestamp
                        # child_find is False at this case.
                        break
                    elif record.ts >= current_node.children[child_index].end_time:
                        # if the record timestamp is greater than the children end time, increment to next child
                        # until find one contains the record
                        child_index += 1
                    else:
                        # current children contains the record
                        processed_node[current_node.children[child_index]] += 1

                        # push child index which will be visited, then continue the loop
                        node_stack.append((current_node, child_index))
                        current_node = current_node.children[child_index]
                        child_index = 0

                # the current_node is the one contains the record at this moment.
                if is_operator_node(current_node):
                    op_memory_table[current_node].append(record)
                    # NOTE: only allocation record can be associated with op. Because deallocation happens at the end
                    # of a tensor's lifetime which is not deterministic.
                    if record.is_allocation:
                        record.op_name = current_node.name
                        if len(node_stack) > 0:
                            record.parent_op_name = node_stack[-1][0].name
                    self.processed_records.append(record)
                else:
                    self.staled_records.append(record)

                # the record is processed
                record_index += 1

        # show summary information
        if len(self.staled_records) > 0 and len(self.memory_records) > 0:
            logger.debug('{} memory records are skipped in total {} memory records and only {} get processed'.format(
                len(self.staled_records), len(self.memory_records), len(self.processed_records)))
        if tree_height > 0:
            logger.debug('max tree height is {}'.format(tree_height))

        all_records = self.get_preprocessed_records()
        return MemorySnapshot(all_records, op_memory_table, processed_node)

    def get_preprocessed_records(self):
        memory_records = sorted(self.memory_records, key=lambda r: r.ts)

        alloc = {}  # allocation events may or may not have paired free event
        prev_ts = float('-inf')  # ensure ordered memory records is ordered
        for i, r in enumerate(memory_records):
            if r.addr is None:
                # profile json data prior to pytorch 1.10 do not have addr
                # we should ignore them
                continue
            assert prev_ts <= r.ts
            prev_ts = r.ts
            addr = r.addr
            size = r.bytes
            if size > 0:
                # Allocation event, to be matched with a Release event
                alloc[addr] = i
            else:
                # Processing a Release event
                if addr in alloc:
                    alloc_r = memory_records[alloc[addr]]
                    r.op_name = alloc_r.op_name
                    r.parent_op_name = alloc_r.parent_op_name
                    del alloc[addr]
        return memory_records
