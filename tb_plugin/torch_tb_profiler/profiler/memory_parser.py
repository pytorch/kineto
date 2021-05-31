# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import os
from collections import defaultdict

from .. import utils
from .node import MemoryMetrics, is_operator_node
from .trace import DeviceType, EventTypes

logger = utils.get_logger()

BENCHMARK_MEMORY = os.getenv('TORCH_PROFILER_BENCHMARK_MEMORY')
if BENCHMARK_MEMORY is not None and BENCHMARK_MEMORY.upper() in ("1", "TRUE", "ON"):
    BENCHMARK_MEMORY = True
else:
    BENCHMARK_MEMORY = False

def benchmark(func):
    def wrapper(*args, **kwargs):
        if BENCHMARK_MEMORY:
            import time
            start = time.time_ns()
            ret = func(*args, **kwargs)
            end = time.time_ns()
            logger.info("{}: {} takes: {} seconds".format(os.getpid(), func.__name__, (end - start) / 1000000000))
            return ret
        else:
            return func(*args, **kwargs)

    return wrapper

class MemoryRecord:
    def __init__(self, scope, pid, tid, ts, device_type, device_id, bytes):
        self.scope = scope
        self.tid = tid
        self.pid = pid
        self.ts = ts
        self.device_type = device_type
        self.device_id = device_id
        self.bytes = bytes

    @property
    def device_name(self):
        if self.device_type == DeviceType.CPU:
            return "CPU"
        elif self.device_type == DeviceType.CUDA:
            return "GPU{}".format(self.device_id)
        else:
            return None


class MemoryParser:
    def __init__(self, tid2tree, op_list):
        self.tid2tree = tid2tree
        self.op_list = op_list

        self.records_by_tid = defaultdict(list)

        # statistics purpose
        self.staled_records = []
        self.processed_records = []

        # the visited node times from parent to child
        # troubleshooting issue purpose.
        self.processed_node = defaultdict(int)
        self.unreached_node = defaultdict(list)

        # normal search
        self.staled_records_normal = []
        self.processed_records_normal = []

        # for troubleshooting issues.
        self.processed_node_normal = set()
        self.unreached_node_normal = defaultdict(list)

    def parse_events(self, events):
        for event in events:
            if event.type == EventTypes.MEMORY:
                record = MemoryRecord(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id, event.bytes)
                self.records_by_tid[record.tid].append(record)

        for val in self.records_by_tid.values():
            val.sort(key=lambda x: x.ts)

        if BENCHMARK_MEMORY:
            self.update_node_recursive()
            self.update_node()
        else:
            self.update_node()

    @benchmark
    def get_memory_statistics(self):
        SELF_METRICS_COUNT = MemoryMetrics.IncreaseSize

        def dict_factory():
            return defaultdict(lambda: [0] * MemoryMetrics.Total)

        # two level keys dictionary
        # first keyed by node, then keyed by device (CPU/GPU0/GPU1/etc.)
        memory_metrics_keyed_by_node = defaultdict(dict_factory)

        def traverse_node_memory(node):
            if BENCHMARK_MEMORY:
                if node not in self.processed_node_normal:
                    self.unreached_node_normal[tid].append(node)

            if node not in self.processed_node:
                self.unreached_node[tid].append(node)
                # since the node has not been visited for insert memory records, just ignore all childrens
                return
            elif is_operator_node(node):
                node_memory_metrics = node.get_memory_metrics()
                for device, metrics in node_memory_metrics.items():
                    # device is name of device like: CPU/GPU0
                    # metrics is an arrary [SelfIncreaseSize, SelfAllocationSize, SelfAllocationCount]
                    for i, value in enumerate(metrics):
                        memory_metrics_keyed_by_node[node][device][i] = value
                        memory_metrics_keyed_by_node[node][device][i + SELF_METRICS_COUNT] += value
            else:
                logger.debug("node {}:{} is not operator node, will skip its self metrics processing".format(node.name, node.start_time))

            # recursive the children nodes
            for child in node.children:
                traverse_node_memory(child)
                # sum up the child metrics
                for device, metrics in memory_metrics_keyed_by_node[child].items():
                    for i in range(SELF_METRICS_COUNT, MemoryMetrics.Total):
                        memory_metrics_keyed_by_node[node][device][i] += metrics[i]

        for tid, root in self.tid2tree.items():
            for child in root.children:
                traverse_node_memory(child)

        # keyed first by device name like CPU/GPU0 etc, then keyed by operator name.
        # the value is array [items indexed by MemoryMetrics]
        memory_metrics_keyed_by_nodename = defaultdict(dict_factory)
        # node: the instance, device_keyed_metrics: dictionary keyed by device name like CPU/GPU0
        for node, device_keyed_metrics in memory_metrics_keyed_by_node.items():
            if not is_operator_node(node):
                # skip the node like Optimizer.step, DataLoader, ProfilerStep#1 etc.
                continue

            for device, metrics in device_keyed_metrics.items():
                for i, metric in enumerate(metrics):
                    memory_metrics_keyed_by_nodename[device][node.name][i] += metric

        # get the op_calls dictionary from module parser result.
        op_calls = defaultdict(int)
        for op in self.op_list:
            op_calls[op.name] += op.calls

        result = defaultdict(defaultdict)
        for device, node_metrics in memory_metrics_keyed_by_nodename.items():
            for node, values in node_metrics.items():
                if any(values):
                    result[device][node] = values + [op_calls[node]]

        if BENCHMARK_MEMORY:
            for tid, nodes in self.unreached_node.items():
                if nodes:
                    logger.info("LOOP: tid-{}: total {} node doesn't get reached.".format(tid, len(nodes)))
                else:
                    logger.info("LOOP: tid-{}: all nodes are covered".format(tid))
            for tid, nodes in self.unreached_node_normal.items():
                if nodes:
                    logger.info("RECURSIVE: tid-{}: total {} node doesn't get reached.".format(tid, len(nodes)))
                else:
                    logger.info("RECURSIVE: tid-{}: all nodes are covered".format(tid))

                # for node in nodes:
                #     logger.debug("node {},{}:{} doesn't reached".format(node.tid, node.name, node.start_time))

            for node, times in self.processed_node.items():
                assert times == 1
                # if times > 1:
                #     logger.info("node {} is processed {} times".format(node.start_time, times))

        return result

    @property
    def record_length(self):
        return sum(len(v) for v in self.records_by_tid.values())

    @benchmark
    def update_node(self):
        tree_height = 0
        for tid, records in self.records_by_tid.items():
            if not records:
                continue

            # each item is (parent_node, child_index) that it is visiting.
            node_stack = []

            record_index = 0
            current_node = self.tid2tree.get(tid)
            child_index = 0

            if current_node:
                self.processed_node[current_node] += 1

            while record_index < len(records):
                '''In the loop, one pass will process one record. The basic logic is:
                It will search from the node that last visited since both the records and tree is ordered already
                1. it current node contains the records, then find the exactly child which just embrace it.
                2. otherwise, find the parent node and set the child_index, so that the parent node could continue from previous visited node.
                3. if there is not any node contains the records, then all remaining records will be ignored.
                '''
                record = records[record_index]

                if len(node_stack) > tree_height:
                    tree_height = len(node_stack)

                if current_node is None:
                    # 3. Ignore all remaining records.
                    logger.debug("could not find the node for tid %d, timestamp: %d, record index: %d, total records: %d" % (record.tid, record.ts, record_index, len(records)))
                    self.staled_records.append(records[record_index])
                    record_index += 1
                    continue

                if record.ts < current_node.start_time:
                    # this should only happens for root node.
                    logger.debug("record timestamp %d is less that the start time of %s" % (record.ts, current_node.name))
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
                        # untile find one contains the records
                        child_index += 1
                    else:
                        # current children contains the record
                        self.processed_node[current_node.children[child_index]] += 1

                        # push child index which will be visited, then continue the loop
                        node_stack.append((current_node, child_index))
                        current_node = current_node.children[child_index]
                        child_index = 0

                # the current_node is the one contains the record at this moment.
                if is_operator_node(current_node):
                    if not BENCHMARK_MEMORY or record not in current_node.memory_records:
                        current_node.add_memory_record(record)
                    self.processed_records.append(record)
                else:
                    self.staled_records.append(record)

                # the record is processed done, increment the index to process next one.
                record_index += 1

        # show summary information
        if len(self.staled_records) > 0 and self.record_length > 0:
            logger.debug("{} memory records are skipped in total {} memory records and only {} get processed".format(len(self.staled_records), self.record_length, len(self.processed_records)))
        if tree_height > 0:
            logger.debug("max tree height is {}".format(tree_height))

    @benchmark
    def update_node_recursive(self):
        def _update_memory_event(record, node):
            if BENCHMARK_MEMORY:
                self.processed_node_normal.add(node)

            child_found = None
            for child in node.children:
                if record.ts >= child.start_time and record.ts < child.end_time:
                    child_found = child
                    break
            if child_found is None:
                # We use left close and right open deliberately here [start time, end time)
                # to avoid one memory be calculated twice in case of it is equal to previous operator's end time
                # and next operator's start time.
                # the result might be different with PyTorch one.
                # https://github.com/pytorch/pytorch/blob/26c1f0f72e71c096648a16993484234399da307c/torch/autograd/profiler.py#L1147-L1152
                if is_operator_node(node) and record.ts >= node.start_time and record.ts < node.end_time:
                    if not BENCHMARK_MEMORY or record not in node.memory_records:
                        node.add_memory_record(record)
                    self.processed_records_normal.append(record)
                else:
                    self.staled_records_normal.append(record)
            else:
                _update_memory_event(record, child_found)

        for tid, records in self.records_by_tid.items():
            root_node = self.tid2tree.get(tid)
            if root_node is None:
                logger.warning("could not find the root node for tid %d " % tid)
                self.staled_records_normal.extend(records)

            for mem_record in records:
                _update_memory_event(mem_record, root_node)

        if len(self.staled_records_normal) > 0 and self.record_length > 0:
            logger.info("{} memory records are skipped in total {} memory records and only {} get processed".format(len(self.staled_records_normal), self.record_length, len(self.processed_records_normal)))
