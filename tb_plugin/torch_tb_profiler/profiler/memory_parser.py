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
    def __init__(self, tid2tree):
        self.tid2tree = tid2tree

        self.records_by_tid = defaultdict(list)

        # statistics purpose
        self.staled_records = []
        self.processed_records = []

    def parse_events(self, events):
        for event in events:
            if event.type == EventTypes.MEMORY:
                record = MemoryRecord(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id, event.bytes)
                self.records_by_tid[record.tid].append(record)

        for val in self.records_by_tid.values():
            val.sort(key=lambda x: x.ts)

        if BENCHMARK_MEMORY:
            import time
            start = time.time_ns()
            self.update_node()
            end = time.time_ns()
            logger.info("{}: update node takes: {}".format(os.getpid(), (end - start) / 1000000000))

            start = time.time_ns()
            self.update_node_loop()
            end = time.time_ns()
            logger.info("{}: update node using loops takes: {}".format(os.getpid(), (end - start) / 1000000000))
        else:
            self.update_node_loop()

    def get_memory_statistics(self):
        if not BENCHMARK_MEMORY:
            return self.get_memory_statistics_internal()
        else:
            import time
            start = time.time_ns()
            result = self.get_memory_statistics_internal()
            end = time.time_ns()
            logger.info("{} get_memory_statistics takes {}".format(os.getpid(), (end - start) / 1000000000))
            return result

    def get_memory_statistics_internal(self):
        SELF_METRICS_COUNT = MemoryMetrics.IncreaseSize

        def dict_factory():
            return defaultdict(lambda: [0] * MemoryMetrics.Total)

        # two level keys dictionary
        # first keyed by node, then keyed by device (CPU/GPU0/GPU1/etc.)
        memory_metrics_keyed_by_node = defaultdict(dict_factory)

        op_calls = defaultdict(int)

        def traverse_node_memory(node):
            node_memory_metrics = node.get_memory_metrics()
            if is_operator_node(node):
                op_calls[node.name] += 1
                for device, metrics in node_memory_metrics.items():
                    # device is name of device like: CPU/GPU0
                    # metrics is an arrary [SelfIncreaseSize, SelfAllocationSize, SelfAllocationCount]
                    for i, value in enumerate(metrics):
                        memory_metrics_keyed_by_node[node][device][i] = value
                        memory_metrics_keyed_by_node[node][device][i + SELF_METRICS_COUNT] += value

            for child in node.children:
                traverse_node_memory(child)
                # sum up the child metrics
                if is_operator_node(node):
                    for device, metrics in memory_metrics_keyed_by_node[child].items():
                        for i in range(SELF_METRICS_COUNT, MemoryMetrics.Total):
                            memory_metrics_keyed_by_node[node][device][i] += metrics[i]

        for root in self.tid2tree.values():
            for child in root.children:
                traverse_node_memory(child)

        # keyed first by device name like CPU/GPU0 etc, then keyed by operator name.
        # the value is array [items indexed by MemoryMetrics] 
        memory_metrics_keyed_by_nodename = defaultdict(dict_factory) 
        # node is the instance
        # device_keyed_metrics is dictionary keyed by device name like CPU/GPU0
        for node, device_keyed_metrics in memory_metrics_keyed_by_node.items():
            for device, metrics in device_keyed_metrics.items():
                for i, metric in enumerate(metrics):
                    memory_metrics_keyed_by_nodename[device][node.name][i] += metric

        result = defaultdict(defaultdict)
        for device, node_metrics in memory_metrics_keyed_by_nodename.items():
            for node, values in node_metrics.items():
                if any(values):
                    result[device][node] = values + [op_calls[node]]

        return result

    @property
    def record_length(self):
        return sum(len(v) for v in self.records_by_tid.values())

    def update_node(self):
        for tid, records in self.records_by_tid.items():
            root_node = self.tid2tree.get(tid)
            if not BENCHMARK_MEMORY:
                if root_node is None:
                    logger.warning("could not find the root node for tid %d " % tid)
                    self.staled_records.extend(records)

            for mem_record in records:
                self._update_memory_event(mem_record, root_node)

        if not BENCHMARK_MEMORY:
            if len(self.staled_records) > 0 and self.record_length > 0:
                logger.info("{} memory records are skipped in total {} memory records and only {} get processed".format(len(self.staled_records), self.record_length, len(self.processed_records)))

    def _update_memory_event(self, record, node):
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
                if not BENCHMARK_MEMORY:
                    node.add_memory_record(record)
                    self.processed_records.append(record)
                else:
                    if record not in node.memory_records:
                        node.add_memory_record(record)
            else:
                if not BENCHMARK_MEMORY:
                    self.staled_records.append(record)
        else:
            self._update_memory_event(record, child_found)

    def update_node_loop(self):
        tree_height = 0
        for tid, records in self.records_by_tid.items():
            if not records:
                return

            # the traverse stack which key is the instance of node, value is the child index last visted.
            traverse_dict = {}
            child_index = 0

            record_index = 0
            current_node = self.tid2tree.get(tid)
            while record_index < len(records):
                record = records[record_index]

                if BENCHMARK_MEMORY and len(traverse_dict) > tree_height:
                    tree_height = len(traverse_dict)

                if current_node is None:
                    if not BENCHMARK_MEMORY:
                        logger.debug("could not find the node for tid %d, timestamp: %d, record index: %d, total records: %d" % (record.tid, record.ts, record_index, len(records)))
                        self.staled_records.append(records[record_index])
                    record_index += 1
                    continue

                if record.ts < current_node.start_time:
                    if not BENCHMARK_MEMORY:
                        logger.debug("record timestamp %d is less that the start time of %s" % (record.ts, current_node.name))
                        # This record has no chance to be appended to following tree node.
                        self.staled_records.append(record)
                    record_index += 1
                    continue
                elif record.ts >= current_node.end_time:
                    current_node = current_node.parent
                    # next child
                    if current_node is not None:
                        # pop out the index of parent children. It is current node's index and plus 1 
                        # So we can continue next node.
                        child_index = traverse_dict.pop(current_node) + 1
                    continue

                child_find = None
                child_find_index = None
                for i in range(child_index, len(current_node.children)):
                    if record.ts < current_node.children[i].start_time:
                        # if current record timestamp is less than the current child's startime,
                        # we will break the search and keep the child_index not change. So that next time 
                        # we can continue from here.
                        break

                    if record.ts >= current_node.children[i].start_time and record.ts < current_node.children[i].end_time:
                        child_find = current_node.children[i]
                        child_find_index = i
                        break

                if child_find:
                    traverse_dict[current_node] = child_find_index
                    current_node = child_find
                    child_index = 0
                    continue

                # could not find the child
                if is_operator_node(current_node):
                    if not BENCHMARK_MEMORY:
                        current_node.add_memory_record(record)
                        self.processed_records.append(record)
                    else:
                        if record not in current_node.memory_records:
                            current_node.add_memory_record(record)
                    record_index += 1
                    continue
                else:
                    if not BENCHMARK_MEMORY:
                        self.staled_records.append(record)
                    record_index += 1

        if not BENCHMARK_MEMORY:
            if len(self.staled_records) > 0 and self.record_length > 0:
                logger.debug("{} memory records are skipped in total {} memory records and only {} get processed".format(len(self.staled_records), self.record_length, len(self.processed_records)))
        else:
            logger.info("max tree size is {}".format(tree_height))
