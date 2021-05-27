from collections import defaultdict

from .. import utils
from .node import MemoryMetrics, is_operator_node
from .trace import EventTypes

logger = utils.get_logger()

class MemoryRecord:
    def __init__(self, scope, pid, tid, ts, device_type, device_id, bytes):
        self.scope = scope
        self.tid = tid
        self.pid = pid
        self.ts = ts
        self.device_type = device_type
        self.device_id = device_id
        self.bytes = bytes

class MemoryParser:
    def __init__(self, tid2tree):
        self.tid2tree = tid2tree

        self.records = []
        self.staled_records = []
        self.processed_record = []

    def parse_events(self, events):
        for event in events:
            if event.type == EventTypes.MEMORY:
                record = MemoryRecord(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id, event.bytes)
                self.records.append(record)

        self.update_node()

    def get_memory_stats(self, device_type, device_id):
        memory_metrics_keyed_by_node = defaultdict(lambda: [0] * MemoryMetrics.Total)
        memory_metrics_keyed_by_nodename = defaultdict(lambda: [0] * MemoryMetrics.Total)
        op_calls = defaultdict(int)

        def traverse_node_memory(node, device_type):
            memory_metric = node.get_memroy_metrics(device_type, device_id)
            self_metric_length =  len(memory_metric)
            if is_operator_node(node):
                op_calls[node.name] += 1
                for i, metric in enumerate(memory_metric):
                    memory_metrics_keyed_by_node[node][i] = metric # self metric
                    memory_metrics_keyed_by_node[node][i + self_metric_length] += metric # metrics include child

            for child in node.children:
                traverse_node_memory(child, device_type)
                # sum up the child metrics
                if is_operator_node(node):
                    for i in range(self_metric_length, MemoryMetrics.Total):
                        memory_metrics_keyed_by_node[node][i] += memory_metrics_keyed_by_node[child][i]

        for root in self.tid2tree.values():
            for child in root.children:
                traverse_node_memory(child, device_type)

        for node, metrics in memory_metrics_keyed_by_node.items():
            for i, metric in enumerate(metrics):
                memory_metrics_keyed_by_nodename[node.name][i] += metric
        return {k: v + [op_calls[k]] for k, v in memory_metrics_keyed_by_nodename.items() if any(v)}

    def update_node(self):
        for mem_record in self.records:
            root_node = self.tid2tree.get(mem_record.tid)
            if root_node is None:
                logger.warning("could not find the root node for tid %d " % mem_record.tid)
                self.staled_records.append(mem_record)
            else:
                self._update_memory_event(mem_record, root_node)

        if len(self.staled_records) > 0 and len(self.records) > 0:
            logger.info("{} memory records are skipped in total {} memory records and only {} get processed".format(len(self.staled_records), len(self.records), len(self.processed_record)))

    def _update_memory_event(self, record, node):
        child_found = None
        for child in node.children:
            if record.ts >= child.start_time and record.ts < child.end_time:
                child_found = child
                break
        if child_found is None:
            if is_operator_node(node) and record.ts >= node.start_time and record.ts < node.end_time:
                node.add_memory_record(record)
                self.processed_record.append(record)
            else:
                self.staled_records.append(record)
        else:
            self._update_memory_event(record, child_found)
