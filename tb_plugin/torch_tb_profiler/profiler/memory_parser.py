from collections import defaultdict

from .. import utils
from .node import MemoryMetrics, is_operator_node
from .trace import DeviceType, EventTypes

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

        self.records = []
        self.staled_records = []
        self.processed_record = []

    def parse_events(self, events):
        for event in events:
            if event.type == EventTypes.MEMORY:
                record = MemoryRecord(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id, event.bytes)
                self.records.append(record)

        self.records.sort(key=lambda x: x.ts)
        self.update_node()

    def get_memory_statistics(self):
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
            # We use left close and right open deliberately here [start time, end time) 
            # to avoid one memory be calculated twice in case of it is equal to previous operator's end time
            # and next operator's start time. 
            # the result might be different with PyTorch one.
            # https://github.com/pytorch/pytorch/blob/26c1f0f72e71c096648a16993484234399da307c/torch/autograd/profiler.py#L1147-L1152
            if is_operator_node(node) and record.ts >= node.start_time and record.ts < node.end_time:
                node.add_memory_record(record)
                self.processed_record.append(record)
            else:
                self.staled_records.append(record)
        else:
            self._update_memory_event(record, child_found)
