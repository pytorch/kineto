from .. import utils
from .node import is_operator_node
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
    def __init__(self):
        self.records = []
        self.staled_records = []
        self.processed_record = []

    def parse_events(self, events):
        for event in events:
            if event.type == EventTypes.MEMORY:
                record = MemoryRecord(event.scope, event.pid, event.tid, event.ts, event.device_type, event.device_id, event.bytes)
                self.records.append(record)
        return self.records

    def update_node(self, root):
        for mem_record in self.records:
            root_node = root[mem_record.tid]
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
