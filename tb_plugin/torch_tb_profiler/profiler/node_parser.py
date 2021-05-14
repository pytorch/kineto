from .. import utils
from .node import (CommunicationNode, DeviceNode, OperatorNode,
                   ProfilerStepNode, RuntimeNode)
from .trace import EventTypes

logger = utils.get_logger()

CommunicationOpNameSet = ['nccl:broadcast', 'nccl:all_reduce']


class NodeContext:
    def __init__(self):
        self.tid2list = {} # value is a list of OperatorNode and ProfilerStepNode. Do not include RuntimeNode
        self.tid2zero_rt_list = {}  # value is a list of RuntimeNode with external_id=0. They will be attached to root nodes.
        self.corrid_to_device = {}  # value is a list of DeviceNode
        self.corrid_to_runtime = {}  # value is a RuntimeNode
        self.externalid_to_runtime = {}  # value is a list of RuntimeNode
class NodeParser:
    def __init__(self):
        self.communication_data = {}
        self.device_node_list = []
        self.runtime_node_list = []
        self.comm_node_list = []

    def parse_events(self, events, context):
        # For OperatorNode and ProfilerStepNode:
        #   Use time interval containing relationship to build father-child correlation,
        #   which is consistent with autograd profiler.
        # For RuntimeNode:
        #   Use external_id to build correlation with its father OperatorNode or ProfilerStepNode.
        #   Because in the case when RuntimeNode has duration 0 and starts at same time as a OperatorNode,
        #   just use interval containing relationship can't tell it is child or brother of the OperatorNode.
        tid2list = context.tid2list
        tid2zero_rt_list = context.tid2zero_rt_list
        corrid_to_device = context.corrid_to_device
        corrid_to_runtime = context.corrid_to_runtime
        externalid_to_runtime = context.externalid_to_runtime
        
        for event in events:
            self._parse_event(event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list)

        # associate CUDA Runtimes with CPU events
        for _, op_list in tid2list.items():
            for op in op_list:
                runtime_nodes = externalid_to_runtime.pop(op.external_id, [])
                if runtime_nodes:
                    op.runtimes.extend(runtime_nodes)
        for ext_id in externalid_to_runtime:
            if ext_id != 0:
                logger.warning("{} Runtime with external id {} don't correlate to any operator!".format(
                    len(externalid_to_runtime[ext_id]), ext_id))

        # Sort the communication node according the start time, this is for correlating communication node between workers
        for comm_node in self.communication_data.values():
            comm_node.kernel_ranges.sort(key=lambda x: (x[0], -x[1]))
            self.comm_node_list.append(comm_node)
        self.comm_node_list.sort(key=lambda x: (x.start_time, -x.end_time))

    def _parse_event(self, event, corrid_to_device, corrid_to_runtime, externalid_to_runtime, tid2list, tid2zero_rt_list):
        corrid = event.args.get("correlation", None)
        tid = event.tid
        if event.type in [EventTypes.KERNEL, EventTypes.MEMCPY, EventTypes.MEMSET]:
            device_node = DeviceNode.create(event)
            if corrid in corrid_to_runtime:
                rt_node = corrid_to_runtime[corrid]  # Don't pop it because it may be used by next kernel.
                if rt_node.device_nodes is None:
                    rt_node.device_nodes = []
                rt_node.device_nodes.append(device_node)

                # Check the external_id
                if rt_node.external_id != device_node.external_id:
                    logger.warning("Runtime and Device-op have same correlation id but with different external id!")
            else:
                corrid_to_device.setdefault(corrid, []).append(device_node)
            self.device_node_list.append(device_node)
        elif event.type == EventTypes.RUNTIME:
            device_nodes = corrid_to_device.pop(corrid, None)
            rt_node = RuntimeNode.create(event, device_nodes)
            corrid_to_runtime[corrid] = rt_node
            externalid_to_runtime.setdefault(rt_node.external_id, []).append(rt_node)
            # Some runtimes has external_id 0, which will not be correlated to any operator.
            # So get them and attach them to root node.
            if rt_node.external_id == 0:
                tid2zero_rt_list.setdefault(tid, []).append(rt_node)
            self.runtime_node_list.append(rt_node)

            # check the external_id
            if device_nodes:
                for device_node in device_nodes:
                    if rt_node.external_id != device_node.external_id:
                        logger.warning("Runtime and Device-op have same correlation id but with different external id!")
        elif event.type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
            if event.type == EventTypes.PROFILER_STEP:
                op_node = ProfilerStepNode.create(event, event.input_shape, event.input_type, None)
            else:
                op_node = OperatorNode.create(event, event.input_shape, event.input_type, event.callstack)
            if event.name in CommunicationOpNameSet:
                self.communication_data[op_node.external_id] = CommunicationNode.create(event, op_node.input_shape, op_node.input_type)
            tid2list.setdefault(tid, []).append(op_node)
