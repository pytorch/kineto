# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import gzip
import io as sysio
import json
import re
import tempfile
from collections import OrderedDict

from .. import io, utils
from . import trace
from .communication import analyze_communication_nodes
from .event_parser import EventParser, ProfileRole
from .kernel_parser import KernelParser
from .module_parser import ModuleParser
from .overall_parser import OverallParser

logger = utils.get_logger()


class RunData(object):
    def __init__(self, name, run_dir):
        self.name = name
        self.run_dir = run_dir
        self.profiles = OrderedDict()

class RunProfileData(object):
    def __init__(self, worker):
        self.worker = worker
        self.data_schema_version = None
        self.events = None
        self.trace_file_path = None
        self.has_runtime = False
        self.has_kernel = False
        self.has_communication = False
        self.has_memcpy_or_memset = False
        self.steps_costs = None
        self.steps_names = None
        self.avg_costs = None
        self.op_list_groupby_name = None
        self.op_list_groupby_name_input = None
        self.stack_lists_group_by_name = None
        self.stack_lists_group_by_name_input = None
        self.kernel_list_groupby_name_op = None
        self.kernel_stat = None
        self.recommendations = []
        self.comm_node_list = None
        self.comm_overlap_costs = None
        self.total_comm_stats = None
        self.step_comm_stats = None

    @staticmethod
    def parse(run_dir, worker, path, caches):
        logger.debug("Parse trace, run_dir=%s, worker=%s", run_dir, path)

        trace_path, trace_json= RunProfileData._preprocess_file(caches, io.join(run_dir, path))

        profile = RunProfileData(worker)
        profile.trace_file_path = trace_path
        if type(trace_json) is dict:
            metadata = trace_json.get("profilerMetadata", None)
            version = metadata.get("DataSchemaVersion") if metadata else None
            profile.data_schema_version = version
            trace_json = trace_json["traceEvents"]

        profile.events = []
        for data in trace_json:
            event = trace.create_event(data)
            if event is not None:
                profile.events.append(event)

        return profile

    @staticmethod
    def _preprocess_file(caches, trace_path):
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        local_file = caches.download_file(trace_path)
        data = io.read(local_file)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        try:
            trace_json = json.loads(data)
        except json.decoder.JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handle JSONDecodeError, re-encode it and save to a temp file
            try:
                trace_json = json.loads(data, strict=False)
            except json.decoder.JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode("utf-8")
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    trace_json = json.loads(fout.getvalue())

            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            logger.warning("Get JSONDecodeError: %s, Re-encode it to temp file: %s", e.msg, fp.name)
            caches.add_file(local_file, fp.name)
            trace_path = fp.name

        return trace_path, trace_json

    def process(self):
        parser = EventParser()
        node_context = parser.parse(self.events)

        self.has_runtime = parser.has_runtime
        self.has_kernel = parser.has_kernel
        self.has_communication = parser.has_communication
        self.has_memcpy_or_memset = parser.has_memcpy_or_memset
        self.steps_names = parser.steps_names

        # Parse communications.
        self.comm_node_list = parser.generate_communication_nodes()

        # Starting aggregate
        logger.debug("ModuleParser")
        module_parser = ModuleParser()
        module_parser.aggregate(node_context)
        self.op_list_groupby_name = module_parser.op_list_groupby_name
        self.op_list_groupby_name_input = module_parser.op_list_groupby_name_input
        self.stack_lists_group_by_name = module_parser.stack_lists_group_by_name
        self.stack_lists_group_by_name_input = module_parser.stack_lists_group_by_name_input
        self.kernel_list_groupby_name_op = module_parser.kernel_list_groupby_name_op

        logger.debug("OverallParser")
        overall_parser = OverallParser()
        overall_parser.aggregate(parser.steps, parser.role_ranges)
        self.avg_costs = overall_parser.avg_costs
        self.steps_costs = overall_parser.steps_costs
        self.comm_overlap_costs = overall_parser.communication_overlap

        if self.has_kernel:
            logger.debug("KernelParser")
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat

    def communication_parse(self):
        self.step_comm_stats, self.total_comm_stats = analyze_communication_nodes(self.comm_node_list)

    def analyze(self):
        self.recommendations = []
        dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / self.avg_costs.costs[ProfileRole.Total]
        if dataloader_ratio > 0.05:
            text = "This run has high time cost on input data loading. " \
                   "{}% of the step time is in DataLoader. You could " \
                   "try to set num_workers on DataLoader's construction " \
                   "and enable multi-processes on data loading. " \
                   "Reference: <a href =\"{}\" target=\"_blank\">Single- and Multi-process Data Loading</a>".format(
                       round(dataloader_ratio * 100, 1),
                       "https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading"
                   )
            self.recommendations.append(text)
