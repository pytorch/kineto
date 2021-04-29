# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import gzip
import io
import json
import os
import re
import tempfile
from collections import OrderedDict

from .. import consts, utils
from . import trace
from .kernel_parser import KernelParser
from .module_parser import ModuleParser
from .overall_parser import OverallParser, ProfileRole

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
        self.comm_node_list = []
        self.total_comm_stats = dict()
        self.step_comm_stats = dict()

    @staticmethod
    def parse(run_dir, worker):
        logger.debug("Parse trace, run_dir=%s, worker=%s", run_dir, worker)

        trace_path = os.path.join(run_dir, "{}{}".format(worker, consts.TRACE_FILE_SUFFIX))
        trace_path, trace_json= RunProfileData._preprocess_file(trace_path)

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
    def _preprocess_file(trace_path):
        fopen = open
        if not os.path.isfile(trace_path):
            trace_path += ".gz"
            fopen = gzip.open

        if not os.path.isfile(trace_path):
            raise FileNotFoundError(trace_path)

        try:
            with fopen(trace_path, 'r') as f:
                trace_json = json.load(f)
        except json.decoder.JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handleJSONDecodeError, re-encode it and save to a temp file
            with fopen(trace_path, 'rt') as f:
                try:
                    trace_json = json.load(f, strict=False)
                except json.decoder.JSONDecodeError:
                    # TODO: remove the workaround after the libkineto fix for N/A is merged into pytorch
                    f.seek(0)
                    with io.StringIO() as fout:
                        for line in f:
                            # only replace the N/A without surrounding double quote
                            fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", line))
                        trace_json = json.loads(fout.getvalue())

            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            logger.warning("Get JSONDecodeError: %s, Re-encode it to temp file: %s", e.msg, fp.name)
            trace_path = fp.name
        
        return trace_path, trace_json

    def process(self):
        logger.debug("ModuleParser")
        module_parser = ModuleParser()
        module_parser.parse_events(self.events)
        self.op_list_groupby_name = module_parser.op_list_groupby_name
        self.op_list_groupby_name_input = module_parser.op_list_groupby_name_input
        self.stack_lists_group_by_name = module_parser.stack_lists_group_by_name
        self.stack_lists_group_by_name_input = module_parser.stack_lists_group_by_name_input
        self.kernel_list_groupby_name_op = module_parser.kernel_list_groupby_name_op

        logger.debug("OverallParser")
        overall_parser = OverallParser()
        overall_parser.parse_events(self.events, module_parser.runtime_node_list, module_parser.device_node_list, module_parser.communication_data)
        self.has_runtime = bool(overall_parser.role_ranges[ProfileRole.Runtime])
        self.has_kernel = bool(overall_parser.role_ranges[ProfileRole.Kernel])
        self.has_memcpy_or_memset = bool(overall_parser.role_ranges[ProfileRole.Memcpy] or overall_parser.role_ranges[ProfileRole.Memset])
        self.steps_costs = overall_parser.steps_costs
        self.steps_names = overall_parser.steps_names
        self.avg_costs = overall_parser.avg_costs
        self.comm_node_list = overall_parser.comm_node_list

        if self.has_kernel:
            logger.debug("KernelParser")
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat

    def communication_parse(self):
        for comm_node in self.comm_node_list:
            if comm_node.step_index not in self.step_comm_stats:
                self.step_comm_stats[comm_node.step_index] = [0, 0]
            self.step_comm_stats[comm_node.step_index][0] += comm_node.total_time
            self.step_comm_stats[comm_node.step_index][1] += comm_node.real_time
            if comm_node.name not in self.total_comm_stats:
                self.total_comm_stats[comm_node.name] = [0, 0, 0, 0]
            self.total_comm_stats[comm_node.name][0] += 1
            bytes_one_value = 0
            for i in range(len(comm_node.input_shape)):
                if comm_node.input_type[i] == 'long int':
                    bytes_one_value = 8
                elif comm_node.input_type[i] == 'float':
                    bytes_one_value = 4
                else:
                    logger.warning("Found an unknown tensor type:", comm_node.input_type[i])
                    bytes_one_value = 0
                total_size = 1
                for size in comm_node.input_shape[i]:
                    total_size *= size
                self.total_comm_stats[comm_node.name][1] += total_size * bytes_one_value
            self.total_comm_stats[comm_node.name][2] += comm_node.total_time
            self.total_comm_stats[comm_node.name][3] += comm_node.real_time

        for k,v in self.total_comm_stats.items():
            print(k,v)
        for k,v in self.step_comm_stats.items():
            print(k,v)

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
