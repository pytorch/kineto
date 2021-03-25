# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import json
import os
import tempfile
from collections import OrderedDict

from . import trace
from .kernel_parser import KernelParser
from .module_parser import ModuleParser
from .overall_parser import OverallParser
from .. import consts, utils

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
        self.kernel_list_groupby_name_op = None
        self.kernel_stat = None
        self.recommendations = []

    @staticmethod
    def parse(run_dir, worker):
        logger.debug("Parse trace, run_dir=%s, worker=%s", run_dir, worker)

        trace_path = os.path.join(run_dir, "{}{}".format(worker, consts.TRACE_FILE_SUFFIX))
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
            with fopen(trace_path, 'r') as f:
                trace_json = json.load(f, strict=False)
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            logger.warning("Get JSONDecodeError: %s, Re-encode it to temp file: %s", e.msg, fp.name)
            trace_path = fp.name

        profile = RunProfileData(worker)
        profile.trace_file_path = trace_path
        if type(trace_json) is dict:
            metadata = trace_json.get("profilerMetadata", None)
            version = metadata.get("DataSchemaVersion") if metadata else None
            profile.data_schema_version = version
            trace_json = trace_json["traceEvents"]

        parser = trace.get_event_parser(profile.data_schema_version)
        profile.events = []
        for data in trace_json:
            event = parser.parse(data)
            if event is not None:
                profile.events.append(event)

        return profile

    def process(self):
        logger.debug("ModuleParser")
        module_parser = ModuleParser()
        module_parser.parse_events(self.events)
        self.op_list_groupby_name = module_parser.op_list_groupby_name
        self.op_list_groupby_name_input = module_parser.op_list_groupby_name_input
        self.kernel_list_groupby_name_op = module_parser.kernel_list_groupby_name_op

        logger.debug("OverallParser")
        overall_parser = OverallParser()
        overall_parser.parse_events(self.events, module_parser.runtime_node_list, module_parser.device_node_list)
        self.has_runtime = overall_parser.has_runtime
        self.has_kernel = overall_parser.has_kernel
        self.has_memcpy_or_memset = overall_parser.has_memcpy_or_memset
        self.steps_costs = overall_parser.steps_costs
        self.steps_names = overall_parser.steps_names
        self.avg_costs = overall_parser.avg_costs

        if self.has_kernel:
            logger.debug("KernelParser")
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat

    def analyze(self):
        self.recommendations = []
        dataloader_ratio = self.avg_costs.dataloader_cost / self.avg_costs.step_total_cost
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
