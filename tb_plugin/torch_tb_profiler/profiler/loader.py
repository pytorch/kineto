# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys

from .. import consts, io, utils
from ..run import Run
from .data import RunData, RunProfileData
from .run_generator import RunGenerator, DistributedRunGenerator

logger = utils.get_logger()


class RunLoader(object):
    def __init__(self, name, run_dir, caches):
        self.run = RunData(name, run_dir)
        self.caches = caches
        self.has_communication = None

    def load(self):
        self._parse()
        if len(self.run.profiles) == 0:
            logger.warning("No profile data found.")
            return None

        self._process()

        self._analyze()

        run = self._generate_run()
        return run

    def _parse(self):
        workers = []
        spans = []
        for path in io.listdir(self.run.run_dir):
            if io.isdir(io.join(self.run.run_dir, path)):
                continue
            match = consts.WORKER_PATTERN.match(path)
            if not match:
                continue

            worker = match.group(1)
            span = match.group(2)
            if span:
                # remove the starting dot (.)
                span = span[1:]
                spans.append(span)

            workers.append((worker, span, path))

        spans.sort()

        for worker, span, path in sorted(workers):
            try:
                # convert the span timestamp to the index.
                span_index = None if span is None else spans.index(span)
                data = RunProfileData.parse(self.run.run_dir, worker, span_index, path, self.caches)
                self.run.profiles[(worker, span)] = data
            except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)

    def _process(self):
        spans = self.run.get_spans()
        if spans is None:
            self.has_communication = self._process_profiles(self.run.profiles.values())
        else:
            self.has_communication = {}
            for span in spans:
                profiles = self.run.get_profiles(span)
                self.has_communication[span] = self._process_profiles(profiles)

    def _process_profiles(self, profiles):
        has_communication = True
        comm_node_lists = []
        for data in profiles:
            logger.debug("Processing profile data")
            data.process()
            # Set has_communication to False and disable distributed view if any one worker has no communication
            if not data.has_communication:
                has_communication = False
            else:
                comm_node_lists.append(data.comm_node_list)
                if len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                    logger.error("Number of communication operation nodes don't match between workers in run:", self.run.name)
                    has_communication = False
            logger.debug("Processing profile data finish")

        if not has_communication:
            return has_communication

        worker_num = len(comm_node_lists)
        for i, node in enumerate(comm_node_lists[0]):
            kernel_range_size = len(node.kernel_ranges)
            # loop for all communication kernel ranges in order
            for j in range(kernel_range_size):
                min_range = sys.maxsize
                # For each kernel_range, find the minist between workers as the real communication time
                for k in range(worker_num):
                    kernel_ranges = comm_node_lists[k][i].kernel_ranges
                    if len(kernel_ranges) != kernel_range_size:
                        logger.error("Number of communication kernels don't match between workers in run:", self.run.name)
                        has_communication = False
                        return has_communication
                    if kernel_ranges:
                        if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                            min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
                for k in range(worker_num):
                    comm_node_lists[k][i].real_time += min_range

        for data in profiles:
            data.communication_parse()

        return has_communication

    def _analyze(self):
        for data in self.run.profiles.values():
            logger.debug("Analyzing profile data")
            data.analyze()
            logger.debug("Analyzing profile data finish")

    def _generate_run(self):
        run = Run(self.run.name, self.run.run_dir)
        for (worker, span), data in self.run.profiles.items():
            generator = RunGenerator(worker, span, data)
            profile = generator.generate_run_profile()
            run.add_profile(profile)

        if isinstance(self.has_communication, dict):
            for span, has_communication in self.has_communication.items():
                if has_communication:
                    generator = DistributedRunGenerator(self.run.get_profiles(span=span), span)
                    # profile has (All, span) as the worker/span pair
                    profile = generator.generate_run_profile()
                    run.add_profile(profile)
        else:
            if self.has_communication:
                generator = DistributedRunGenerator(self.run.profiles.values(), None)
                profile = generator.generate_run_profile()
                run.add_profile(profile)
        return run
