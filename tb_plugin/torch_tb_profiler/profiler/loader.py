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
        self.has_communication = True

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

            workers.append((worker, span, path))

        for worker, span, path in sorted(workers):
            try:
                data = RunProfileData.parse(self.run.run_dir, worker, span, path, self.caches)
                self.run.profiles[(worker, span)] = data
            except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)

    def _process(self):
        comm_node_lists = []
        for data in self.run.profiles.values():
            logger.debug("Processing profile data")
            data.process()
            # Set has_communication to False and disable distributed view if any one worker has no communication
            if not data.has_communication:
                self.has_communication = False
            else:
                comm_node_lists.append(data.comm_node_list)
                if len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                    logger.error("Number of communication operation nodes don't match between workers in run:", self.run.name)
                    self.has_communication = False
            logger.debug("Processing profile data finish")

        if not self.has_communication:
            return

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
                        self.has_communication = False
                        return
                    if kernel_ranges:
                        if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                            min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
                for k in range(worker_num):
                    comm_node_lists[k][i].real_time += min_range

        for data in self.run.profiles.values():
            data.communication_parse()


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
        if self.has_communication:
            generator = DistributedRunGenerator(self.run.profiles)
            profile = generator.generate_run_profile()
            run.add_profile(profile)
        return run
