# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys
from collections import OrderedDict
from multiprocessing import Barrier, Process, Queue

from .. import consts, io, utils
from ..run import Run
from .data import DistributedRunProfileData, RunData, RunProfileData
from .run_generator import DistributedRunGenerator, RunGenerator

logger = utils.get_logger()


class RunLoader(object):
    def __init__(self, name, run_dir, caches):
        self.run = RunData(name, run_dir)
        self.caches = caches
        self.has_communication = True
        self.queue = Queue()

    def load(self):
        workers = []
        for path in io.listdir(self.run.run_dir):
            if io.isdir(io.join(self.run.run_dir, path)):
                continue
            match = consts.WORKER_PATTERN.match(path)
            if not match:
                continue

            worker = match.group(1)
            workers.append((worker, path))

        barrier = Barrier(len(workers) + 1)
        for worker, path in sorted(workers):
            p = Process(target=self._process_data, args=(worker, path, barrier))
            p.start()

        logger.info("starting all processing")
        # since there is one queue, its data must be read before join.
        # https://stackoverflow.com/questions/31665328/python-3-multiprocessing-queue-deadlock-when-calling-join-before-the-queue-is-em
        barrier.wait()

        distributed_data = OrderedDict()
        run = Run(self.run.name, self.run.run_dir)
        while self.queue.qsize() > 0:
            r, d = self.queue.get()
            run.add_profile(r)
            distributed_data[d.worker] = d

        distributed_profile = self._process_communication(distributed_data)
        if distributed_profile is not None:
            run.add_profile(distributed_profile)

        # for no daemon process, no need to join them since it will automatically join
        return run

    def _process_data(self, worker, path, barrier):
        import absl.logging
        absl.logging.use_absl_handler()

        try:
            logger.debug("starting process_data")
            data = RunProfileData.parse(self.run.run_dir, worker, path, self.caches)
            data.process()
            data.analyze()

            generator = RunGenerator(worker, data)
            profile = generator.generate_run_profile()
            dist_data = DistributedRunProfileData(data)

            self.queue.put((profile, dist_data))
        except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)
        barrier.wait()
        logger.debug("finishing process data")

    def _process_communication(self, profiles):
        comm_node_lists = []
        for data in profiles.values():
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
            return None

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
                        return None
                    if kernel_ranges:
                        if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                            min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
                for k in range(worker_num):
                    comm_node_lists[k][i].real_time += min_range

        for data in profiles.values():
            data.communication_parse()

        generator = DistributedRunGenerator(profiles)
        profile = generator.generate_run_profile()
        return profile
