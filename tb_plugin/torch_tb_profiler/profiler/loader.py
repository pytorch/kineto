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
        self.queue = Queue()

    def load(self):
        workers = []
        spans_by_workers = {}
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
                spans_by_workers.setdefault(worker, []).append(span)

            workers.append((worker, span, path))

        for s in spans_by_workers.values():
            s.sort()

        barrier = Barrier(len(workers) + 1)
        for worker, span, path in sorted(workers):
            # convert the span timestamp to the index.
            s = spans_by_workers.get(worker)
            span_index = None if span is None else s.index(span)
            p = Process(target=self._process_data, args=(worker, span_index, path, barrier))
            p.start()

        logger.info("starting all processing")
        # since there is one queue, its data must be read before join.
        # https://stackoverflow.com/questions/31665328/python-3-multiprocessing-queue-deadlock-when-calling-join-before-the-queue-is-em
        barrier.wait()

        run = Run(self.run.name, self.run.run_dir)
        while self.queue.qsize() > 0:
            r, d = self.queue.get()
            run.add_profile(r)
            self.run.add_profile(d)

        distributed_profiles = self._process_spans()
        if distributed_profiles is not None:
            if isinstance(distributed_profiles, list):
                for d in distributed_profiles:
                    run.add_profile(d)
            else:
                run.add_profile(distributed_profiles)

        # for no daemon process, no need to join them since it will automatically join
        return run

    def _process_data(self, worker, span, path, barrier):
        import absl.logging
        absl.logging.use_absl_handler()

        try:
            logger.debug("starting process_data")
            data = RunProfileData.parse(self.run.run_dir, worker, span, path, self.caches)
            data.process()
            data.analyze()

            generator = RunGenerator(worker, span, data)
            profile = generator.generate_run_profile()
            dist_data = DistributedRunProfileData(data)

            self.queue.put((profile, dist_data))
        except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)
        barrier.wait()
        logger.debug("finishing process data")

    def _process_spans(self):
        spans = self.run.get_spans()
        if spans is None:
            return self._process_profiles(self.run.distributed_profiles.values(), None)
        else:
            span_profiles = []
            for span in spans:
                profiles = self.run.get_profiles(span=span)
                p = self._process_profiles(profiles)
                if p is not None:
                    span_profiles.append(p)
            return span_profiles

    def _process_profiles(self, profiles, span):
        has_communication = True
        comm_node_lists = []
        for data in profiles:
            logger.debug("Processing profile data")
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
                        has_communication = False
                        return None
                    if kernel_ranges:
                        if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                            min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
                for k in range(worker_num):
                    comm_node_lists[k][i].real_time += min_range

        for data in profiles:
            data.communication_parse()

        generator = DistributedRunGenerator(profiles, span)
        profile = generator.generate_run_profile()
        return profile
