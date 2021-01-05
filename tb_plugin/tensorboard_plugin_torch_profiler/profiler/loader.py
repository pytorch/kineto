# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from .data import RunData, RunProfileData
from .run_generator import RunGenerator
from .. import consts, utils
from ..run import Run

logger = utils.get_logger()


class RunLoader(object):
    def __init__(self, name, run_dir):
        self.run = RunData(name, run_dir)

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
        for path in os.listdir(self.run.run_dir):
            if os.path.isdir(path):
                continue
            for pattern in [consts.TRACE_GZIP_FILE_SUFFIX, consts.TRACE_FILE_SUFFIX]:
                if path.endswith(pattern):
                    worker = path[:-len(pattern)]
                    workers.append(worker)
                    break

        for worker in sorted(workers):
            try:
                data = RunProfileData.parse(self.run.run_dir, worker)
                self.run.profiles[worker] = data
            except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)

    def _process(self):
        for data in self.run.profiles.values():
            logger.debug("Processing profile data")
            data.process()
            logger.debug("Processing profile data finish")

    def _analyze(self):
        for data in self.run.profiles.values():
            logger.debug("Analyzing profile data")
            data.analyze()
            logger.debug("Analyzing profile data finish")

    def _generate_run(self):
        run = Run(self.run.name, self.run.run_dir)
        for worker, data in self.run.profiles.items():
            generator = RunGenerator(worker, data)
            profile = generator.generate_run_profile()
            run.add_profile(profile)
        return run
