# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import logging
import os
from contextlib import contextmanager

from . import consts


def get_logging_level():
    log_level = os.environ.get('TORCH_PROFILER_LOG_LEVEL', "INFO").upper()
    if log_level not in logging._levelToName.values():
        log_level = logging.getLevelName(logging.INFO)
    return log_level

logger=None

def get_logger():
    global logger
    if logger is None:
        logger = logging.getLogger(consts.PLUGIN_NAME)
        logger.setLevel(get_logging_level())
    return logger

class LoggingContext:
    '''This class is same with the following contextmanager decorator

    @contextmanager
    def log_context(logger, level=None, handler=None):
        old_level = None
        if level is not None:
            old_level = logger.level
            logger.setLevel(level)
        try:
            if handler:
                logger.addHandler(handler)
            logger.setLevel(level)
            yield logger
        finally:
            if handler:
                logger.removeHandler(handler)
            if old_level is not None:
                logger.setLevel(old_level)
    '''
    def __init__(self, logger, level=None, handler=None, close=True):
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)
        return self.logger

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()

@contextmanager
def mp_logging(log_level=None):
    import multiprocessing as mp
    if log_level is None:
        log_level = get_logging_level()

    with LoggingContext(mp.get_logger(), log_level, logging.StreamHandler()) as logger:
        yield logger

def is_chrome_trace_file(path):
    return consts.WORKER_PATTERN.match(path)
