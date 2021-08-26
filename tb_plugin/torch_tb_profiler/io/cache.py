# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import os

from .. import utils
from ..profiler import multiprocessing as mp
from . import file
from .file import download_file, read

logger = utils.get_logger()

class Cache:
    def __init__(self):
        self._lock = mp.Lock()
        self._manager = mp.Manager()
        self._cache_dict = self._manager.dict()

    def __getstate__(self):
        '''The multiprocessing module can start one of three ways: spawn, fork, or forkserver. 
        The default mode is fork in Unix and spawn on Windows and macOS.
        Therefore, the __getstate__ and __setstate__ are used to pickle/unpickle the state in spawn mode.
        '''
        data = self.__dict__.copy()
        # remove the _manager to bypass the following pickle error
        # TypeError: cannot pickle 'weakref' object
        if hasattr(self, '_manager'):
            del data['_manager']
        logger.debug("Cache.__getstate__: %s " % data)
        return data, file._REGISTERED_FILESYSTEMS

    def __setstate__(self, state):
        '''The default logging level in new process is warning. Only warning and error log can be written to 
        streams. 
        So, we need call use_absl_handler in the new process.
        '''
        from absl import logging
        logging.use_absl_handler()
        logger.debug("Cache.__setstate__ %s " % (state,))
        data, file._REGISTERED_FILESYSTEMS = state
        self.__dict__.update(data)

    def read(self, filename):
        local_file = self.get_remote_cache(filename)
        return read(local_file)

    def get_remote_cache(self, filename):
        '''Try to get the local file in the cache. download it to local if it cannot be found in cache.'''
        local_file = self.get_file(filename)
        if local_file is None:
            local_file = download_file(filename)
            # skip the cache for local files
            if local_file != filename:
                self.add_file(filename, local_file)

        return local_file

    def get_file(self, filename):
        return self._cache_dict.get(filename)

    def add_file(self, source_file, local_file):
        with self._lock:
            logger.debug("add local cache %s for file %s" % (local_file, source_file))
            self._cache_dict[source_file] = local_file

    def close(self):
        for key, value in self._cache_dict.items():
            if key != value:
                logger.info("remove temporary file %s" % value)
                os.remove(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self._manager.__exit__(exc_type, exc_value, traceback)

