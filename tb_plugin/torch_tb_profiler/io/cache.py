import multiprocessing as mp
import os

from .. import utils
from .file import File, download_file

logger = utils.get_logger()

class Cache:
    def __init__(self):
        self._lock = mp.Lock()
        self._manager = mp.Manager()
        self._cache_dict = self._manager.dict()
        self._tempfiles = self._manager.list()

    def __getstate__(self):
        '''The multiprocessing module can start one of three ways: spawn, fork, or forkserver. 
        The default mode is fork in Unix and spawn on Windows and macOS.
        Therefore, the __getstate__ and __setstate__ are used to pickle/unpickle the state in spawn mode.
        '''
        data = self.__dict__.copy()
        # remove the _manager to bypass the following pickle error
        # TypeError: cannot pickle 'weakref' object
        del data['_manager']
        logger.debug("Cache.__getstate__: %s " % data)
        return data

    def __setstate__(self, state):
        '''The default logging level in new process is warning. Only warning and error log can be written to 
        streams.
        '''
        with utils.mp_logging() as logger:
            logger.debug("Cache.__setstate__ %s " % state)
        self.__dict__.update(state)

    def read(self, filename):
        local_file = self._cache_dict.get(filename)
        if local_file is None:
            local_file = download_file(filename)
            # skip the cache for local files
            if local_file != filename:
                with self._lock:
                    self._cache_dict[filename] = local_file

        logger.debug("reading local cache %s for file %s" % (local_file, filename))
        with File(local_file, 'rb') as f:
            return f.read()

    def add_tempfile(self, filename):
        self._tempfiles.append(filename)

    def close(self):
        for file in self._tempfiles:
            logger.info("remove tempfile %s" % file)
            os.remove(file)
        for key, value in self._cache_dict.items():
            if key != value:
                logger.info("remove temporary file %s" % value)
                os.remove(value)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        self._manager.__exit__(exc_type, exc_value, traceback)
