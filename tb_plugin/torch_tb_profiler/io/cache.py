from multiprocessing import Lock, Manager

from .. import utils
from .file import File, download_file

logger = utils.get_logger()

class Cache:
    def __init__(self):
        self._lock = Lock()
        self._manager = Manager()
        self._cache_dict = self._manager.dict()

    def read(self, filename):
        local_file = self._cache_dict.get(filename)
        if local_file is None:
            local_file = download_file(filename)
            with self._lock:
                self._cache_dict[filename] = local_file

        logger.debug("reading local cache %s for file %s" % (local_file, filename))
        with File(local_file, 'rb') as f:
            return f.read()
