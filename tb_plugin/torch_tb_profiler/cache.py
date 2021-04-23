from . import io, utils

logger = utils.get_logger()

def read_cache(caches, filename):
    local_file = caches.get(filename)
    if local_file is None:
        local_file = io.download_file(filename)
        caches[filename] = local_file

    logger.debug("reading local cache for file %s" % filename)
    with io.File(local_file, 'rb') as f:
        return f.read()
