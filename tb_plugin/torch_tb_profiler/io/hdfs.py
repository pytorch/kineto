import os

import fsspec
from fsspec.implementations import arrow

from .. import utils
from .base import BaseFileSystem, RemotePath, StatData
from .utils import as_bytes, as_text, parse_blob_url

logger = utils.get_logger()

class HadoopFileSystem(RemotePath, BaseFileSystem):
    def __init__(self) -> None:
        super().__init__()
    
    def get_fs(self) -> arrow.HadoopFileSystem:
        return fsspec.filesystem("hdfs")

    def exists(self, filename):
        return self.get_fs().exists(filename)
    
    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        fs = self.get_fs()
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        offset = None
        if continue_from is not None:
            offset = continue_from.get("opaque_offset", None)
        with fs.open(path=filename, mode=mode, encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)
        continuation_token = {"opaque_offset": f.tell()}
        return (data, continuation_token)
    
    def write(self, filename, file_content, binary_mode=False):
        fs = self.get_fs()
        if binary_mode:
            fs.write_bytes(filename, as_bytes(file_content))
        else:
            fs.write_text(filename, as_text(file_content), encoding="utf8")
    
    def glob(self, filename):
        return self.get_fs().glob(filename)
    
    def isdir(self, dirname):
        return self.get_fs().isdir(dirname)
    
    def listdir(self, dirname):
        fs = self.get_fs()
        full_path = fs.listdir(dirname, detail=False)
        # strip the protocol from the root path because the path returned by
        # pyarrow listdir is not prefixed with the protocol.
        root_path_to_strip = fs._strip_protocol(dirname)
        return [os.path.relpath(path, root_path_to_strip) for path in full_path]
    
    def makedirs(self, path):
        return self.get_fs().makedirs(path, exist_ok=True)
    
    def stat(self, filename):
        stat = self.get_fs().stat(filename)
        return StatData(stat['size'])
    
    def support_append(self):
        return False
    
    def download_file(self, file_to_download, file_to_save):
        return self.get_fs().download(file_to_download, file_to_save, recursive=True)