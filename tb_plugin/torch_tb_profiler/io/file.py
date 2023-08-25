"""
This file is forked from
https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/tensorflow_stub/io/gfile.py.
The following functionalities are added after forking:
* Check Azure Blob & Google Cloud available or not
* get_filesystem changes to support Azure Blobs
* add BaseFileSystem and PathBase abstracted class for the filesystem.
* add download_file for each file system to cache the remote file to local temporary folder.
* add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for S3 file system which is not supported by tensorboard.
* add Azure blob file system
* add Google Cloud file system
* add specialized walk for Local file system, Azure Blob and Google Cloud to improve the walk performance.
* add global wrapper for abspath, basename, join, download_file.
* change the global walk wrapper to support specialized walk.
"""
import glob as py_glob
import os
import tempfile

from .. import utils
from .base import BaseFileSystem, LocalPath, RemotePath, StatData
from .utils import as_bytes, as_text, parse_blob_url

logger = utils.get_logger()

try:
    import boto3
    import botocore.exceptions

    S3_ENABLED = True
except ImportError:
    S3_ENABLED = False

try:
    from azure.storage.blob import ContainerClient
    BLOB_ENABLED = True
except ImportError:
    BLOB_ENABLED = False

try:
    # Imports the Google Cloud client library
    from google.cloud import storage
    GS_ENABLED = True
except ImportError:
    GS_ENABLED = False

try:
    # Imports the HDFS library
    from fsspec.implementations.arrow import HadoopFileSystem
    HDFS_ENABLED = True
except ImportError:
    HDFS_ENABLED = False

_DEFAULT_BLOCK_SIZE = 16 * 1024 * 1024

# Registry of filesystems by prefix.
#
# Currently supports:
#  * "s3://" URLs for S3 based on boto3
#  * "https://<account>.blob.core.windows.net" for Azure Blob based on azure-storage-blob
#  * "gs://" URLs for Google Cloud based on google-cloud-storage
#  * Local filesystem when not match any prefix.
_REGISTERED_FILESYSTEMS = {}


def register_filesystem(prefix, filesystem):
    if ":" in prefix:
        raise ValueError("Filesystem prefix cannot contain a :")
    _REGISTERED_FILESYSTEMS[prefix] = filesystem


def get_filesystem(filename):
    """Return the registered filesystem for the given file."""
    prefix = ""
    index = filename.find("://")
    if index >= 0:
        prefix = filename[:index]
    if prefix.upper() in ('HTTP', 'HTTPS'):
        root, _ = parse_blob_url(filename)
        if root.lower().endswith('.blob.core.windows.net'):
            fs = _REGISTERED_FILESYSTEMS.get('blob', None)
        else:
            raise ValueError("Not supported file system for prefix %s" % root)
    else:
        fs = _REGISTERED_FILESYSTEMS.get(prefix, None)
    if fs is None:
        raise ValueError("No recognized filesystem for prefix %s" % prefix)
    return fs


class LocalFileSystem(LocalPath, BaseFileSystem):
    def __init__(self):
        pass

    def exists(self, filename):
        return os.path.exists(filename)

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        mode = "rb" if binary_mode else "r"
        encoding = None if binary_mode else "utf8"
        if not self.exists(filename):
            raise FileNotFoundError(filename)

        offset = None
        if continue_from is not None:
            offset = continue_from.get("opaque_offset", None)
        with open(filename, mode, encoding=encoding) as f:
            if offset is not None:
                f.seek(offset)
            data = f.read(size)
            # The new offset may not be `offset + len(data)`, due to decoding
            # and newline translation.
            # So, just measure it in whatever terms the underlying stream uses.
            continuation_token = {"opaque_offset": f.tell()}
            return (data, continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file, overwriting any existing contents.
        """
        self._write(filename, file_content, "wb" if binary_mode else "w")

    def support_append(self):
        return True

    def append(self, filename, file_content, binary_mode=False):
        """Append string file contents to a file.
        """
        self._write(filename, file_content, "ab" if binary_mode else "a")

    def _write(self, filename, file_content, mode):
        encoding = None if "b" in mode else "utf8"
        with open(filename, mode, encoding=encoding) as f:
            compatify = as_bytes if "b" in mode else as_text
            f.write(compatify(file_content))

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        if isinstance(filename, str):
            return [
                matching_filename
                for matching_filename in py_glob.glob(filename)
            ]
        else:
            return [
                matching_filename
                for single_filename in filename
                for matching_filename in py_glob.glob(single_filename)
            ]

    def isdir(self, dirname):
        return os.path.isdir(dirname)

    def listdir(self, dirname):
        entries = os.listdir(dirname)
        entries = [item for item in entries]
        return entries

    def makedirs(self, path):
        os.makedirs(path, exist_ok=True)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # NOTE: Size of the file is given by .st_size as returned from
        # os.stat(), but we convert to .length
        file_length = os.stat(filename).st_size
        return StatData(file_length)

    def walk(self, top, topdown=True, onerror=None):
        # Note on followlinks=True: per the tensorboard documentation [1], users are encouraged to
        # use symlink trees to have fine-grained control over the filesystem layout of runs. To
        # support such trees, we must follow links.
        # [1] https://github.com/tensorflow/tensorboard/blob/master/README.md#logdir--logdir_spec-legacy-mode
        yield from os.walk(top, topdown, onerror, followlinks=True)


class S3FileSystem(RemotePath, BaseFileSystem):
    """Provides filesystem access to S3."""

    def __init__(self):
        if not boto3:
            raise ImportError("boto3 must be installed for S3 support.")
        self._s3_endpoint = os.environ.get("S3_ENDPOINT", None)
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            boto3.setup_default_session(
                aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        if url.startswith("s3://"):
            url = url[len("s3://"):]
        idx = url.index("/")
        bucket = url[:idx]
        path = url[(idx + 1):]
        return bucket, path

    def exists(self, filename):
        """Determines whether a path exists or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string."""
        s3 = boto3.resource("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        args = {}

        # S3 use continuation tokens of the form: {byte_offset: number}
        offset = 0
        if continue_from is not None:
            offset = continue_from.get("byte_offset", 0)

        endpoint = ""
        if size is not None:
            endpoint = offset + size

        if offset != 0 or endpoint != "":
            args["Range"] = "bytes={}-{}".format(offset, endpoint)

        logger.info("s3: starting reading file %s" % filename)
        try:
            stream = s3.Object(bucket, path).get(**args)["Body"].read()
        except botocore.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in ["416", "InvalidRange"]:
                if size is not None:
                    # Asked for too much, so request just to the end. Do this
                    # in a second request so we don't check length in all cases.
                    client = boto3.client("s3", endpoint_url=self._s3_endpoint)
                    obj = client.head_object(Bucket=bucket, Key=path)
                    content_length = obj["ContentLength"]
                    endpoint = min(content_length, offset + size)
                if offset == endpoint:
                    # Asked for no bytes, so just return empty
                    stream = b""
                else:
                    args["Range"] = "bytes={}-{}".format(offset, endpoint)
                    stream = s3.Object(bucket, path).get(**args)["Body"].read()
            else:
                raise

        logger.info("s3: file %s download is done, size is %d" %
                    (filename, len(stream)))
        # `stream` should contain raw bytes here (i.e., there has been neither decoding nor newline translation),
        # so the byte offset increases by the expected amount.
        continuation_token = {"byte_offset": (offset + len(stream))}
        if binary_mode:
            return (bytes(stream), continuation_token)
        else:
            return (stream.decode("utf-8"), continuation_token)

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        if binary_mode:
            if not isinstance(file_content, bytes):
                raise TypeError("File content type must be bytes")
        else:
            file_content = as_bytes(file_content)
        client.put_object(Body=file_content, Bucket=bucket, Key=path)

    def download_file(self, file_to_download, file_to_save):
        logger.info("s3: starting downloading file %s as %s" %
                    (file_to_download, file_to_save))
        # Use boto3.resource instead of boto3.client('s3') to support minio.
        # https://docs.min.io/docs/how-to-use-aws-sdk-for-python-with-minio-server.html
        # To support minio, the S3_ENDPOINT need to be set like: S3_ENDPOINT=http://localhost:9000
        s3 = boto3.resource("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(file_to_download)
        s3.Bucket(bucket).download_file(path, file_to_save)
        logger.info("s3: file %s is downloaded as %s" % (file_to_download, file_to_save))
        return

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find("*")
        quest_i = filename.find("?")
        if quest_i >= 0:
            raise NotImplementedError("{} not supported".format(filename))
        if star_i != len(filename) - 1:
            return []

        filename = filename[:-1]
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)
        p = client.get_paginator("list_objects")
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path):
            for o in r.get("Contents", []):
                key = o["Key"][len(path):]
                if key:
                    keys.append(filename + key)
        return keys

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        if not path.endswith("/"):
            path += "/"
        r = client.list_objects(Bucket=bucket, Prefix=path, Delimiter="/")
        if r.get("Contents") or r.get("CommonPrefixes"):
            return True
        return False

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(dirname)
        p = client.get_paginator("list_objects")
        if not path.endswith("/"):
            path += "/"
        keys = []
        for r in p.paginate(Bucket=bucket, Prefix=path, Delimiter="/"):
            keys.extend(
                o["Prefix"][len(path): -1] for o in r.get("CommonPrefixes", [])
            )
            for o in r.get("Contents", []):
                key = o["Key"][len(path):]
                if key:
                    keys.append(key)
        return keys

    def makedirs(self, dirname):
        """Creates a directory and all parent/intermediate directories."""
        if not self.exists(dirname):
            client = boto3.client("s3", endpoint_url=self._s3_endpoint)
            bucket, path = self.bucket_and_path(dirname)
            if not path.endswith("/"):
                path += "/"
            client.put_object(Body="", Bucket=bucket, Key=path)

    def stat(self, filename):
        """Returns file statistics for a given path."""
        # Size of the file is given by ContentLength from S3
        client = boto3.client("s3", endpoint_url=self._s3_endpoint)
        bucket, path = self.bucket_and_path(filename)

        obj = client.head_object(Bucket=bucket, Key=path)
        return StatData(obj["ContentLength"])


register_filesystem("", LocalFileSystem())
if S3_ENABLED:
    register_filesystem("s3", S3FileSystem())

if BLOB_ENABLED:
    from .azureblob import AzureBlobSystem
    register_filesystem("blob", AzureBlobSystem())

if GS_ENABLED:
    from .gs import GoogleBlobSystem
    register_filesystem("gs", GoogleBlobSystem())

if HDFS_ENABLED:
    from .hdfs import HadoopFileSystem
    register_filesystem("hdfs", HadoopFileSystem())


class File:
    def __init__(self, filename, mode):
        if mode not in ("r", "rb", "br", "w", "wb", "bw"):
            raise ValueError("mode {} not supported by File".format(mode))
        self.filename = filename
        self.fs = get_filesystem(self.filename)
        self.fs_supports_append = self.fs.support_append()
        self.buff = None
        self.buff_chunk_size = _DEFAULT_BLOCK_SIZE
        self.buff_offset = 0
        self.continuation_token = None
        self.write_temp = None
        self.write_started = False
        self.binary_mode = "b" in mode
        self.write_mode = "w" in mode
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        self.buff = None
        self.buff_offset = 0
        self.continuation_token = None

    def __iter__(self):
        return self

    def _read_buffer_to_offset(self, new_buff_offset):
        old_buff_offset = self.buff_offset
        read_size = min(len(self.buff), new_buff_offset) - old_buff_offset
        self.buff_offset += read_size
        return self.buff[old_buff_offset: old_buff_offset + read_size]

    def read(self, n=None):
        """Reads contents of file to a string.

        Args:
            n: int, number of bytes or characters to read, otherwise
                read all the contents of the file

        Returns:
            Subset of the contents of the file as a string or bytes.
        """
        if self.write_mode:
            raise OSError("File not opened in read mode")

        result = None
        if self.buff and len(self.buff) > self.buff_offset:
            # read from local buffer
            if n is not None:
                chunk = self._read_buffer_to_offset(self.buff_offset + n)
                if len(chunk) == n:
                    return chunk
                result = chunk
                n -= len(chunk)
            else:
                # add all local buffer and update offsets
                result = self._read_buffer_to_offset(len(self.buff))

        # read from filesystem
        read_size = max(self.buff_chunk_size, n) if n is not None else None
        (self.buff, self.continuation_token) = self.fs.read(
            self.filename, self.binary_mode, read_size, self.continuation_token)
        self.buff_offset = 0

        # add from filesystem
        if n is not None:
            chunk = self._read_buffer_to_offset(n)
        else:
            # add all local buffer and update offsets
            chunk = self._read_buffer_to_offset(len(self.buff))
        result = result + chunk if result else chunk

        return result

    def write(self, file_content):
        """Writes string file contents to file, clearing contents of the file
        on first write and then appending on subsequent calls.
        """
        if not self.write_mode:
            raise OSError("File not opened in write mode")

        if self.closed:
            raise OSError("File already closed")

        if self.fs_supports_append:
            if not self.write_started:
                # write the first chunk to truncate file if it already exists
                self.fs.write(self.filename, file_content, self.binary_mode)
                self.write_started = True
            else:
                # append the later chunks
                self.fs.append(self.filename, file_content, self.binary_mode)
        else:
            # add to temp file, but wait for flush to write to final filesystem
            if self.write_temp is None:
                mode = "w+b" if self.binary_mode else "w+"
                self.write_temp = tempfile.TemporaryFile(mode)

            compatify = as_bytes if self.binary_mode else as_text
            self.write_temp.write(compatify(file_content))

    def __next__(self):
        line = None
        while True:
            if not self.buff:
                # read one unit into the buffer
                line = self.read(1)
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()
            else:
                index = self.buff.find("\n", self.buff_offset)
                if index != -1:
                    # include line until now plus newline
                    chunk = self.read(index + 1 - self.buff_offset)
                    line = line + chunk if line else chunk
                    return line

                # read one unit past end of buffer
                chunk = self.read(len(self.buff) + 1 - self.buff_offset)
                line = line + chunk if line else chunk
                if line and (line[-1] == "\n" or not self.buff):
                    return line
                if not self.buff:
                    raise StopIteration()

    def next(self):
        return self.__next__()

    def flush(self):
        if self.closed:
            raise OSError("File already closed")

        if not self.fs_supports_append:
            if self.write_temp is not None:
                # read temp file from the beginning
                self.write_temp.flush()
                self.write_temp.seek(0)
                chunk = self.write_temp.read()
                if chunk is not None:
                    # write full contents and keep in temp file
                    self.fs.write(self.filename, chunk, self.binary_mode)
                    self.write_temp.seek(len(chunk))

    def close(self):
        self.flush()
        if self.write_temp is not None:
            self.write_temp.close()
            self.write_temp = None
            self.write_started = False
        self.closed = True


def exists(filename):
    """Determines whether a path exists or not."""
    return get_filesystem(filename).exists(filename)


def abspath(path):
    return get_filesystem(path).abspath(path)


def basename(path):
    return get_filesystem(path).basename(path)


def relpath(path, start):
    return get_filesystem(path).relpath(path, start)


def join(path, *paths):
    return get_filesystem(path).join(path, *paths)


def download_file(file_to_download, file_to_save):
    """Downloads the file, returning a temporary path to the file after finishing."""
    get_filesystem(file_to_download).download_file(file_to_download, file_to_save)


def glob(filename):
    """Returns a list of files that match the given pattern(s)."""
    return get_filesystem(filename).glob(filename)


def is_local(path):
    """Returns whether the path is a local path"""
    return isinstance(get_filesystem(path), LocalFileSystem)


def isdir(dirname):
    """Returns whether the path is a directory or not."""
    return get_filesystem(dirname).isdir(dirname)


def listdir(dirname):
    """Returns a list of entries contained within a directory.

    The list is in arbitrary order. It does not contain the special entries "."
    and "..".
    """
    return get_filesystem(dirname).listdir(dirname)


def makedirs(path):
    """Creates a directory and all parent/intermediate directories."""
    return get_filesystem(path).makedirs(path)


def walk(top, topdown=True, onerror=None):
    """Recursive directory tree generator for directories.

    Args:
      top: string, a Directory name
      topdown: bool, Traverse pre order if True, post order if False.
      onerror: optional handler for errors. Should be a function, it will be
        called with the error as argument. Rethrowing the error aborts the walk.

    Errors that happen while listing directories are ignored.

    Yields:
      Each yield is a 3-tuple:  the pathname of a directory, followed by lists
      of all its subdirectories and leaf files.
      (dirname, [subdirname, subdirname, ...], [filename, filename, ...])
      as strings
    """
    fs = get_filesystem(top)
    if hasattr(fs, "walk"):
        yield from fs.walk(top, topdown, onerror)
    else:
        top = fs.abspath(top)
        listing = fs.listdir(top)

        files = []
        subdirs = []
        for item in listing:
            full_path = fs.join(top, item)
            if fs.isdir(full_path):
                subdirs.append(item)
            else:
                files.append(item)

        here = (top, subdirs, files)

        if topdown:
            yield here

        for subdir in subdirs:
            joined_subdir = fs.join(top, subdir)
            for subitem in walk(joined_subdir, topdown, onerror=onerror):
                yield subitem

        if not topdown:
            yield here


def stat(filename):
    """Returns file statistics for a given path."""
    return get_filesystem(filename).stat(filename)


def read(file):
    with File(file, 'rb') as f:
        return f.read()
