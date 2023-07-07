# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import os

from azure.storage.blob import ContainerClient

from .. import utils
from .base import BaseFileSystem, RemotePath, StatData
from .utils import as_bytes, as_text, parse_blob_url

logger = utils.get_logger()


class AzureBlobSystem(RemotePath, BaseFileSystem):
    """Provides filesystem access to S3."""

    def __init__(self):
        if not ContainerClient:
            raise ImportError('azure-storage-blob must be installed for Azure Blob support.')
        self.connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', None)

    def exists(self, dirname):
        """Returns whether the path is a directory or not."""
        basename, parts = self.split_blob_path(dirname)
        if basename is None or parts is None:
            return False
        if basename == '':
            # root container case
            return True
        else:
            return basename == parts[0]

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        """Reads contents of a file to a string."""
        logger.info('azure blob: starting reading file %s' % filename)
        account, container, path = self.container_and_path(filename)
        client = self.create_container_client(account, container)
        blob_client = client.get_blob_client(path)
        if not blob_client.exists():
            raise FileNotFoundError("file %s doesn't exist!" % path)

        downloader = blob_client.download_blob(offset=continue_from, length=size)
        if continue_from is not None:
            continuation_token = continue_from + downloader.size
        else:
            continuation_token = downloader.size

        data = downloader.readall()
        logger.info('azure blob: file %s download is done, size is %d' % (filename, len(data)))
        if binary_mode:
            return as_bytes(data), continuation_token
        else:
            return as_text(data), continuation_token

    def write(self, filename, file_content, binary_mode=False):
        """Writes string file contents to a file."""
        account, container, path = self.container_and_path(filename)
        client = self.create_container_client(account, container)

        if binary_mode:
            if not isinstance(file_content, bytes):
                raise TypeError('File content type must be bytes')
        else:
            file_content = as_bytes(file_content)
        client.upload_blob(path, file_content)

    def download_file(self, file_to_download, file_to_save):
        logger.info('azure blob: starting downloading file %s as %s' % (file_to_download, file_to_save))
        account, container, path = self.container_and_path(file_to_download)
        client = self.create_container_client(account, container)
        blob_client = client.get_blob_client(path)
        if not blob_client.exists():
            raise FileNotFoundError("file %s doesn't exist!" % path)

        downloader = blob_client.download_blob()
        with open(file_to_save, 'wb') as downloaded_file:
            data = downloader.readall()
            downloaded_file.write(data)
            logger.info('azure blob: file %s is downloaded as %s, size is %d' %
                        (file_to_download, file_to_save, len(data)))

    def glob(self, filename):
        """Returns a list of files that match the given pattern(s)."""
        # Only support prefix with * at the end and no ? in the string
        star_i = filename.find('*')
        quest_i = filename.find('?')
        if quest_i >= 0:
            raise NotImplementedError(
                '{} not supported by compat glob'.format(filename)
            )
        if star_i != len(filename) - 1:
            return []

        filename = filename[:-1]

        account, container, path = self.container_and_path(filename)
        client = self.create_container_client(account, container)
        blobs = client.list_blobs(name_starts_with=path)
        return [blob.name for blob in blobs]

    def isdir(self, dirname):
        """Returns whether the path is a directory or not."""
        basename, parts = self.split_blob_path(dirname)
        if basename is None or parts is None:
            return False
        if basename == '':
            # root container case
            return True
        else:
            return basename == parts[0] and len(parts) > 1

    def listdir(self, dirname):
        """Returns a list of entries contained within a directory."""
        account, container, path = self.container_and_path(dirname)
        client = self.create_container_client(account, container)
        blob_iter = client.list_blobs(name_starts_with=path)
        items = []
        for blob in blob_iter:
            item = self.relpath(blob.name, path)
            if items not in items:
                items.append(item)
        return items

    def makedirs(self, dirname):
        """No need create directory since the upload blob will automatically create"""
        pass

    def stat(self, filename):
        """Returns file statistics for a given path."""
        account, container, path = self.container_and_path(filename)
        client = self.create_container_client(account, container)
        blob_client = client.get_blob_client(path)
        props = blob_client.get_blob_properties()
        return StatData(props.size)

    def walk(self, top, topdown=True, onerror=None):
        account, container, path = self.container_and_path(top)
        client = self.create_container_client(account, container)
        blobs = client.list_blobs(name_starts_with=path)
        results = {}
        for blob in blobs:
            dirname, basename = self.split(blob.name)
            dirname = 'https://{}/{}/{}'.format(account, container, dirname)
            results.setdefault(dirname, []).append(basename)
        for key, value in results.items():
            yield key, None, value

    def split_blob_path(self, blob_path):
        """ Find the first blob start with blob_path, then get the relative path starting from dirname(blob_path).
        Finally, split the relative path.
        return (basename(blob_path), [relative splitted paths])
        If blob_path doesn't exist, return (None, None)
        For example,
            For blob https://trainingdaemon.blob.core.windows.net/tests/test1/test2/test.txt
            * If the blob_path is '', return ('', [test1, test2, test.txt])
            * If the blob_path is test1, return (test1, [test2, test.txt])
            * If the blob_path is test1/test2, return (test2, [test2, test.txt])
            * If the blob_path is test1/test2/test.txt, return (test.txt, [test.txt])
        """
        account, container, path = self.container_and_path(blob_path)
        client = self.create_container_client(account, container)
        blobs = client.list_blobs(name_starts_with=path, maxresults=1)

        for blob in blobs:
            dir_path, basename = self.split(path)
            if dir_path:
                rel_path = blob.name[len(dir_path):]
                parts = rel_path.lstrip('/').split('/')
            else:
                parts = blob.name.split('/')
            return (basename, parts)
        return (None, None)

    def container_and_path(self, url):
        """Split an Azure blob -prefixed URL into container and blob path."""
        root, parts = parse_blob_url(url)
        if len(parts) != 2:
            raise ValueError('Invalid azure blob url %s' % url)
        return root, parts[0], parts[1]

    def create_container_client(self, account, container):
        if self.connection_string:
            client = ContainerClient.from_connection_string(self.connection_string, container)
        else:
            client = ContainerClient.from_container_url('https://{}/{}'.format(account, container))
        return client
