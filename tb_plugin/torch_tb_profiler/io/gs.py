# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
from google.cloud import storage
from google.auth import exceptions

from .. import utils
from .base import BaseFileSystem, RemotePath, StatData

logger = utils.get_logger()


class GoogleBlobSystem(RemotePath, BaseFileSystem):
    """Provides filesystem access to S3."""

    def __init__(self):
        if not storage:
            raise ImportError('google-cloud-storage must be installed for Google Cloud Blob support.')

    def exists(self, dirname):
        """Returns whether the path is a directory or not."""
        bucket_name, path = self.bucket_and_path(dirname)
        client = self.create_google_cloud_client()
        bucket = client.bucket(bucket_name)
        return bucket.blob(path).exists()

    def read(self, filename, binary_mode=False, size=None, continue_from=None):
        raise NotImplementedError

    def write(self, filename, file_content, binary_mode=False):
        raise NotImplementedError

    def glob(self, filename):
        raise NotImplementedError

    def download_file(self, file_to_download, file_to_save):
        bucket_name, path = self.bucket_and_path(file_to_download)
        client = self.create_google_cloud_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(path)
        blob.download_to_filename(file_to_save)

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
        bucket_name, path = self.bucket_and_path(dirname)
        client = self.create_google_cloud_client()
        blobs = client.list_blobs(bucket_name, prefix=path)
        items = []
        for blob in blobs:
            item = self.relpath(blob.name, path)
            if items not in items:
                items.append(item)
        return items

    def makedirs(self, dirname):
        """No need create directory since the upload blob will automatically create"""
        pass

    def stat(self, filename):
        """Returns file statistics for a given path."""
        bucket_name, path = self.bucket_and_path(filename)
        client = self.create_google_cloud_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.get_blob(path)
        return StatData(blob.size)

    def walk(self, top, topdown=True, onerror=None):
        bucket_name, path = self.bucket_and_path(top)
        client = self.create_google_cloud_client()
        blobs = client.list_blobs(bucket_name, prefix=path)
        results = {}
        for blob in blobs:
            dirname, basename = self.split(blob.name)
            dirname = 'gs://{}/{}'.format(bucket_name, dirname)
            results.setdefault(dirname, []).append(basename)
        for key, value in results.items():
            yield key, None, value

    def split_blob_path(self, blob_path):
        """ Find the first blob start with blob_path, then get the relative path starting from dirname(blob_path).
        Finally, split the relative path.
        return (basename(blob_path), [relative splitted paths])
        If blob_path doesn't exist, return (None, None)
        For example,
            For blob gs://tests/test1/test2/test.txt
            * If the blob_path is '', return ('', [test1, test2, test.txt])
            * If the blob_path is test1, return (test1, [test2, test.txt])
            * If the blob_path is test1/test2, return (test2, [test2, test.txt])
            * If the blob_path is test1/test2/test.txt, return (test.txt, [test.txt])
        """
        bucket_name, path = self.bucket_and_path(blob_path)
        client = self.create_google_cloud_client()
        blobs = client.list_blobs(bucket_name, prefix=path, delimiter=None, max_results=1)

        for blob in blobs:
            dir_path, basename = self.split(path)
            if dir_path:
                rel_path = blob.name[len(dir_path):]
                parts = rel_path.lstrip('/').split('/')
            else:
                parts = blob.name.split('/')
            return (basename, parts)
        return (None, None)

    def bucket_and_path(self, url):
        """Split an S3-prefixed URL into bucket and path."""
        if url.startswith('gs://'):
            url = url[len('gs://'):]
        idx = url.index('/')
        bucket = url[:idx]
        path = url[(idx + 1):]
        return bucket, path

    def create_google_cloud_client(self):
        try:
            client = storage.Client()
            logger.debug('Using default Google Cloud credentials.')
        except exceptions.DefaultCredentialsError:
            client = storage.Client.create_anonymous_client()
            logger.debug(
                'Default Google Cloud credentials not available. '
                'Falling back to anonymous credentials.')
        return client
