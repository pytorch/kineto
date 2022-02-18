# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
import os
from abc import ABC, abstractmethod
from collections import namedtuple

# Data returned from the Stat call.
StatData = namedtuple('StatData', ['length'])


class BaseFileSystem(ABC):
    def support_append(self):
        return False

    def append(self, filename, file_content, binary_mode=False):
        pass

    def download_file(self, file_to_download, file_to_save):
        pass

    @abstractmethod
    def exists(self, filename):
        raise NotImplementedError

    @abstractmethod
    def read(self, file, binary_mode=False, size=None, continue_from=None):
        raise NotImplementedError

    @abstractmethod
    def write(self, filename, file_content, binary_mode=False):
        raise NotImplementedError

    @abstractmethod
    def glob(self, filename):
        raise NotImplementedError

    @abstractmethod
    def isdir(self, dirname):
        raise NotImplementedError

    @abstractmethod
    def listdir(self, dirname):
        raise NotImplementedError

    @abstractmethod
    def makedirs(self, path):
        raise NotImplementedError

    @abstractmethod
    def stat(self, filename):
        raise NotImplementedError


class BasePath(ABC):
    @abstractmethod
    def join(self, path, *paths):
        pass

    @abstractmethod
    def abspath(self, path):
        pass

    @abstractmethod
    def basename(self, path):
        pass

    @abstractmethod
    def relpath(self, path, start):
        pass


class LocalPath(BasePath):
    def abspath(self, path):
        return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    def basename(self, path):
        return os.path.basename(path)

    def relpath(self, path, start):
        return os.path.relpath(path, start)

    def join(self, path, *paths):
        return os.path.join(path, *paths)


class RemotePath(BasePath):
    def split(self, path):
        """Split a pathname.  Returns tuple '(head, tail)' where 'tail' is
        everything after the final slash.  Either part may be empty."""
        sep = '/'
        i = path.rfind(sep) + 1
        head, tail = path[:i], path[i:]
        head = head.rstrip(sep)
        return (head, tail)

    def join(self, path, *paths):
        """Join paths with a slash."""
        return '/'.join((path,) + paths)

    def abspath(self, path):
        return path

    def basename(self, path):
        return path.split('/')[-1]

    def relpath(self, path, start):
        if not path.startswith(start):
            return path
        start = start.rstrip('/')
        begin = len(start) + 1  # include the ending slash '/'
        return path[begin:]
