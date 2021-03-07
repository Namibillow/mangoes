# -*-coding:utf-8 -*
"""Utility functions to read and write data from/to text files

"""
import fileinput
import logging
import os
import os.path
from contextlib import contextmanager

import mangoes.utils
from mangoes.constants import ENCODING

logger = logging.getLogger(__name__)


def recursive_list_files(path):
    """Return list of base_path to the files present inside the directory (and all subdirectories')
    pointed to by 'base_path'.

    Parameters
    ----------
    path: str
        base_path to the folder to scan for files

    Returns
    -------
    list of str
    """
    files = []
    if os.path.isfile(path):
        files.append(path)
    else:
        for entry in sorted(os.listdir(path)):
            if not entry.startswith("."):
                a_file_path = os.path.join(path, entry)
                if os.path.isfile(a_file_path):
                    files.append(a_file_path)
                else:
                    files.extend(recursive_list_files(a_file_path))
    return files


@contextmanager
def get_reader(source):
    """Context manager to get a reader to a source

    Parameters
    ----------
    source : str or Iterable
        If the source is a path to a file, an archive or a directory, yields a reader to this(these) file(s)
        If the source is an iterable, yields it

    Yields
    ------
    Iterable
    """
    def hook_compressed_encoded_text(filename, mode='r', encoding=ENCODING):
        ext = os.path.splitext(filename)[1]
        if ext == '.gz':
            import gzip
            return gzip.open(filename, mode + 't', encoding=encoding)
        else:
            return open(filename, mode, encoding=encoding)

    if isinstance(source, str):
        if not os.path.exists(source):
            raise mangoes.utils.exceptions.ResourceNotFound(path=source)

        try:
            file_path_list = sorted(recursive_list_files(source))
            reader = fileinput.input(file_path_list, mode="r", openhook=hook_compressed_encoded_text)
            yield reader
        finally:
            try:
                reader.close()
            except UnboundLocalError:
                pass  # reader has not been opened yet
    else:
        yield source
