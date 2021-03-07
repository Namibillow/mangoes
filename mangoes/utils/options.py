# -*- coding: utf-8 -*-
"""Class and functions to add optional features (progress bars, ...).

"""
import logging

logger = logging.getLogger(__name__)


def _in_notebook():
    """
    Returns True if the module is running in a Jupyter notebook, False if in IPython shell or other Python shell.
    """
    import sys
    return 'ipykernel' in sys.modules


class _NoProgressBar:
    """Default class to use if no progress bar library is available"""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def update(self):
        pass


try:
    if _in_notebook():
        from tqdm import tqdm_notebook

        ProgressBar = tqdm_notebook
    else:
        from tqdm import tqdm

        ProgressBar = tqdm
except ImportError:
    logger.info("Install tqdm to add progress bars")
    ProgressBar = _NoProgressBar
