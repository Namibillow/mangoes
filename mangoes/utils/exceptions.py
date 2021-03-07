# -*-coding:utf-8 -*
"""Set of Mangoes's Exceptions

"""
import logging

logger = logging.getLogger(__name__)


class MangoesError(Exception):
    """Basic exception for errors raised by mangoes"""

    def __init__(self, msg, original_exception=None):
        if original_exception:
            msg += " : " + original_exception
        super(MangoesError, self).__init__("{}".format(msg))
        self.original_exception = original_exception


class NotAllowedValue(MangoesError, ValueError):
    """Raise when a value is not allowed"""

    def __init__(self, value=None, allowed_values=None, msg=None):
        if msg is None:
            msg = ""
            if value:
                msg += "'{}' is not allowed. ".format(value)
            if allowed_values:
                msg += "Allowed values are {}".format(", ".join(["'{}'".format(v) for v in allowed_values]))
        super(NotAllowedValue, self).__init__(msg)


class IncompatibleValue(MangoesError, ValueError):
    """Raise when 2 values are not compatible"""


class UnsupportedType(MangoesError, TypeError):
    """Raise when the type of an argument is not supported"""


class ResourceNotFound(MangoesError, FileNotFoundError):
    """Raise when an expected resource is not found on a given path"""

    def __init__(self, path=None, msg=None):
        if msg is None:
            msg = "Resource '{}' does not exist".format(path)
        super(ResourceNotFound, self).__init__(msg)


class RequiredValue(MangoesError):  # TODO : or missing value ?
    """Raise when a required argument is empty"""

class RuntimeError(MangoesError, RuntimeError):
    """Raised when an error is detected that doesn’t fall in any of the other categories.
    The associated value is a string indicating what precisely went wrong."""

class OutOfVocabulary(MangoesError, ValueError):
    """Raise when a word is not in a vocabulary"""

    def __init__(self, value=None, msg=None):
        if msg is None:
            msg = ""
            if value:
                msg += "'{}' is out of vocabulary. ".format(value)
        super(OutOfVocabulary, self).__init__(msg)
