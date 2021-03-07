# -*- coding: utf-8 -*-

import pytest

import logging
import mangoes.utils.multiproc

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


def test_data_parallel():
    source = ["a b c d", "e f g", "h i j k l m", "n o p q", "r s t u v w", "x y z"]

    def count_words(sentences):
        result = 0
        for sentence in sentences:
            result += len(sentence.split())
        yield result

    def add(value_1, value_2):
        return value_1 + value_2

    parallel = mangoes.utils.multiproc.DataParallel(count_words, add, 1)
    assert 26 == parallel.run(source)

    parallel = mangoes.utils.multiproc.DataParallel(count_words, add, 4)
    assert 26 == parallel.run(source)
