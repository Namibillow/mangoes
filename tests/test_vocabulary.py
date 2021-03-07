# -*- coding: utf-8 -*-
import logging
from collections import Counter

import mangoes
import mangoes.corpus
import mangoes.utils.exceptions
import pytest

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Unit tests

source = {"aa": 0, "bb": 1}
source_list = ["aa", "bb"]


@pytest.mark.unittest
def test_vocabulary_from_dict():
    vocabulary = mangoes.Vocabulary(source)
    assert vocabulary.word_index == source
    assert vocabulary.words == source_list


@pytest.mark.unittest
def test_vocabulary_from_list():
    vocabulary = mangoes.Vocabulary(source_list)
    assert vocabulary.word_index == source
    assert vocabulary.words == source_list


@pytest.mark.unittest
def test_vocabulary_from_vocabulary():
    vocabulary = mangoes.Vocabulary(source)
    actual = mangoes.Vocabulary(vocabulary)
    assert actual.word_index == source
    assert actual.words == source_list


@pytest.mark.unittest
def test_sentence_to_indices():
    sentence = ["a", "b", "c", "a", "b"]
    vocabulary = mangoes.Vocabulary(['z', 'a', 'b'])

    assert [1, 2, -1, 1, 2] == vocabulary.indices(sentence)


def test_extend():
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c'])
    vocabulary.extend(['a', 'd', 'c', 'e'])

    assert mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e']) == vocabulary


def test_extend_not_inplace():
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c'])
    merged = vocabulary.extend(['a', 'd', 'c', 'e'], inplace=False)

    assert mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e']) == merged
    assert merged is not vocabulary


def test_extend_return_map():
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c'])
    merged, mapping = vocabulary.extend(['a', 'd', 'b', 'e'], return_map=True)

    assert mangoes.Vocabulary(['a', 'b', 'c', 'd', 'e']) == vocabulary
    assert {0:0, 1:3, 2:1, 3:4} == mapping


# ###########################################################################################
# ### Integration tests

# Persistence
@pytest.mark.integration
def test_save_load(save_temp_dir):
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c'], language="xx")
    vocabulary.save(str(save_temp_dir), "vocabulary_test")

    loaded_vocabulary = mangoes.Vocabulary.load(str(save_temp_dir), "vocabulary_test")

    assert vocabulary == loaded_vocabulary


@pytest.mark.integration
def test_save_load_lemma(save_temp_dir):
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c'], language="xx", entity="lemma")
    vocabulary.save(str(save_temp_dir), "vocabulary_test")

    loaded_vocabulary = mangoes.Vocabulary.load(str(save_temp_dir), "vocabulary_test")

    assert vocabulary == loaded_vocabulary


@pytest.mark.integration
def test_save_load_tokens(save_temp_dir):
    import collections
    Token = collections.namedtuple('Token', 'lemma POS')

    vocabulary = mangoes.Vocabulary([Token('a', 'A'),
                                     Token('b', 'B'),
                                     Token('c', 'C')],
                                    entity=('lemma', 'POS'),
                                    language="xx")
    vocabulary.save(str(save_temp_dir), "vocabulary_test")

    loaded_vocabulary = mangoes.Vocabulary.load(str(save_temp_dir), "vocabulary_test")

    assert vocabulary.words == loaded_vocabulary.words
    assert vocabulary == loaded_vocabulary


@pytest.mark.integration
def test_save_load_bigrams(save_temp_dir):
    vocabulary = mangoes.Vocabulary(['a', 'b', 'c', 'a b'], language="xx")
    vocabulary.save(str(save_temp_dir), "vocabulary_test")

    loaded_vocabulary = mangoes.Vocabulary.load(str(save_temp_dir), "vocabulary_test")

    assert vocabulary == loaded_vocabulary


@pytest.mark.integration
def test_save_load_tokens_bigrams(save_temp_dir):
    import collections
    Token = collections.namedtuple('Token', 'lemma POS')

    vocabulary = mangoes.Vocabulary([Token('a', 'A'),
                                     Token('b', 'B'),
                                     Token('c', 'C'),
                                     (Token('a', 'A'), Token('b', 'B'))],
                                    entity=('lemma', 'POS'),
                                    language="xx")
    vocabulary.save(str(save_temp_dir), "vocabulary_test")

    loaded_vocabulary = mangoes.Vocabulary.load(str(save_temp_dir), "vocabulary_test")

    assert vocabulary == loaded_vocabulary

########################
# Exceptions
def test_exception_negative_frequencies():
    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        mangoes.corpus.remove_least_frequent(-1, Counter())

    with pytest.raises(mangoes.utils.exceptions.NotAllowedValue):
        mangoes.corpus.remove_most_frequent(-1, Counter())


def test_exception_unsupported_source():
    with pytest.raises(mangoes.utils.exceptions.UnsupportedType):
        mangoes.Vocabulary(1)
