# -*- coding: utf-8 -*-

import logging
import os.path
from collections import Counter

import pytest

import mangoes
import mangoes.corpus
import mangoes.utils.exceptions

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Mocks

class DummyRawReader:
    def __init__(self, source, lower=False, *args, **kwargs):
        self.source = source
        self.lower = lower
        self.annotated = False

    def sentences(self):
        if self.lower:
            for sentence in self.source:
                yield sentence.lower().split()
        else:
            for sentence in self.source:
                yield sentence.split()


class DummyAnnotatedReader:
    def __init__(self, source, lower=False, *args, **kwargs):
        self.source = source
        self.annotated = True

    def sentences(self):
        yield from self.source


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_raw_corpus(raw_source_string, raw_sentences):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader)

    assert 6 == corpus.nb_sentences
    assert 30 == corpus.size
    assert 15 == len(corpus.words_count)
    expected_word_count = {
        'is': 6, 'better': 6, 'than': 6,
        'Beautiful': 1, 'ugly': 1, 'Explicit': 1, 'implicit': 1,
        'Simple': 1, 'complex': 1, 'Complex': 1, 'complicated': 1,
        'Flat': 1, 'nested': 1, 'Sparse': 1, 'dense': 1}

    assert expected_word_count == corpus.words_count

    for expected, sentence in zip(raw_sentences, corpus):
        assert expected == sentence

    assert 14 == len(corpus.bigrams_count)
    expected_bigrams_count = {
        ('is', 'better'): 6, ('better', 'than'):6,
        ('Beautiful', 'is'): 1, ('than', 'ugly'): 1, ('Explicit', 'is'): 1, ('than', 'implicit'): 1,
        ('Simple', 'is'): 1, ('than', 'complex'): 1, ('Complex', 'is'): 1, ('than', 'complicated'): 1,
        ('Flat', 'is'): 1, ('than', 'nested'): 1, ('Sparse', 'is'): 1, ('than', 'dense'): 1}
    assert expected_bigrams_count == corpus.bigrams_count


@pytest.mark.unittest
def test_raw_corpus_lower(raw_source_string, raw_sentences):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader, lower=True)

    assert 6 == corpus.nb_sentences
    assert 30 == corpus.size
    assert 14 == len(corpus.words_count)
    expected_word_count = {
        'is': 6, 'better': 6, 'than': 6,
        'beautiful': 1, 'ugly': 1, 'explicit': 1, 'implicit': 1,
        'simple': 1, 'complex': 2, 'complicated': 1,
        'flat': 1, 'nested': 1, 'sparse': 1, 'dense': 1}

    assert expected_word_count == corpus.words_count

    for expected, sentence in zip(raw_sentences, corpus):
        assert [token.lower() for token in expected] == sentence


@pytest.mark.unittest
def test_annotated_corpus(annotated_sentences):
    corpus = mangoes.Corpus(annotated_sentences, reader=DummyAnnotatedReader)

    assert 6 == corpus.nb_sentences
    assert 30 == corpus.size
    assert 15 == len(corpus.words_count)

    expected_word_count = {('is', 'be', 'VBZ'): 6, ('than', 'than', 'IN'): 6,
                           ('better', 'better', 'JJR'): 6,
                           ('Beautiful', 'beautiful', 'JJ'): 1, ('ugly', 'ugly', 'JJ'): 1,
                           ('Explicit', 'Explicit', 'NNP'): 1, ('implicit', 'implicit', 'JJ'): 1,
                           ('Simple', 'simple', 'NN'): 1, ('complex', 'complex', 'JJ'): 1,
                           ('Complex', 'complex', 'NN'): 1, ('complicated', 'complicate', 'VBN'): 1,
                           ('Flat', 'Flat', 'NNP'): 1, ('nested', 'nested', 'JJ'): 1,
                           ('Sparse', 'Sparse', 'NNP'): 1, ('dense', 'dense', 'JJ'): 1}

    assert Counter(expected_word_count) == corpus.words_count


@pytest.mark.integration
def test_annotated_corpus_brown_lower(brown_source_string):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)

    assert 6 == corpus.nb_sentences
    assert 30 == corpus.size
    assert 15 == len(corpus.words_count)

    expected_word_count = {('is', 'be', 'VBZ'): 6, ('than', 'than', 'IN'): 6,
                           ('better', 'better', 'JJR'): 6,
                           ('beautiful', 'beautiful', 'JJ'): 1, ('ugly', 'ugly', 'JJ'): 1,
                           ('explicit', 'Explicit', 'NNP'): 1, ('implicit', 'implicit', 'JJ'): 1,
                           ('simple', 'simple', 'NN'): 1, ('complex', 'complex', 'JJ'): 1,
                           ('complex', 'complex', 'NN'): 1, ('complicated', 'complicate', 'VBN'): 1,
                           ('flat', 'Flat', 'NNP'): 1, ('nested', 'nested', 'JJ'): 1,
                           ('sparse', 'Sparse', 'NNP'): 1, ('dense', 'dense', 'JJ'): 1}

    assert Counter(expected_word_count) == corpus.words_count


@pytest.mark.integration
def test_annotated_corpus_brown_digit():
    source = ["a/A/a 12/NUM/12 b/B/b 14.5/NUM/14.5 one/NUM/one"]
    corpus = mangoes.Corpus(source, reader=mangoes.corpus.BROWN, digit=True)

    assert 1 == corpus.nb_sentences
    assert 5 == corpus.size
    assert 4 == len(corpus.words_count)

    expected_word_count = {('a', 'a', 'A'): 1, ('b', 'b', 'B'): 1,
                           ('0', '0', 'NUM'): 2, ('one', 'one', 'NUM'): 1}

    assert Counter(expected_word_count) == corpus.words_count

@pytest.mark.integration
def test_annotated_corpus_conll_lower(conll_source_string):
    corpus = mangoes.Corpus(conll_source_string, reader=mangoes.corpus.CONLL, lower=True)

    assert 6 == corpus.nb_sentences
    assert 30 == corpus.size
    assert 15 == len(corpus.words_count)

    expected_word_count = {('is', 'be', 'VBZ'): 6, ('than', 'than', 'IN'): 6,
                           ('better', 'better', 'JJR'): 6,
                           ('beautiful', 'beautiful', 'JJ'): 1, ('ugly', 'ugly', 'JJ'): 1,
                           ('explicit', 'Explicit', 'NNP'): 1, ('implicit', 'implicit', 'JJ'): 1,
                           ('simple', 'simple', 'NN'): 1, ('complex', 'complex', 'JJ'): 1,
                           ('complex', 'complex', 'NN'): 1, ('complicated', 'complicate', 'VBN'): 1,
                           ('flat', 'Flat', 'NNP'): 1, ('nested', 'nested', 'JJ'): 1,
                           ('sparse', 'Sparse', 'NNP'): 1, ('dense', 'dense', 'JJ'): 1}

    assert Counter(expected_word_count) == corpus.words_count


@pytest.mark.unittest
def test_lazy_sentences_count_when_words_count(raw_source_string):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader, lazy=True)
    assert corpus.nb_sentences is None

    corpus.words_count
    assert 6 == corpus.nb_sentences


@pytest.mark.unittest
def test_vocabulary_from_corpus(raw_source_string):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader)
    vocabulary = corpus.create_vocabulary()
    assert {'is', 'better', 'than', 'Beautiful', 'ugly', 'Explicit', 'implicit', 'Simple', 'complex',
            'Complex', 'complicated', 'Flat', 'nested', 'Sparse', 'dense'} == set(vocabulary.words)


@pytest.mark.unittest
def test_vocabulary_from_corpus_is_sorted(raw_source_string):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader)
    vocabulary = corpus.create_vocabulary()
    assert {'is', 'better', 'than', 'Beautiful', 'ugly', 'Explicit', 'implicit', 'Simple', 'complex',
            'Complex', 'complicated', 'Flat', 'nested', 'Sparse', 'dense'} == set(vocabulary.words)

    assert {'is', 'better', 'than'} == set(vocabulary.words[:3])
    assert {'Beautiful', 'ugly', 'Explicit', 'implicit', 'Simple', 'complex', 'Complex', 'complicated',
            'Flat', 'nested', 'Sparse', 'dense'} == set(vocabulary.words[3:])


@pytest.mark.unittest
def test_vocabulary_from_corpus_with_dummy_filter(raw_source_string):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader, lower=True)

    def dummy_filter(counter):
        return {"spam": 0}

    vocabulary = corpus.create_vocabulary(filters=[dummy_filter])
    assert set(vocabulary.word_index) == {'spam'}


@pytest.mark.unittest
def test_vocabulary_of_entities_from_annotated_corpus(annotated_sentences):
    corpus = mangoes.Corpus(annotated_sentences, reader=DummyAnnotatedReader)

    vocabulary = corpus.create_vocabulary()
    assert {('is', 'be', 'VBZ'), ('better', 'better', 'JJR'), ('than', 'than', 'IN'),
            ('Beautiful', 'beautiful', 'JJ'), ('ugly', 'ugly', 'JJ'),
            ('Explicit', 'Explicit', 'NNP'), ('implicit', 'implicit', 'JJ'),
            ('Simple', 'simple', 'NN'), ('complex', 'complex', 'JJ'),
            ('Complex', 'complex', 'NN'), ('complicated', 'complicate', 'VBN'),
            ('Flat', 'Flat', 'NNP'), ('nested', 'nested', 'JJ'),
            ('Sparse', 'Sparse', 'NNP'), ('dense', 'dense', 'JJ')} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes="form")
    assert {'is', 'better', 'than',
            'Beautiful', 'ugly',
            'Explicit', 'implicit',
            'Simple', 'complex',
            'Complex', 'complicated',
            'Flat', 'nested',
            'Sparse', 'dense'} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes=("POS", "lemma"))
    assert {('VBZ', 'be'), ('JJR', 'better'), ('IN', 'than'),
            ('JJ', 'beautiful'), ('JJ', 'ugly'),
            ('NNP', 'Explicit'), ('JJ', 'implicit'),
            ('NN', 'simple'), ('JJ', 'complex'),
            ('NN', 'complex'), ('VBN', 'complicate'),
            ('NNP', 'Flat'), ('JJ', 'nested'),
            ('NNP', 'Sparse'), ('JJ', 'dense')} == set(vocabulary.words)


# Persistence
@pytest.mark.unittest
def test_save_load_metadata_corpus(tmpdir_factory, raw_source_string):
    # TODO : mock save in file
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader)
    path = str(tmpdir_factory.mktemp('data').join('.corpus'))
    corpus.save_metadata(path)

    assert os.path.isfile(path)

    loaded_corpus = mangoes.Corpus.load_from_metadata(path)

    assert corpus.content == loaded_corpus.content
    assert corpus.nb_sentences == loaded_corpus.nb_sentences
    assert corpus.words_count == loaded_corpus.words_count
    assert corpus.size == loaded_corpus.size


# ###########################################################################################
# ### Integration tests

# ## Integration with SentenceGenerators
# raw text
@pytest.mark.integration
def test_raw_corpus_from_list_of_strings(raw_source_string, raw_sentences):
    corpus = mangoes.Corpus(raw_source_string)
    for i, sentence in enumerate(corpus):
        assert raw_sentences[i] == sentence


@pytest.mark.integration
def test_raw_corpus_lower_from_list_of_strings(raw_source_string, raw_sentences_lowered):
    corpus = mangoes.Corpus(raw_source_string, lower=True)
    for i, sentence in enumerate(corpus):
        assert raw_sentences_lowered[i] == sentence


@pytest.mark.integration
def test_raw_corpus_from_file(raw_source_file, raw_sentences):
    corpus = mangoes.Corpus(raw_source_file)
    for i, sentence in enumerate(corpus):
        assert raw_sentences[i] == sentence


@pytest.mark.integration
def test_raw_corpus_lower_from_list_of_strings(raw_source_file, raw_sentences_lowered):
    corpus = mangoes.Corpus(raw_source_file, lower=True)
    for i, sentence in enumerate(corpus):
        assert raw_sentences_lowered[i] == sentence


@pytest.mark.integration
def test_raw_corpus_from_dir(raw_source_dir, raw_sentences):
    corpus = mangoes.Corpus(raw_source_dir)
    for i, sentence in enumerate(corpus):
        assert raw_sentences[i] == sentence


def test_raw_corpus_ignore_punctuation():
    source = ['a b , c d .']
    corpus = mangoes.Corpus(source, ignore_punctuation=True)
    sentence = corpus.__iter__().__next__()
    assert ['a', 'b', 'c', 'd'] == sentence


def test_raw_corpus_replace_digit():
    source = ['a 0 20 b 65.4']
    corpus = mangoes.Corpus(source, digit=True)
    sentence = corpus.__iter__().__next__()
    assert ['a', '0', '0', 'b', '0'] == sentence


# Annotated text : 3 formats : brown, xml or conll
# XML
@pytest.mark.integration
def test_corpus_from_xml_string(xml_source_string, fully_annotated_sentences):
    corpus = mangoes.Corpus(xml_source_string, reader=mangoes.corpus.XML)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_corpus_from_xml_file(xml_source_file, fully_annotated_sentences):
    corpus = mangoes.Corpus(xml_source_file, reader=mangoes.corpus.XML)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_corpus_from_xml_dir(xml_source_dir, fully_annotated_sentences):
    corpus = mangoes.Corpus(xml_source_dir, reader=mangoes.corpus.XML)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence


# BROWN
@pytest.mark.integration
def test_corpus_from_brown_string(brown_source_string, annotated_sentences):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN)
    for i, sentence in enumerate(corpus):
        assert annotated_sentences[i] == sentence


@pytest.mark.integration
def test_corpus_from_brown_file(brown_source_file, annotated_sentences):
    corpus = mangoes.Corpus(brown_source_file, reader=mangoes.corpus.BROWN)
    for i, sentence in enumerate(corpus):
        assert annotated_sentences[i] == sentence


@pytest.mark.integration
def test_corpus_from_brown_dir(brown_source_dir, annotated_sentences):
    corpus = mangoes.Corpus(brown_source_dir, reader=mangoes.corpus.BROWN)
    for i, sentence in enumerate(corpus):
        assert annotated_sentences[i] == sentence


# CONLL
def test_corpus_from_conll_string(conll_source_string, fully_annotated_sentences):
    corpus = mangoes.Corpus(conll_source_string, reader=mangoes.corpus.CONLL)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence


def test_corpus_from_conll_file(conll_source_file, fully_annotated_sentences):
    corpus = mangoes.Corpus(conll_source_file, reader=mangoes.corpus.CONLL)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence


def test_corpus_from_conll_dir(conll_source_dir, fully_annotated_sentences):
    corpus = mangoes.Corpus(conll_source_dir, reader=mangoes.corpus.CONLL)
    for i, sentence in enumerate(corpus):
        assert fully_annotated_sentences[i] == sentence

# ## Integration with vocabulary filters
@pytest.mark.integration
def test_vocabulary_from_corpus_with_filters(raw_source_string):
    corpus = mangoes.Corpus(raw_source_string, reader=DummyRawReader, lower=True)

    min_frequency = 2
    max_frequency = 4
    max_nb = 5

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.truncate(max_nb)])
    assert len(vocabulary.words) == 5

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_least_frequent(min_frequency)])
    assert set(vocabulary.words) == {'is', 'better', 'than', 'complex'}

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_most_frequent(max_frequency),
                                                   mangoes.corpus.remove_least_frequent(min_frequency)])
    assert set(vocabulary.words) == {'complex'}

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_least_frequent(min_frequency),
                                                   mangoes.corpus.remove_elements(['is'])])
    assert set(vocabulary.words) == {'better', 'than', 'complex'}


@pytest.mark.integration
def test_vocabulary_of_entities_from_annotated_corpus_with_filters(annotated_sentences):
    corpus = mangoes.Corpus(annotated_sentences, reader=DummyAnnotatedReader)

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_least_frequent(2)])
    assert {('is', 'be', 'VBZ'), ('better', 'better', 'JJR'), ('than', 'than', 'IN')} == set(
        vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes="form", filters=[mangoes.corpus.remove_least_frequent(2)])
    assert {'is', 'better', 'than'} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes="lemma", filters=[mangoes.corpus.remove_least_frequent(2)])
    assert {'be', 'better', 'than', 'complex'} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes=("POS", "lemma"), filters=[mangoes.corpus.remove_least_frequent(2)])
    assert {('VBZ', 'be'), ('JJR', 'better'), ('IN', 'than')} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(attributes=("POS", "lemma"),
                                          filters=[mangoes.corpus.remove_least_frequent(2),
                                                   mangoes.corpus.remove_elements("be", attribute="lemma")])
    assert {('JJR', 'better'), ('IN', 'than')} == set(vocabulary.words)


@pytest.mark.integration
def test_vocabulary_of_entities_filtered_by_attributes(annotated_sentences):
    corpus = mangoes.Corpus(annotated_sentences, reader=DummyAnnotatedReader)

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.filter_by_attribute('POS', 'NN')])
    assert {('Simple', 'simple', 'NN'), ('Complex', 'complex', 'NN')} == set(vocabulary.words)

    vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.filter_by_attribute('POS', 'NN')],
                                          attributes=('POS', 'lemma'))
    assert {('NN', 'simple'), ('NN', 'complex')} == set(vocabulary.words)


# ## Persistence
@pytest.mark.integration
def test_save_load_metadata_corpus_file(raw_source_file):
    corpus = mangoes.Corpus(raw_source_file, lower=True)
    path = os.path.join(os.path.dirname(raw_source_file), '.corpus')
    corpus.save_metadata(path)

    assert os.path.isfile(path)

    loaded_corpus = mangoes.Corpus.load_from_metadata(path)

    assert corpus.content == loaded_corpus.content
    assert corpus.nb_sentences == loaded_corpus.nb_sentences
    assert corpus.words_count == loaded_corpus.words_count
    assert corpus.size == loaded_corpus.size
    assert corpus.reader.transform.__name__ == loaded_corpus.reader.transform.__name__


@pytest.mark.integration
def test_save_load_metadata_annotated_corpus_file(xml_source_file):
    corpus = mangoes.Corpus(xml_source_file, reader=mangoes.corpus.XML)
    path = os.path.join(os.path.dirname(xml_source_file), '.corpus')
    corpus.save_metadata(path)

    assert os.path.isfile(path)

    loaded_corpus = mangoes.Corpus.load_from_metadata(path)

    assert corpus.content == loaded_corpus.content
    assert corpus.nb_sentences == loaded_corpus.nb_sentences
    assert corpus.words_count == loaded_corpus.words_count
    assert corpus.size == loaded_corpus.size


########################
# Exceptions
@pytest.mark.skip(reason="should raise specific exception")
def test_exception_unknown_format():
    with pytest.raises(AttributeError) as exc:
        corpus = mangoes.Corpus(RAW_SOURCE, reader=print)
        for _ in corpus:
            pass
    assert "'NoneType' object has no attribute 'sentences'" == exc.value.args[0]


def test_exception_wrong_path():
    with pytest.raises(mangoes.utils.exceptions.ResourceNotFound) as exc:
        corpus = mangoes.Corpus("not_existing_dir", lazy=True)
        for _ in corpus:
            pass
    assert "Resource 'not_existing_dir' does not exist" == exc.value.args[0]
