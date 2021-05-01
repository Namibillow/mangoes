# -*- coding: utf-8 -*-

import collections
import logging
import math
import unittest.mock

import numpy as np
import pytest
import scipy

import mangoes.context
import mangoes.counting
from mangoes.corpus import Corpus
from mangoes.vocabulary import Vocabulary

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Mocks
class DummyVocabulary:  # TODO : should be renamed Fake instead of Dummy (like most of all other Dummies)
    def __init__(self, words, entity=None):
        self.words = words
        self.word_index = {word: index for index, word in enumerate(words)}
        self.entity = entity

        self.params = {"language": "en"}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.word_index

    def __eq__(self, other):
        return self.words == other.words

    def index(self, word):
        return self.word_index[word]

    def indices(self, sentence):
        return [self.word_index[word] if word in self.words else -1 for word in sentence]

    def __iter__(self):
        return self.words.__iter__()

    def entity_filter(self):
        def filter_sentence(sentence):
            return sentence
        return filter_sentence

    def get_bigrams(self):
        return None


class DummyContext:
    # window of one-one around the position
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.params = {}

    def __call__(self, sentence, mask=None):#, position):
        contexts = []
        for position in range(len(sentence)):
            before = [w for w in sentence[max(0, position - 1):position] if w in self.vocabulary]
            after = [w for w in sentence[position + 1:min(position + 1 + 1, len(sentence))] if w in self.vocabulary]
            contexts.append(before + after)
        return contexts

    def call_on_encoded(self, encoded_sentence):
        contexts = []
        for position in range(len(encoded_sentence)):
            before = [w for w in encoded_sentence[max(0, position - 1):position] if w >= 0]
            after = [w for w in encoded_sentence[position + 1:min(position + 1 + 1, len(encoded_sentence))] if w >= 0]
            contexts.append(before + after)
        return contexts

    def filter_sentence(self, sentence):
        return sentence


# ###########################################################################################
# ### Unit tests

# Corpus : (in conftest.py)
# Beautiful is better than ugly
# Explicit is better than implicit
# Simple is better than complex
# Complex is better than complicated
# Flat is better than nested
# Sparse is better than dense

#                 is   better    than
expected_count = [[1, 0, 0],  # beautiful
                  [0, 0, 1],  # ugly
                  [1, 0, 0],  # simple
                  [1, 0, 1],  # complex
                  [0, 0, 1]]  # complicated


@pytest.mark.unittest
def test_count_cooccurrence(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = DummyVocabulary(["is", "better", "than"])

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=DummyContext(context_words),
                                                 nb_workers=1)
    assert np.array_equiv(expected_count, result.matrix.toarray())

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=DummyContext(context_words),
                                                 nb_workers=4)
    assert np.array_equiv(expected_count, result.matrix.toarray())


@pytest.mark.integration
def test_count_cooccurrence_no_context_vocabulary(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=mangoes.context.Window(), nb_workers=1)

    assert (5, 2) == result.matrix.shape


@pytest.mark.unittest
def test_count_cooccurrence_no_context_vocabulary_parallel(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated", "is"])

    expected = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                   context=mangoes.context.Window())
    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=mangoes.context.Window(),
                                                 nb_workers=5)

    assert (6, 9) == result.matrix.shape

    assert set(expected.contexts_words.words) == set(result.contexts_words.words)
    assert {"is", "beautiful", "better", "than", "explicit", "simple", "complex",
            "flat", "sparse"} == set(result.contexts_words.words)
    np.testing.assert_array_equal(expected.matrix.sum(), result.matrix.sum())


@pytest.mark.integration
def test_cooccurrence_word_word(brown_source_string):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)
    words = Vocabulary(["beautiful", "ugly", "simple", "complex", "complicated"],
                       entity="form")
    context_words = Vocabulary(["is", "better", "than"],
                                entity="form")

    result = mangoes.counting.count_cooccurrence(corpus, words,
                                                 context=mangoes.context.Window(context_words),
                                                 nb_workers=1)

    assert np.array_equiv(expected_count, result.matrix.toarray())


@pytest.mark.integration
def test_cooccurrence_word_lemma(brown_source_string):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)
    words = mangoes.Vocabulary(["beautiful", "ugly", "simple", "complex", "complicated"],
                            entity="form")
    context_words = mangoes.Vocabulary(["be", "better", "than"], entity="lemma")

    result = mangoes.counting.count_cooccurrence(corpus, words,
                                                 context=mangoes.context.Window(context_words))

    assert np.array_equiv(expected_count, result.matrix.toarray())


@pytest.mark.integration
def test_cooccurrence_token_lemma(brown_source_string):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)
    words = mangoes.Vocabulary([('beautiful', 'beautiful', 'JJ'),
                             ('ugly', 'ugly', 'JJ'),
                             ('simple', 'simple', 'NN'),
                             ('complex', 'complex', 'JJ'),
                             ('complex', 'complex', 'NN'),
                             ('complicated', 'complicate', 'VBN')])#,
                            # entity=("word", "pos", "lemma"))
    context_words = mangoes.Vocabulary(["be", "better", "than"], entity="lemma")

    #                     is   better    than
    expected_count_alt = [[1, 0, 0],  # beautiful
                          [0, 0, 1],  # ugly
                          [1, 0, 0],  # simple
                          [0, 0, 1],  # complex/JJ
                          [1, 0, 0],  # complex/NN
                          [0, 0, 1]]  # complicated

    result = mangoes.counting.count_cooccurrence(corpus, words,
                                                 context=mangoes.context.Window(context_words),
                                                 nb_workers=1)

    assert np.array_equiv(expected_count_alt, result.matrix.toarray())


def test_cooccurrence_with_bigrams():
    corpus = mangoes.Corpus(["i love new york"])
    words = mangoes.Vocabulary(['i', 'love', ('new', 'york')])

    #                  i  love   new york
    expected_count = [[0,   1,     0],  # i
                      [1,   0,     1],  # love
                      [0,   1,     0]]  # new york

    result = mangoes.counting.count_cooccurrence(corpus, words, context=words, nb_workers=1)

    assert np.array_equiv(expected_count, result.matrix.toarray())


def test_cooccurrence_with_token_bigrams(brown_source_string):
    Token = collections.namedtuple('Token', ('form', 'lemma', 'POS'))
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)
    words = mangoes.Vocabulary([(Token('is', 'be', 'VBZ'), Token('better', 'better', 'JJR'))])

    context_words = mangoes.Vocabulary(["beautiful", "better", "than", "complex"], entity="lemma")

    #                   beautiful   better   than    complex
    expected_count = [[         1,      0,      6,      1], # is_better
                      ]

    result = mangoes.counting.count_cooccurrence(corpus, words, context=context_words, nb_workers=1)

    assert np.array_equiv(expected_count, result.matrix.toarray())

def test_cooccurrence_with_token_bigrams_2(brown_source_string):
    Token = collections.namedtuple('Token', ('lemma', 'POS'))

    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)
    words = mangoes.Vocabulary([(Token('be', 'VBZ'), Token('better', 'JJR'))], entity=("lemma", "POS"))

    context_words = mangoes.Vocabulary(["beautiful", "better", "than", "complex"], entity="lemma")

    #                   beautiful   better   than    complex
    expected_count = [[         1,      0,      6,      1]] # be_better

    result = mangoes.counting.count_cooccurrence(corpus, words, context=context_words, nb_workers=1)

    assert np.array_equiv(expected_count, result.matrix.toarray())


# parameters
@pytest.mark.unittest
def test_create_subsampler(dummy_raw_corpus):
    # frequencies of 'is', 'better' and 'than' = 6/30 = 0.2
    # frequencies of 'complex' = 2/30 = 0.0666
    # frequencies of others = 1/30

    # default : threshold = 10**-5
    result = mangoes.counting.create_subsampler(dummy_raw_corpus)

    assert dummy_raw_corpus.words_count.keys() == result.keys()

    assert 1 - math.sqrt(10 ** -5 / (6 / 30)) == result["is"] == result["better"] == result["than"]
    assert 1 - math.sqrt(10 ** -5 / (2 / 30)) == result["complex"]
    assert 1 - math.sqrt(10 ** -5 / (1 / 30)) == result["beautiful"]

    threshold = 0.05
    result = mangoes.counting.create_subsampler(dummy_raw_corpus, threshold=threshold)

    assert {"is", "better", "than", "complex"} == result.keys()

    assert 1 - math.sqrt(threshold / (6 / 30)) == result["is"] == result["better"] == result["than"]
    assert 1 - math.sqrt(threshold / (2 / 30)) == result["complex"]

    threshold = 0.125
    result = mangoes.counting.create_subsampler(dummy_raw_corpus, threshold=threshold)

    assert {"is", "better", "than"} == result.keys()

    expected_probability = 1 - math.sqrt(threshold / 0.2)
    assert expected_probability == result["is"] == result["better"] == result["than"]

    threshold = 0.5
    result = mangoes.counting.create_subsampler(dummy_raw_corpus, threshold=threshold)

    assert {} == result


@pytest.mark.unittest
def test_subsample():
    sentence = ["beautiful", "is", "better", "than", "ugly"]
    expected = [None, "is", "better", "than", "ugly"]

    subsampler = {"beautiful": 0.5, "is": 0.3}

    with unittest.mock.patch('random.random', return_value=0.4):
        assert expected == mangoes.counting._subsample(sentence, subsampler)


# ###########################################################################################
# ### Integration tests

# ## Integration with Corpus and Vocabulary

# raw text
@pytest.mark.integration
def test_count_cooccurrence_raw_text_from_string(raw_source_string):
    corpus = Corpus(raw_source_string, lower=True)
    words = Vocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = Vocabulary(["is", "better", "than"])

    result = mangoes.counting.count_cooccurrence(corpus, words, context_words)

    assert np.array_equiv(expected_count, result.matrix.toarray())


@pytest.mark.integration
def test_count_cooccurrence_raw_text_from_file(raw_source_file):
    corpus = Corpus(raw_source_file, lower=True)
    words = Vocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = Vocabulary(["is", "better", "than"])

    result = mangoes.counting.count_cooccurrence(corpus, words, context_words)

    assert np.array_equiv(result.matrix.toarray(), expected_count)


# annotated text
@pytest.mark.integration
def test_cooccurrence_word_word_from_xml_string(xml_source_string):
    corpus = Corpus(xml_source_string, reader=mangoes.corpus.XML, lower=True)
    words = Vocabulary(["beautiful", "ugly", "simple", "complex", "complicated"],
                       entity="form")
    context_words = Vocabulary(["is", "better", "than"],
                               entity="form")

    result = mangoes.counting.count_cooccurrence(corpus, words, context_words)

    assert np.array_equiv(expected_count, result.matrix.toarray())


# ## Integration with contexts
@pytest.mark.integration
def test_count_cooccurrence_window_2_2(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = DummyVocabulary(["is", "better", "than"])
    symmetric_window_2_2 = mangoes.context.Window(context_words, size=2)

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=symmetric_window_2_2)

    #                     is   better    than
    expected_count_alt = [[1, 1, 0],  # beautiful
                          [0, 1, 1],  # ugly
                          [1, 1, 0],  # simple
                          [1, 2, 1],  # complex
                          [0, 1, 1]]  # complicated

    assert np.array_equiv(expected_count_alt, result.matrix.toarray())


@pytest.mark.integration
def test_count_cooccurrence_window_1_1_dirty(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "simple", "complex"])
    context_words = DummyVocabulary(["ugly", "complex", "complicated"])
    symmetric_window_dirty = mangoes.context.Window(context_words, window_half_size=1, dirty=True)

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=symmetric_window_dirty)

    #                     ugly   complex  complicated
    expected_count_alt = [[1, 0, 0],  # beautiful
                          [0, 1, 0],  # simple
                          [0, 0, 1]]  # complex

    assert np.array_equiv(expected_count_alt, result.matrix.toarray())


@pytest.mark.integration
def test_count_cooccurrence_dynamic_window(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = DummyVocabulary(["is", "better", "than"])
    dynamic_window = mangoes.context.Window(context_words, window_half_size=3, dynamic=True)

    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words,
                                                 context=dynamic_window)

    #                     is   better    than
    expected_count_alt = [[1, 1, 1],  # beautiful
                          [1, 1, 1],  # ugly
                          [1, 1, 1],  # simple
                          [2, 2, 2],  # complex
                          [1, 1, 1]]  # complicated

    # with dynamic window, at least one count should be lower than "expected"
    assert any(result.matrix.toarray().flatten() < np.array(expected_count_alt).flatten())


# TODO(nami) fix.
@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context():
    conll_string = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	    _	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "telescope"], entity="form")
    context_words = mangoes.Vocabulary(["australian", "scientist", "star", "telescope"])

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary=context_words, collapse=True,
                                                               dependencies="stanford-dependencies", entity="form")

    #             australian scientist  star   telescope
    expected_count = [[0,         1,      0,      0],  # australian
                      [1,         0,      0,      0],  # scientist
                      [0,         1,      1,      1],  # discovers
                      [0,         0,      0,      0],  # star
                      [0,         0,      0,      0]]  # telescope

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    np.testing.assert_array_equal(expected_count, result.matrix.A)


@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_with_labels():
    conll_string = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	    _	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "telescope"], entity="form")
    context_words = mangoes.Vocabulary(["australian", "scientist", "scientist", "star", "telescope"])

    dep_based_context = mangoes.context.DependencyBasedContext(context_words, collapse=True, labels=True,
                                                               dependencies="stanford-dependencies")

    # scientist/amod  australian/amod    star/dobj   telescope/prep_with    scientist/nsubj 
    # expected_count = [
    #       [1,               0,                0,            0,              0],  # australian
    #       [0,               1,                0,            0,              0],  # scientist
    #       [0,               0,                1,            1,              1],  # discovers
    #       [0,               0,                0,            0,              0],  # star
    #       [0,               0,                0,            0,              0]  # telescope
    # ]
    # the vocabulary is built during counting so ordering might not be respected

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    assert 5 == len(result.contexts_words)
    assert 5 == result.matrix.sum()


@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_no_vocabulary():
    conll_string = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	    _	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "with", "telescope"], entity="form")

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, dependencies="stanford-dependencies")

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)
    assert 5 == len(result.contexts_words)
    assert 8 == result.matrix.sum()


@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_with_labels_no_vocabulary():
    conll_string = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	    _	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "with", "telescope"], entity="form")

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, labels=True,
                                                               dependencies="stanford-dependencies")

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    assert 8 == len(result.contexts_words)
    assert 8 == result.matrix.sum()

@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_depth_2():
    conll_string = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	8	case	_	_",
                    "6	very	very	ADV	RB	_	7	advmod	_	_",
                    "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                    "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "with", "very", "large", "telescope"], entity="form")

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, depth=2)
    
    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    assert 7 == len(result.contexts_words)
    assert 24 == result.matrix.sum()

@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_depth_2_with_labels():
    conll_string = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	8	case	_	_",
                    "6	very	very	ADV	RB	_	7	advmod	_	_",
                    "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                    "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "with", "very", "large", "telescope"], entity="form")

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, labels=True, depth=2)
    
    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    assert 24 == len(result.contexts_words)
    assert 24 == result.matrix.sum()
    assert {"scientist/amod",
            "discovers/amod+nsubj",
            "australian/amod",
            "discovers/nsubj",
            "star/nsubj+dobj",
            "telescope/nsubj+case_with",
            "australian/nsubj+amod",
            "scientist/nsubj",
            "star/dobj",
            "telescope/case_with",
            "large/case_with+amod",
            "discovers/dobj",
            "scientist/dobj+nsubj",
            "telescope/dobj+case_with",
            "large/advmod",
            "telescope/advmod+amod",
            "very/advmod",
            "telescope/amod",
            "discovers/amod+case_with",
            "large/amod",
            "very/amod+advmod",
            "discovers/case_with",
            "star/case_with+dobj",
            "scientist/case_with+nsubj"} == set(result.contexts_words)

@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_deprel():
    conll_string = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	8	case	_	_",
                    "6	very	very	ADV	RB	_	7	advmod	_	_",
                    "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                    "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star", "with", "very", "large", "telescope"], entity="form")

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, labels=True, depth=2, deprel_keep=("nmod", "nsubj", "amod"))
    
    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    assert 14 == len(result.contexts_words)
    assert 14 == result.matrix.sum()
    assert {"scientist/amod",
            "discovers/amod+nsubj",
            "australian/amod",
            "discovers/nsubj",
            "telescope/nsubj+case_with",
            "australian/nsubj+amod",
            "scientist/nsubj",
            "telescope/case_with",
            "large/case_with+amod",
            "telescope/amod",
            "discovers/amod+case_with",
            "large/amod",
            "discovers/case_with",
            "scientist/case_with+nsubj"} == set(result.contexts_words)

# TODO(nami) fix cython. 
@pytest.mark.skip(reason="Weight doesn't work yet")
@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_weight():
    conll_string = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	8	case	_	_",
                    "6	very	very	ADV	RB	_	7	advmod	_	_",
                    "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                    "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star","very", "large", "telescope"], entity="form")
    context_words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star","very", "large", "telescope"])

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary=context_words, collapse=True, depth=2,
                                                               weight=True, entity="form")

    #             australian scientist  discovers  star   very   large    telescope
    expected_count = [[0,         1,      0.5,      0,    0,    0,  0],  # australian
                      [1,         0,      1,      0.5,   0,    0,    0.5],  # scientist
                      [0.5,         1,      0,      1,   0,    0.5,    1],  # discovers
                      [0,         0.5,      1,      0,   0,    0,    0.5],  # star
                      [0,         0,      0,      0,   0,    1,    0.5],  # very
                      [0,         0,      0.5,      0,   1,    0,    1],  # large
                      [0,         0.5,      1,      0.5,   0.5,    1,    0]]  # telescope

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    np.testing.assert_array_equal(expected_count, result.matrix.A)

@pytest.mark.skip(reason="Weight doesn't work yet")
@pytest.mark.integration
def test_count_cooccurrence_dependency_based_context_weight_scheme():
    conll_string = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	8	case	_	_",
                    "6	very	very	ADV	RB	_	7	advmod	_	_",
                    "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                    "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

    corpus = mangoes.Corpus(conll_string, reader=mangoes.corpus.CONLLU)
    words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star","very", "large", "telescope"], entity="form")
    context_words = mangoes.Vocabulary(["australian", "scientist", "discovers", "star","very", "large", "telescope"])

    weight_scheme = {"nsubj": 5, "dobj": 4, "obj": 4, "amod": 3, "advmod":2}

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary=context_words, collapse=True, depth=2,
                                                               weight=True, weight_scheme=weight_scheme, entity="form")

    #             australian scientist  discovers  star   very   large    telescope
    expected_count = [[0,         3,      5,      0,    0,    0,  0],  # australian
                      [3,         0,      5,      5,   0,    0,    5],  # scientist
                      [5,         5,      0,      4,   0,    3,    1],  # discovers
                      [0,         5,      4,      0,   0,    0,    4],  # star
                      [0,         0,      0,      0,   0,    2,    3],  # very
                      [0,         0,      3,      0,   2,    0,    3],  # large
                      [0,         5,      1,      4,   3,    3,    0]]  # telescope

    result = mangoes.counting.count_cooccurrence(corpus, words, context=dep_based_context)

    np.testing.assert_array_equal(expected_count, result.matrix.A)
    
# ## Integration with subsampling
@pytest.mark.integration
def test_count_cooccurrence_subsampling(dummy_raw_corpus):
    words = DummyVocabulary(["beautiful", "ugly", "simple", "complex", "complicated"])
    context_words = DummyVocabulary(["is", "better", "than"])
    context = DummyContext(context_words)

    # subsampling threshold == 1 -> no subsampling
    result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words, context, subsampling=1)
    assert np.array_equiv(expected_count, result.matrix.toarray())

    # subsampling threshold == 10**-5, frequencies > 0,03333 -> p ~ 1 for each word
    with unittest.mock.patch('random.random', return_value=0.5):
        result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words, context, subsampling=True, nb_workers=1)
        np.testing.assert_array_equal(np.zeros(shape=(5, 3)), result.matrix.toarray())

    # subsampling threshold == 0.125 -> p = 0.2 for 'is', 'better' and 'than'
    with unittest.mock.patch('random.random') as mock_random:
        mock_random.side_effect = [0.1, 0.3] * 20#9
        result = mangoes.counting.count_cooccurrence(dummy_raw_corpus, words, context, subsampling=0.125, nb_workers=1)

        # Beautiful is      better than ugly
        #           0.1     0.3     0.1
        # Explicit  is      better than implicit
        #           0.3     0.1     0.3
        # Simple    is      better than complex
        #           0.1     0.3     0.1
        # Complex   is      better than complicated
        #           0.3     0.1     0.3

        #                     is   better    than
        expected_count_alt = [[0, 0, 0],  # beautiful
                              [0, 0, 0],  # ugly
                              [0, 0, 0],  # simple
                              [1, 0, 0],  # complex
                              [0, 0, 1]]  # complicated

        assert np.array_equiv(expected_count_alt, result.matrix.toarray())


# ## Integration with vocabulary of tokens, filtered from corpus
@pytest.mark.integration
@pytest.mark.timeout(5)
def test_cooccurrence_filtered_token(brown_source_string):
    corpus = mangoes.Corpus(brown_source_string, reader=mangoes.corpus.BROWN, lower=True)

    words = mangoes.Vocabulary([('beautiful', 'JJ'),
                                ('ugly', 'JJ'),
                                ('simple', 'NN'),
                                ('complex', 'JJ'),
                                ('complex', 'NN'),
                                ('complicate', 'VBN')], entity=('lemma', 'POS'))

    context_words = corpus.create_vocabulary(filters=[mangoes.corpus.remove_least_frequent(3)],
                                             attributes=('lemma', 'POS'))
    context_words = mangoes.Vocabulary(sorted(context_words.words, key=lambda x: x.lemma),
                                       entity=('lemma', 'POS'))
    # sort to be sure the words would be in the expected order
    # TODO : a sort method should be added to Vocabulary class

    result = mangoes.counting.count_cooccurrence(corpus, words, context=context_words, nb_workers=3)

    #                     is   better    than
    expected_count_alt = [[1, 0, 0],  # beautiful/JJ
                          [0, 0, 1],  # ugly/JJ
                          [1, 0, 0],  # simple/NN
                          [0, 0, 1],  # complex/JJ
                          [1, 0, 0],  # complex/NN
                          [0, 0, 1]]  # complicated/VBN

    assert np.array_equiv(expected_count_alt, result.matrix.toarray())


# ## Merge CooccurrenceCounts
@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_simple(matrix_type):
    words = mangoes.Vocabulary(['a', 'b', 'c'])
    contexts = mangoes.Vocabulary(['x', 'y', 'z'])
    counts = matrix_type(np.array(range(9)).reshape((3,3)))

    cc1 = mangoes.CountBasedRepresentation(words, contexts, counts)
    cc2 = mangoes.CountBasedRepresentation(words, contexts, counts)

    cc = mangoes.counting.merge(cc1, cc2)

    # Merge :
    #     x  y  z                  x  y  z                     x   y   z
    # a : 0  1  2              a : 0  1  2                a :  0   2   4
    # b : 3  4  5      and     b : 3  4  5        gives   b :  6   8  10
    # c : 6  7  8              c : 6  7  8                c : 12  14  16

    assert ['a', 'b', 'c'] == cc.words.words
    assert ['x', 'y', 'z'] == cc.contexts_words.words
    np.testing.assert_array_equal([[0,2,4], [6,8,10], [12,14,16]], cc.matrix.as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_different_contexts(matrix_type):
    words = mangoes.Vocabulary(['a', 'b', 'c'])

    contexts1 = mangoes.Vocabulary(['x', 'y', 'z'])
    counts1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    cc1 = mangoes.CountBasedRepresentation(words, contexts1, counts1)

    contexts2 = mangoes.Vocabulary(['x', 't'])
    counts2 = matrix_type(np.array(range(6)).reshape((3, 2)))
    cc2 = mangoes.CountBasedRepresentation(words, contexts2, counts2)

    # Merge :
    #     x  y  z                  x  t                       x  y  z  t
    # a : 0  1  2              a : 0  1                  a :  0  1  2  1
    # b : 3  4  5      and     b : 2  3         gives    b :  5  4  5  3
    # c : 6  7  8              c : 4  5                  c : 10  7  8  5

    cc = mangoes.counting.merge(cc1, cc2)
    assert ['a', 'b', 'c'] == cc.words.words
    assert ['x', 'y', 'z', 't'] == cc.contexts_words.words
    np.testing.assert_array_equal([[0,1,2,1], [5,4,5,3], [10,7,8,5]], cc.matrix.as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_different_target_words(matrix_type):
    contexts = mangoes.Vocabulary(['x', 'y', 'z'])

    words1 = mangoes.Vocabulary(['a', 'b', 'c'])
    counts1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    cc1 = mangoes.CountBasedRepresentation(words1, contexts, counts1)

    words2 = mangoes.Vocabulary(['a', 'd'])
    counts2 = matrix_type(np.array(range(6)).reshape((2,3)))
    cc2 = mangoes.CountBasedRepresentation(words2, contexts, counts2)


    # Merge :
    #     x  y  z                  x  y  z                      x  y  z
    # a : 0  1  2              a : 0  1  2                  a : 0  2  4
    # b : 3  4  5      and     d : 3  4  5        gives     b : 3  4  5
    # c : 6  7  8                                           c : 6  7  8
    #                                                       d : 3  4  5

    cc = mangoes.counting.merge(cc1, cc2)

    assert ['a', 'b', 'c', 'd'] == cc.words.words
    assert ['x', 'y', 'z'] == cc.contexts_words.words
    np.testing.assert_array_equal([[0,2,4], [3,4,5], [6,7,8], [3,4,5]], cc.matrix.as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_different_words_and_different_contexts(matrix_type):
    words1 = mangoes.Vocabulary(['a', 'b', 'c'])
    contexts1 = mangoes.Vocabulary(['x', 'y', 'z'])
    counts1 = matrix_type(np.array(range(9)).reshape((3, 3)))
    cc1 = mangoes.CountBasedRepresentation(words1, contexts1, counts1)

    words2 = mangoes.Vocabulary(['a', 'd'])
    contexts2 = mangoes.Vocabulary(['x', 't'])
    counts2 = matrix_type(np.array(range(4)).reshape((2,2)))
    cc2 = mangoes.CountBasedRepresentation(words2, contexts2, counts2)

    # Merge :
    #     x  y  z                  x  t                       x  y  z  t
    # a : 0  1  2              a : 0  1                  a :  0  1  2  1
    # b : 3  4  5      and     d : 2  3        gives     b :  3  4  5  0
    # c : 6  7  8                                        c :  6  7  8  0
    #                                                    d :  2  0  0  3

    cc = mangoes.counting.merge(cc1, cc2)

    assert ['a', 'b', 'c', 'd'] == cc.words.words
    assert ['x', 'y', 'z', 't'] == cc.contexts_words.words
    np.testing.assert_array_equal([[0,1,2,1], [3,4,5,0], [6,7,8,0], [2,0,0,3]],
                                  cc.matrix.as_dense())


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_with_keys(matrix_type):

    vocabulary1 = mangoes.Vocabulary(['A', 'B', 'C', 'D'], language='l1')
    vocabulary2 = mangoes.Vocabulary(['A', 'B', 'E', 'F'], language='l2')

    context1 = mangoes.Vocabulary(['1', '2', '3'])
    context2 = mangoes.Vocabulary(['1', '2', '3'])

    matrix1 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    matrix2 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]])

    count1 = mangoes.CountBasedRepresentation(vocabulary1, context1, matrix1)
    count2 = mangoes.CountBasedRepresentation(vocabulary2, context2, matrix2)

    mrg_count = mangoes.counting.merge(count1, count2, word_keys=True)

    expected_matrix = [[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1]]

    assert np.array_equiv(expected_matrix, mrg_count.matrix.as_dense())
    assert mrg_count.contexts_words.words == context1.words
    assert 8 == len(mrg_count.words)

    assert ['l1_A', 'l1_B', 'l1_C', 'l1_D', 'l2_A', 'l2_B', 'l2_E', 'l2_F'] == mrg_count.words.words


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_3_counts(matrix_type):
    vocabulary1 = mangoes.Vocabulary(['A', 'B', 'C', 'D'], language='l1')
    vocabulary2 = mangoes.Vocabulary(['A', 'B', 'E', 'F'], language='l2')
    vocabulary3 = mangoes.Vocabulary(['A', 'C', 'E', 'G'], language='l3')

    context = mangoes.Vocabulary(['1', '2', '3'])

    matrix1 = scipy.sparse.csr_matrix([[1, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]])

    matrix2 = scipy.sparse.csr_matrix([[1, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 0],
                                       [0, 0, 1]])

    matrix3 = scipy.sparse.csr_matrix([[1, 0, 0],
                                       [0, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]])

    count1 = mangoes.CountBasedRepresentation(vocabulary1, context, matrix1)
    count2 = mangoes.CountBasedRepresentation(vocabulary2, context, matrix2)
    count3 = mangoes.CountBasedRepresentation(vocabulary3, context, matrix3)

    mrg_count = mangoes.counting.merge(count1, count2, count3, word_keys=True)

    expected_matrix = scipy.sparse.csr_matrix([[1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [1, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 0],
                                               [0, 0, 1],
                                               [1, 0, 0],
                                               [0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]
                                               ])

    assert np.array_equiv(expected_matrix.A, mrg_count.matrix.A)
    assert mrg_count.contexts_words.words == context.words
    assert 12 == len(mrg_count.words)


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_tokens_vocabularies_with_keys(matrix_type):
    Token = collections.namedtuple('Token', 'form POS lemma')
    vocabulary1 = mangoes.Vocabulary([Token('A', 'X', 'a'),
                                      Token('B', 'Y', 'b'),
                                      Token('C', 'X', 'c'),
                                      Token('D', 'Y', 'd')], language='l1')
    vocabulary2 = mangoes.Vocabulary([Token('A', 'X', 'a'),
                                      Token('B', 'X', 'b'),
                                      Token('E', 'Z', 'e'),
                                      Token('F', 'X', 'f')], language='l2')

    context1 = mangoes.Vocabulary(['1', '2', '3'])
    context2 = mangoes.Vocabulary(['1', '2', '3'])

    matrix1 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    matrix2 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]])

    count1 = mangoes.CountBasedRepresentation(vocabulary1, context1, matrix1)
    count2 = mangoes.CountBasedRepresentation(vocabulary2, context2, matrix2)

    TokenWithLang = collections.namedtuple('TokenWithLang', 'form POS lemma lang')
    mrg_count = mangoes.counting.merge(count1, count2, word_keys=True, concat=lambda k, w: TokenWithLang(*w, k))

    expected_matrix = [[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1]]

    assert np.array_equiv(expected_matrix, mrg_count.matrix.as_dense())
    assert mrg_count.contexts_words.words == context1.words
    assert 8 == len(mrg_count.words)
    assert ('A', 'X', 'a', 'l1') in mrg_count.words.words


@pytest.mark.parametrize("matrix_type",
                         [mangoes.utils.arrays.NumpyMatrix, mangoes.utils.arrays.csrSparseMatrix],
                         ids=["dense", "sparse"])
def test_merge_with_keys_with_bigrams(matrix_type):

    vocabulary1 = mangoes.Vocabulary(['A B', 'C', 'D'], language='l1')
    vocabulary2 = mangoes.Vocabulary(['A B', 'E', 'F'], language='l2')

    context1 = mangoes.Vocabulary(['1', '2', '3'])
    context2 = mangoes.Vocabulary(['1', '2', '3'])

    matrix1 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]])

    matrix2 = matrix_type([[1, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]])

    count1 = mangoes.CountBasedRepresentation(vocabulary1, context1, matrix1)
    count2 = mangoes.CountBasedRepresentation(vocabulary2, context2, matrix2)

    def concat(k, w):
        if isinstance(w, mangoes.vocabulary.Bigram):
            return mangoes.vocabulary.Bigram(concat(k, w[0]), concat(k, w[1]))
        return k + '_' + w
    mrg_count = mangoes.counting.merge(count1, count2, word_keys=True, concat=concat)

    expected_matrix = [[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1]]

    assert np.array_equiv(expected_matrix, mrg_count.matrix.as_dense())
    assert mrg_count.contexts_words.words == context1.words
    assert 6 == len(mrg_count.words)

    assert [('l1_A', 'l1_B'), 'l1_C', 'l1_D', ('l2_A',  'l2_B'), 'l2_E', 'l2_F'] == mrg_count.words.words


########################
# Exceptions

def test_exception_no_vocabulary(save_temp_dir):
    with pytest.raises(mangoes.utils.exceptions.RequiredValue):
        corpus = mangoes.Corpus(save_temp_dir, lazy=True)
        mangoes.counting.count_cooccurrence(corpus, words=None)
