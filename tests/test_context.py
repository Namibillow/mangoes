# -*- coding: utf-8 -*-

import pytest
import unittest.mock
import logging

import mangoes.context

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)

# ###########################################################################################
# ### TESTS
#             0       1        2       3       4       5       6       7       8
sentence = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
vocabulary = ['quick', 'brown', 'fox', 'lazy', 'dog']


@pytest.mark.unittest
def test_sentence_no_oov():
    expected = [
        ['quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['The', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['The', 'quick', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['The', 'quick', 'brown', 'jumps', 'over', 'the', 'lazy', 'dog'],
        ['The', 'quick', 'brown', 'fox', 'over', 'the', 'lazy', 'dog'],
        ['The', 'quick', 'brown', 'fox', 'jumps', 'the', 'lazy', 'dog'],
        ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog'],
        ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'dog'],
        ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy']
    ]
    assert expected == mangoes.context.Sentence()(sentence)


@pytest.mark.unittest
def test_sentence_with_oov():
    expected = [
        ['quick', 'brown', 'fox', 'lazy', 'dog'],
        ['brown', 'fox', 'lazy', 'dog'],
        ['quick', 'fox', 'lazy', 'dog'],
        ['quick', 'brown', 'lazy', 'dog'],
        ['quick', 'brown', 'fox', 'lazy', 'dog'],
        ['quick', 'brown', 'fox', 'lazy', 'dog'],
        ['quick', 'brown', 'fox', 'lazy', 'dog'],
        ['quick', 'brown', 'fox', 'dog'],
        ['quick', 'brown', 'fox', 'lazy']
    ]
    assert expected == mangoes.context.Sentence(vocabulary)(sentence)


@pytest.mark.unittest
def test_symmetric_window_no_oov():
    expected = [['quick'],
                ['The', 'brown'],
                ['quick', 'fox'],
                ['brown', 'jumps'],
                ['fox', 'over'],
                ['jumps', 'the'],
                ['over', 'lazy'],
                ['the', 'dog'],
                ['lazy']]
    assert expected == mangoes.context.Window()(sentence)


@pytest.mark.unittest
def test_symmetric_window_with_oov():
    expected = [['quick'],
                ['brown'],
                ['quick', 'fox'],
                ['brown'],
                ['fox'],
                [],
                ['lazy'],
                ['dog'],
                ['lazy'], ]
    assert expected == mangoes.context.Window(mangoes.Vocabulary(vocabulary))(sentence)


@pytest.mark.unittest
def test_symmetric_window_size_2():
    expected = [['quick', 'brown'],
                ['brown', 'fox'],
                ['quick', 'fox'],
                ['quick', 'brown'],
                ['brown', 'fox'],
                ['fox', 'lazy'],
                ['lazy', 'dog'],
                ['dog'],
                ['lazy']]
    assert expected == mangoes.context.Window(vocabulary, size=2)(sentence)


@pytest.mark.unittest
def test_symmetric_window_dirty():
    expected = [['quick'],
                ['brown'],
                ['quick', 'fox'],
                ['brown', 'lazy'],
                ['fox', 'lazy'],
                ['fox', 'lazy'],
                ['fox', 'lazy'],
                ['fox', 'dog'],
                ['lazy']]
    assert expected == mangoes.context.Window(vocabulary, dirty=True)(sentence)


@pytest.mark.unittest
def test_asymmetric_window():
    expected = [['quick', 'brown'],
                ['brown', 'fox'],
                ['quick', 'fox'],
                ['brown'],
                ['fox'],
                ['lazy'],
                ['lazy', 'dog'],
                ['dog'],
                ['lazy']]
    assert expected == mangoes.context.Window(vocabulary, size=(1, 2))(sentence)


@pytest.mark.unittest
def test_asymmetric_window_dirty():
    expected = [['quick', 'brown'],
                ['brown', 'fox'],
                ['quick', 'fox', 'lazy'],
                ['brown', 'lazy', 'dog'],
                ['fox', 'lazy', 'dog'],
                ['fox', 'lazy', 'dog'],
                ['fox', 'lazy', 'dog'],
                ['fox', 'dog'],
                ['lazy']]
    result = mangoes.context.Window(vocabulary, size=(1, 2), dirty=True)(sentence)
    assert expected == result


@pytest.mark.unittest
def test_symmetric_window_2grams():
    expected = [[('quick', 'brown')],
                [('brown', 'fox')],
                [],
                [('quick', 'brown')],
                [('brown', 'fox')],
                [],
                [('lazy', 'dog')],
                [],
                []]
    assert expected == mangoes.context.Window(vocabulary, size=2, n_grams=2)(sentence)


@pytest.mark.unittest
def test_symmetric_window_dirty_2grams():
    expected = [[('quick', 'brown'), ('brown', 'fox')],
                [('brown', 'fox'), ('lazy', 'dog')],
                [('lazy', 'dog')],
                [('quick', 'brown'), ('lazy', 'dog')],
                [('quick', 'brown'), ('brown', 'fox'), ('lazy', 'dog')],
                [('quick', 'brown'), ('brown', 'fox'), ('lazy', 'dog')],
                [('quick', 'brown'), ('brown', 'fox'), ('lazy', 'dog')],
                [('quick', 'brown'), ('brown', 'fox')],
                [('quick', 'brown'), ('brown', 'fox')]]
    assert expected == mangoes.context.Window(vocabulary, dirty=True, size=2, n_grams=2)(sentence)


@pytest.mark.unittest
def test_symmetric_window_3grams():
    expected = [[('quick', 'brown', 'fox')],
                [],
                [],
                [],
                [('quick','brown', 'fox')],
                [],
                [],
                [],
                []]
    assert expected == mangoes.context.Window(vocabulary, size=3, n_grams=3)(sentence)


@pytest.mark.unittest
def test_symmetric_window_distance():
    expected = [[('quick',1), ('brown',2)],
                [('brown',1), ('fox',2)],
                [('quick',-1), ('fox',1)],
                [('quick',-2), ('brown',-1)],
                [('brown',-2), ('fox',-1)],
                [('fox',-2), ('lazy',2)],
                [('lazy',1), ('dog',2)],
                [('dog',1)],
                [('lazy',-1)]]
    assert expected == mangoes.context.Window(vocabulary, size=2, distance=True)(sentence)

    result = mangoes.context.Window(size=(2,0), distance=True)(sentence)
    assert [[], [('The', -1)], [('The', -2), ('quick', -1)]] == result[:3]


@pytest.mark.unittest
def test_symmetric_window_2grams_distance():
    expected = [[(('quick', 'brown'),1), (('brown', 'fox'),2)],
                [(('brown', 'fox'),1)],
                [],
                [(('quick', 'brown'),-1)],
                [(('quick', 'brown'),-2), (('brown', 'fox'),-1)],
                [(('brown', 'fox'),-2), (('lazy', 'dog'),2)],
                [(('lazy', 'dog'),1)],
                [],
                []]
    assert expected == mangoes.context.Window(vocabulary, size=3, n_grams=2, distance=True)(sentence)


@pytest.mark.unittest
def test_dynamic_window():
    with unittest.mock.patch('random.randint') as mock_random:
        mock_random.side_effect = [3, 1, 3, 1, 2, 3, 3, 2, 2]

        expected = [['quick', 'brown', 'fox'],
                    ['brown'],
                    ['quick', 'fox'],
                    ['brown'],
                    ['brown', 'fox'],
                    ['brown', 'fox', 'lazy', 'dog'],
                    ['fox', 'lazy', 'dog'],
                    ['dog'],
                    ['lazy']]

        result = mangoes.context.Window(vocabulary, size=3, dynamic=True)(sentence)

        assert expected == result


@pytest.mark.unittest
def test_dynamic_dirty_window():
    with unittest.mock.patch('random.randint') as mock_random:
        mock_random.side_effect = [3, 1, 3, 1, 2, 3, 3, 2, 2]

        expected = [['quick', 'brown', 'fox'],
                    ['brown'],
                    ['quick', 'fox', 'lazy', 'dog'],
                    ['brown', 'lazy'],
                    ['brown', 'fox', 'lazy', 'dog'],
                    ['quick', 'brown', 'fox', 'lazy', 'dog'],
                    ['quick', 'brown', 'fox', 'lazy', 'dog'],
                    ['brown', 'fox', 'dog'],
                    ['fox', 'lazy']]

        result = mangoes.context.Window(vocabulary, size=3, dirty=True, dynamic=True)(sentence)
        assert expected == result


@pytest.mark.unittest
def test_dynamic_asymetric_window():
    with unittest.mock.patch('random.randint') as mock_random:
        mock_random.side_effect = [1, 1, 2, 2, 1, 3, 2, 1, 1, 2, 2, 3, 1, 1, 2, 2, 1, 3]

        expected = [['quick'],  # 1-1 :              ^  *The*   quick
                    ['brown', 'fox'],  # 2-2 :         ^ (The) *quick* brown fox
                    ['quick', 'fox'],  # 1-3 :           quick *brown* fox (jumps) (over)
                    ['quick', 'brown'],  # 2-1 :     quick brown *fox*   (jumps)
                    ['fox'],  # 1-2 :             fox *jumps* (over) (the)
                    ['fox', 'lazy', 'dog'],  # 2-3 :     fox (jumps) *over*  (the) lazy dog
                    ['lazy'],  # 1-1 :          (over) *the*   lazy
                    ['dog'],  # 2-2 :    (over) (the) *lazy*  dog $
                    ['lazy']]  # 1-3 :            lazy *dog*   $

        result = mangoes.context.Window(vocabulary, size=(2, 3), dynamic=True)(sentence)

        assert expected == result


conllu_string_ud = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                    "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                    "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                    "4	star	star	NOUN	NN	_	3	dobj	_	_",
                    "5	with	with	ADP	IN	_	6	case	_	_",
                    "6	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]
# java edu.stanford.nlp.pipeline.StanfordCoreNLP -props
# -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse -outputFormat conllu


conllu_string_sd = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	_	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]
# java edu.stanford.nlp.pipeline.StanfordCoreNLP -props -parse.originalDependencies
# -annotators tokeplit,pos,parse -outputFormat conllu


xml_string = """<root><document><sentences>
                      <sentence id="1">
                        <tokens>
                          <token id="1"><word>australian</word><lemma>australian</lemma><POS>JJ</POS></token>
                          <token id="2"><word>scientist</word><lemma>scientist</lemma><POS>NN</POS></token>
                          <token id="3"><word>discovers</word><lemma>discover</lemma><POS>VBZ</POS></token>
                          <token id="4"><word>star</word><lemma>star</lemma><POS>NN</POS></token>
                          <token id="5"><word>with</word><lemma>with</lemma><POS>IN</POS></token>
                          <token id="6"><word>telescope</word><lemma>telescope</lemma><POS>NN</POS></token>
                        </tokens>
                        <dependencies type="basic-dependencies">
                          <dep type="root">
                            <governor idx="0">ROOT</governor><dependent idx="3">discovers</dependent>
                          </dep>
                          <dep type="amod">
                            <governor idx="2">scientist</governor><dependent idx="1">australian</dependent>
                          </dep>
                          <dep type="nsubj">
                            <governor idx="3">discovers</governor><dependent idx="2">scientist</dependent>
                          </dep>
                          <dep type="dobj">
                            <governor idx="3">discovers</governor><dependent idx="4">star</dependent>
                          </dep>
                          <dep type="case">
                            <governor idx="6">telescope</governor><dependent idx="5">with</dependent>
                          </dep>
                          <dep type="nmod">
                            <governor idx="3">discovers</governor><dependent idx="6">telescope</dependent>
                          </dep>
                        </dependencies>
                      </sentence></sentences></document></root>"""


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud(source, reader):
    dep_based_context = mangoes.context.DependencyBasedContext(entity="form")

    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    expected = [{"scientist"},  # australian
                {"australian", "discovers"},  # scientist
                {"scientist", "star", "telescope"},  # discovers
                {"discovers"},  # star
                {"telescope"},  # with
                {"with", "discovers"}]

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_filter_vocabulary(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    vocabulary = ["scientist", "telescope"]

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary, entity="form")

    expected = [{"scientist"},  # australian
                set(),  # scientist
                {"scientist", "telescope"},  # discovers
                set(),  # star
                {"telescope"},  # with
                set()]  # telescope

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, entity="form")
    expected = [{"scientist"},  # australian
                {"australian", "discovers"},  # scientist
                {"scientist", "star", "telescope"},  # discovers
                {"discovers"},  # star
                set(),  # with
                {"discovers"}, ]  # telescope

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_labels(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(labels=True)

    expected = [{"scientist/amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"scientist/nsubj", "star/dobj", "telescope/nmod"},  # discovers
                {"discovers/dobj-"},  # star
                {"telescope/case-"},  # with
                {"with/case", "discovers/nmod-"}]  # telescope

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],

                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_labels_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, labels=True)

    expected = [{"scientist/amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"scientist/nsubj", "star/dobj", "telescope/case_with"},  # discovers
                {"discovers/dobj-"},  # star
                set(),  # with
                {"discovers/case_with-"}]  # telescope

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_labels_collapse_vocabulary(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()
    # vocabulary = ["scientist/amod-", "telescope/case_with"]
    vocabulary = ["scientist", "telescope"]

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary, collapse=True, labels=True)

    expected = [{"scientist/amod-"},  # australian
                set(),  # scientist
                {"scientist/nsubj", "telescope/case_with"},  # discovers
                set(),  # star
                set(),  # with
                set()]  # telescope

    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator), ])
def test_dependency_based_context_sd_labels_collapse(source, reader):
    dep_based_context = mangoes.context.DependencyBasedContext(dependencies="stanford-dependencies",
                                                               collapse=True, labels=True)
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    expected = [{"scientist/amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"scientist/nsubj", "star/dobj", "telescope/prep_with"},  # discovers
                {"discovers/dobj-"},  # star
                set(),  # with
                {"discovers/prep_with-"}]  # telescope


    assert expected == dep_based_context(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator),])
def test_parse_sentence_ud(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    parser = mangoes.context.DependencyBasedContext.ud_sentence_parser
    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod')},
            set(),
            set(),
            {(4, 'case')}] == parser(sentence)

    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'case_with')},
            set(),
            set(),
            set()] == parser(sentence, collapse=True)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_parse_sentence_sd(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    parser = mangoes.context.DependencyBasedContext.stanford_dependencies_sentence_parser
    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (4, 'prep')},
            set(),
            {(5, 'pobj')},
            set()] == parser(sentence)

    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'prep_with')},
            set(),
            set(),
            set()] == parser(sentence, collapse=True)

def test_add_grand_children():
    sentence = [set(),
                {(0, 'amod')},
                {(1, 'nsubj'), (3, 'dobj'), (4, 'prep')},
                set(),
                {(5, 'pobj')},
                set()]

    expected = [set(),
                {(0, 'amod')},
                {(1, 'nsubj'), (0, 'nsubj_amod'), (3, 'dobj'), (4, 'prep'), (5, 'prep_pobj')},
                set(),
                {(5, 'pobj')},
                set()]

    assert expected == mangoes.context.DependencyBasedContext.add_children(sentence)


@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_depth_2(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form", depth=2)

    expected = [{"scientist", "discovers"},  # australian
                {"australian", "discovers"},  # scientist
                {"australian", "scientist", "star", "with", "telescope"},  # discovers
                {"discovers"},  # star
                {"discovers", "telescope"},  # with
                {"with", "discovers"}]

    assert expected == dep_based_context(sentence)

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form", labels=True, depth=2)

    expected = [{"scientist/amod-", "discovers/nsubj_amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"australian/nsubj_amod", "scientist/nsubj", "star/dobj", "with/nmod_case","telescope/nmod"},  # discovers
                {"discovers/dobj-"},  # star
                {"discovers/nmod_case-", "telescope/case-"},  # with
                {"with/case", "discovers/nmod-"}]  # telescope

    assert expected == dep_based_context(sentence)

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form", collapse=True, depth=2)

    expected = [{"scientist", "discovers"},  # australian
                {"australian", "discovers"},  # scientist
                {"australian", "scientist", "star", "telescope"},  # discovers
                {"discovers"},  # star
                set(),  # with
                {"discovers"}]

    assert expected == dep_based_context(sentence)

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form", labels=True, collapse=True, depth=2)

    expected = [{"scientist/amod-", "discovers/nsubj_amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"australian/nsubj_amod", "scientist/nsubj", "star/dobj", "telescope/case_with"},  # discovers
                {"discovers/dobj-"},  # star
                set(),  # with
                {"discovers/case_with-"}]  # telescope

    assert expected == dep_based_context(sentence)

def test_dependency_based_context_ud_depth_3():
    conllu_string_ud = ["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                        "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                        "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                        "4	star	star	NOUN	NN	_	3	dobj	_	_",
                        "5	with	with	ADP	IN	_	8	case	_	_",
                        "6	very	very	ADV	RB	_	7	advmod	_	_",
                        "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                        "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]
    corpus = mangoes.Corpus(conllu_string_ud, reader=mangoes.corpus.CONLLU)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form", labels=True, collapse=True, depth=3)

    expected = [{"scientist/amod-", "discovers/nsubj_amod-"},  # australian
                {"australian/amod", "discovers/nsubj-"},  # scientist
                {"australian/nsubj_amod", "scientist/nsubj", "star/dobj", "telescope/case_with",
                 "large/case_with_amod", "very/case_with_amod_advmod"},  # discovers
                {"discovers/dobj-"},  # star
                set(),  # with
                {"large/advmod-", "telescope/amod_advmod-", "discovers/case_with_amod_advmod-"},  # very
                {"very/advmod", "telescope/amod-", "discovers/case_with_amod-"},  # large
                {"discovers/case_with-", "large/amod", "very/amod_advmod"}]  # telescope

    assert expected == dep_based_context(sentence)



