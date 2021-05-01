# -*- coding: utf-8 -*-

import pytest
import unittest.mock
import logging
import mangoes
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
# java edu.stanford.nlp.pipeline.StanfordCoreNLP 
# -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse -outputFormat conllu


conllu_string_sd = ["1	australian	_	_	JJ	_	2	amod	_	_",
                    "2	scientist	_	_	NN	_	3	nsubj	_	_",
                    "3	discovers	_	_	VBZ	_	0	root	_	_",
                    "4	star	_	_	NN	_	3	dobj	_	_",
                    "5	with	    _	_	IN	_	3	prep	_	_",
                    "6	telescope	_	_	NN	_	5	pobj	_	_"]
# java edu.stanford.nlp.pipeline.StanfordCoreNLP -parse.originalDependencies
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


conllu_strign_sd_2 =   ["1	australian	australian	_	JJ	_	2	amod	_	_",
                        "2	scientist	scientist	_	NN	_	3	nsubj	_	_",
                        "3	discovers	discover	_	VBZ	_	0	root	_	_",
                        "4	star	star	_	NN	_	3	dobj	_	_",
                        "5	with	with	_	IN	_	3	prep	_	_",
                        "6	very	very	_	RB	_	7	advmod	_	_",
                        "7	large	large	_	JJ	_	8	amod	_	_",
                        "8	telescope	telescope	_	NN	_	5	pobj	_	_"]

conllu_string_ud_2 = ["  1	australian	australian	ADJ	JJ	_	2	amod	_	_",
                        "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
                        "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
                        "4	star	star	NOUN	NN	_	3	dobj	_	_",
                        "5	with	with	ADP	IN	_	8	case	_	_",
                        "6	very	very	ADV	RB	_	7	advmod	_	_",
                        "7	large	large	ADJ	JJ	Degree=Pos	8	amod	_	_",
                        "8	telescope	telescope	NOUN	NN	_	3	nmod	_	_"]

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(entity="form")
    expected = [{"scientist/1"},  # australian
                {"australian/1", "discovers/1"},  # scientist
                {"scientist/1", "star/1", "telescope/1"},  # discovers
                {"discovers/1"},  # star
                {"telescope/1"},  # with
                {"with/1", "discovers/1"}]

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_dependency_based_context_sd(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(dependencies="stanford-dependencies")
    expected = [{"scientist/1"},  # australian
                {"australian/1", "discovers/1"},  # scientist
                {"scientist/1", "star/1", "with/1"},  # discovers
                {"discovers/1"},  # star
                {"telescope/1", "discovers/1"},  # with
                {"with/1"}]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_filter_vocabulary(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    vocabulary = ["scientist", "telescope"]

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary, entity="form")
    expected = [{"scientist/1"},  # australian
                set(),  # scientist
                {"scientist/1", "telescope/1"},  # discovers
                set(),  # star
                {"telescope/1"},  # with
                set()]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_dependency_based_context_sd_filter_vocabulary(source, reader):                    
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    vocabulary = ["scientist", "telescope"]

    dep_based_context = mangoes.context.DependencyBasedContext(vocabulary, dependencies="stanford-dependencies")
    expected = [{"scientist/1"},  # australian
                set(),  # scientist
                {"scientist/1"},  # discovers
                set(),  # star
                {"telescope/1"},  # with
                set()]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, entity="form")
    expected = [{"scientist/1"},  # australian
                {"australian/1", "discovers/1"},  # scientist
                {"scientist/1", "star/1", "telescope/1"},  # discovers
                {"discovers/1"},  # star
                set(),  # with
                {"discovers/1"}, ]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_dependency_based_context_sd_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, dependencies="stanford-dependencies")
    expected = [{"scientist/1"},  # australian
                {"australian/1", "discovers/1"},  # scientist
                {"scientist/1", "star/1", "telescope/1"},  # discovers
                {"discovers/1"},  # star
                set(),  # with
                {"discovers/1"}]  # telescope

    assert expected == dep_based_context(sentence) 

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_directed(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(directed=True, entity="form")
    expected = [set(),
                {"australian/1"},
                {"scientist/1", "star/1", "telescope/1"},
                set(),
                set(),
                {"with/1"},]
                
    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud_2, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_dependency_based_context_ud_directed_depth_2(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(directed=True, depth=2, entity="form")
    expected = [set(),
                {"australian/1"},
                {"australian/1","scientist/1", "star/1", "telescope/1", "large/1", "with/1"},
                set(),
                set(),
                set(),
                {"very/1"},
                {"with/1", "large/1", "very/1"},]
                
    assert expected == dep_based_context(sentence) 

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_directed_labels(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(directed=True, labels=True, entity="form")
    expected = [set(),
                {"australian/amod/1"},
                {"scientist/nsubj/1", "star/dobj/1", "telescope/nmod/1"},
                set(),
                set(),
                {"with/case/1"},]
                
    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_labels(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(labels=True)

    expected = [{"scientist/amod/1"},  # australian
                {"australian/amod/1", "discovers/nsubj/1"},  # scientist
                {"scientist/nsubj/1", "star/dobj/1", "telescope/nmod/1"},  # discovers
                {"discovers/dobj/1"},  # star
                {"telescope/case/1"},  # with
                {"with/case/1", "discovers/nmod/1"}]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud, mangoes.utils.reader.ConllUSentenceGenerator),
                          (xml_string, mangoes.utils.reader.XmlSentenceGenerator), ])
def test_dependency_based_context_ud_labels_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(collapse=True, labels=True)

    expected = [{"scientist/amod/1"},  # australian
                {"australian/amod/1", "discovers/nsubj/1"},  # scientist
                {"scientist/nsubj/1", "star/dobj/1", "telescope/case_with/1"},  # discovers
                {"discovers/dobj/1"},  # star
                set(),  # with
                {"discovers/case_with/1"}]  # telescope

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

    expected = [{"scientist/amod/1"},  # australian
                set(),  # scientist
                {"scientist/nsubj/1", "telescope/case_with/1"},  # discovers
                set(),  # star
                set(),  # with
                set()]  # telescope

    assert expected == dep_based_context(sentence)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_sd, mangoes.utils.reader.ConllUSentenceGenerator), ])
def test_dependency_based_context_sd_labels_collapse(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context = mangoes.context.DependencyBasedContext(dependencies="stanford-dependencies",
                                                               collapse=True, labels=True)

    expected = [{"scientist/amod/1"},  # australian
                {"australian/amod/1", "discovers/nsubj/1"},  # scientist
                {"scientist/nsubj/1", "star/dobj/1", "telescope/prep_with/1"},  # discovers
                {"discovers/dobj/1"},  # star
                set(),  # with
                {"discovers/prep_with/1"}]  # telescope


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
    sentence_sd = [set(),
                {(0, 'amod')},
                {(1, 'nsubj'), (3, 'dobj'), (4, 'prep')},
                set(),
                {(5, 'pobj')},
                set()]

    assert [set(),
                {(0, 'amod')},
                {(1, 'nsubj'), (0, 'nsubj+amod'), (3, 'dobj'), (4, 'prep'), (5, 'prep+pobj')},
                set(),
                {(5, 'pobj')},
                set()] == mangoes.context.DependencyBasedContext.add_children(sentence_sd, depth=2)
    
    collapsed_sentence_sd = [set(),
                            {(0, 'amod')},
                            {(1, 'nsubj'), (3, 'dobj'), (5, 'prep_with')},
                            set(),
                            set(),
                            set()]

    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (0, 'nsubj+amod'), (3, 'dobj'), (5, 'prep_with')},
            set(),
            set(),
            set()] == mangoes.context.DependencyBasedContext.add_children(collapsed_sentence_sd, depth=2)

    sentence_ud = [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod')},
            set(),
            set(),
            {(4, 'case')}]
            
    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod'), (0,"nsubj+amod"), (4, "nmod+case")},
            set(),
            set(),
            {(4, 'case')}] == mangoes.context.DependencyBasedContext.add_children(sentence_ud, depth=2)

    sentence_ud_2 = [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (7, 'nmod'), (0, 'nsubj+amod'), (4, 'nmod+case'), (6, "nmod+amod")},
            set(),
            set(),
            set(),
            {(5, "advmod")},
            {(6, "amod"), (5,"amod+advmod")}]

    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (7, 'nmod'), (0, 'nsubj+amod'), (4, 'nmod+case'), (6, "nmod+amod"), (5,"nmod+amod+advmod")},
            set(),
            set(),
            set(),
            {(5, "advmod")},
            {(6, "amod"), (5,"amod+advmod")}] == mangoes.context.DependencyBasedContext.add_children(sentence_ud_2, depth=3)

    collapsed_sentence_ud = [set(),
                            {(0, 'amod')},
                            {(1, 'nsubj'), (3, 'dobj'), (5, 'case_with')},
                            set(),
                            set(),
                            set()]

    assert [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (0,"nsubj+amod"), (5, "case_with")},
            set(),
            set(),
            set()] == mangoes.context.DependencyBasedContext.add_children(collapsed_sentence_ud, depth=2)

def test_add_length_path():
    sentence_sd = [set(),
                {(0, 'amod')},
                {(1, 'nsubj'), (3, 'dobj'), (4, 'prep')},
                set(),
                {(5, 'pobj')},
                set()]

    assert [{(1, 'amod'), (2, 'amod+nsubj')},
            {(0, 'amod'), (2, 'nsubj'), (4, 'nsubj+prep'), (3, 'nsubj+dobj')},
            {(0, 'nsubj+amod'), (1, 'nsubj'), (3, 'dobj'), (4, 'prep'), (5, 'prep+pobj')},
            {(2,'dobj'), (1, 'dobj+nsubj'), (4, 'dobj+prep')},
            {(5, 'pobj'), (2,'prep'), (3, 'prep+dobj'), (1, 'prep+nsubj')},
            {(4, 'pobj'), (2,'pobj+prep')}] == mangoes.context.DependencyBasedContext.add_length_path(sentence_sd, 2)
    
    collapsed_sentence_sd = [set(),
                            {(0, 'amod')},
                            {(1, 'nsubj'), (3, 'dobj'), (5, 'prep_with')},
                            set(),
                            set(),
                            set()]

    assert [{(1, 'amod'), (2, 'amod+nsubj')},
            {(0, 'amod'), (2, 'nsubj'), (5, 'nsubj+prep_with'), (3, 'nsubj+dobj')},
            {(0, 'nsubj+amod'), (1, 'nsubj'), (3, 'dobj'), (5, 'prep_with')},
            {(2,'dobj'), (1, 'dobj+nsubj'), (5, 'dobj+prep_with')},
            set(),
            {(3, 'prep_with+dobj'), (1, 'prep_with+nsubj'), (2,'prep_with')}] == mangoes.context.DependencyBasedContext.add_length_path(collapsed_sentence_sd, depth=2)

    sentence_ud = [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod')},
            set(),
            set(),
            {(4, 'case')}]
            
    assert [{(1, 'amod'), (2, 'amod+nsubj')},
            {(0, 'amod'), (2, 'nsubj'), (3, 'nsubj+dobj'), (5, 'nsubj+nmod')},
            {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod'), (0,"nsubj+amod"), (4, "nmod+case")},
            {(2, 'dobj'), (1, 'dobj+nsubj'), (5, 'dobj+nmod')},
            {(5, 'case'), (2, 'case+nmod')},
            {(4, 'case'), (2, 'nmod'), (3, 'nmod+dobj'), (1, 'nmod+nsubj')}] == mangoes.context.DependencyBasedContext.add_length_path(sentence_ud, depth=2)

    sentence_ud_2 = [set(),
            {(0, 'amod')},
            {(1, 'nsubj'), (3, 'dobj'), (7, 'nmod')},
            set(),
            set(),
            set(),
            {(5, "advmod")},
            {(6, "amod"), (4, 'case')}]

    assert [{(1, 'amod'), (2, 'amod+nsubj'), (3, 'amod+nsubj+dobj'), (7, 'amod+nsubj+nmod')},
            {(0, 'amod'), (2, 'nsubj'), (3, 'nsubj+dobj'), (7, 'nsubj+nmod'), (4, 'nsubj+nmod+case'), (6, 'nsubj+nmod+amod')},
            {(1, 'nsubj'), (3, 'dobj'), (7, 'nmod'), (0, 'nsubj+amod'), (4, 'nmod+case'), (6, "nmod+amod"), (5,"nmod+amod+advmod")},
            {(2, 'dobj'), (1, 'dobj+nsubj'), (0, 'dobj+nsubj+amod'), (7, 'dobj+nmod'), (4, 'dobj+nmod+case'),(6, 'dobj+nmod+amod')},
            {(7, 'case'), (2, 'case+nmod'), (3, 'case+nmod+dobj'), (1, 'case+nmod+nsubj'), (6, 'case+amod'), (5, 'case+amod+advmod')},
            {(6, 'advmod'),(7, 'advmod+amod'), (4, 'advmod+amod+case'), (2, 'advmod+amod+nmod')},
            {(5, "advmod"), (7, 'amod'), (4, 'amod+case'), (2, 'amod+nmod'), (3, 'amod+nmod+dobj'), (1, 'amod+nmod+nsubj')},
            {(6, "amod"), (5,"amod+advmod"), (4, 'case'), (2, 'nmod'), (3, 'nmod+dobj'), (1, 'nmod+nsubj'), (0, 'nmod+nsubj+amod')}
            ] == mangoes.context.DependencyBasedContext.add_length_path(sentence_ud_2, depth=3)

@pytest.mark.parametrize(['source', 'reader'],
                         [(conllu_string_ud_2, mangoes.utils.reader.ConllUSentenceGenerator),])
def test_dependency_based_context_ud_deprel(source, reader):
    corpus = mangoes.Corpus(source, reader=reader)
    sentence = corpus.reader.sentences().__next__()

    dep_based_context =  mangoes.context.DependencyBasedContext

    assert [{"scientist/amod/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1"},  # scientist
            {"scientist/nsubj/1", "star/dobj/1", "telescope/nmod/1"},  # discovers
            {"discovers/dobj/1"},  # star
            {"telescope/case/1"},  # with
            {"large/advmod/1"}, # very
            {"telescope/amod/1", "very/advmod/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1"} # telescope
            ] == dep_based_context(entity="form", depth=1, labels=True)(sentence)
        
    assert [{"scientist/amod/1", "discovers/amod+nsubj/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "star/nsubj+dobj/1", "telescope/nsubj+nmod/1"},  # scientist
            {"scientist/nsubj/1", "star/dobj/1","telescope/nmod/1", "australian/nsubj+amod/1","large/nmod+amod/1", "with/nmod+case/1"},  # discovers
            {"discovers/dobj/1", "scientist/dobj+nsubj/1", "telescope/dobj+nmod/1"},  # star
            {"telescope/case/1", "discovers/case+nmod/1", "large/case+amod/1"},  # with
            {"large/advmod/1", "telescope/advmod+amod/1"}, # very
            {"telescope/amod/1", "discovers/amod+nmod/1", "very/advmod/1", "with/amod+case/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1", "very/amod+advmod/1", "star/nmod+dobj/1", "scientist/nmod+nsubj/1"} # telescope
            ] == dep_based_context(entity="form", depth=2, labels=True)(sentence)

    assert [{"scientist/amod/1", "discovers/amod+nsubj/1", "star/amod+nsubj+dobj/1", "telescope/amod+nsubj+nmod/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "star/nsubj+dobj/1", "telescope/nsubj+nmod/1", "with/nsubj+nmod+case/1", "large/nsubj+nmod+amod/1"},  # scientist
            {"scientist/nsubj/1", "star/dobj/1","telescope/nmod/1", "australian/nsubj+amod/1","large/nmod+amod/1", "with/nmod+case/1", "very/nmod+amod+advmod/1"},  # discovers
            {"discovers/dobj/1", "scientist/dobj+nsubj/1", "telescope/dobj+nmod/1", "australian/dobj+nsubj+amod/1", "with/dobj+nmod+case/1", "large/dobj+nmod+amod/1"},  # star
            {"telescope/case/1", "discovers/case+nmod/1", "large/case+amod/1", "very/case+amod+advmod/1", "star/case+nmod+dobj/1", "scientist/case+nmod+nsubj/1"},  # with
            {"large/advmod/1", "telescope/advmod+amod/1", "with/advmod+amod+case/1", "discovers/advmod+amod+nmod/1"}, # very
            {"telescope/amod/1", "discovers/amod+nmod/1", "very/advmod/1", "with/amod+case/1", "star/amod+nmod+dobj/1", "scientist/amod+nmod+nsubj/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1", "very/amod+advmod/1", "star/nmod+dobj/1", "scientist/nmod+nsubj/1", "australian/nmod+nsubj+amod/1"} # telescope
            ] == dep_based_context(entity="form", depth=3, labels=True)(sentence)

    assert [{"scientist/1"},  # australian
            {"discovers/1", "australian/1"},  # scientist
            {"scientist/1", "telescope/1"},  # discovers
            set(),  # star
            {"telescope/1"},  # with
            set(), # very
            {"telescope/1"}, # large
            {"discovers/1", "large/1", "with/1"} # telescope
            ] == dep_based_context(entity="form", depth=1, deprel_keep=("nmod", "nsubj", "amod"))(sentence)
        
    assert [{"scientist/1", "discovers/1"},  # australian
            {"discovers/1", "australian/1", "telescope/1"},  # scientist
            {"scientist/1", "telescope/1", "australian/1","large/1", "with/1"},  # discovers
            set(),  # star
            {"telescope/1", "discovers/1", "large/1"},  # with
            set(), # very
            {"telescope/1", "discovers/1", "with/1"}, # large
            {"discovers/1", "large/1", "with/1", "scientist/1"} # telescope
            ] == dep_based_context(entity="form", depth=2, deprel_keep=("nmod", "nsubj", "amod"))(sentence)

    assert [{"scientist/1", "discovers/1", "telescope/1"},  # australian
            {"discovers/1", "australian/1", "telescope/1", "with/1", "large/1"},  # scientist
            {"scientist/1", "telescope/1", "australian/1","large/1", "with/1"},  # discovers
            set(),  # star
            {"telescope/1", "discovers/1", "large/1", "scientist/1"},  # with
            set(), # very
            {"telescope/1", "discovers/1", "with/1", "scientist/1"}, # large
            {"discovers/1", "large/1", "with/1", "scientist/1", "australian/1"} # telescope
            ] == dep_based_context(entity="form", depth=3, deprel_keep=("nmod", "amod", "nsubj"))(sentence)

    assert [{"scientist/amod/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1"},  # scientist
            {"scientist/nsubj/1", "telescope/nmod/1"},  # discovers
            set(),  # star
            {"telescope/case/1"},  # with
            set(), # very
            {"telescope/amod/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1"} # telescope
            ]  == dep_based_context(entity="form", labels=True, depth=1, deprel_keep=("nmod", "nsubj", "amod"))(sentence)
 
    assert [{"scientist/amod/1", "discovers/amod+nsubj/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "telescope/nsubj+nmod/1"},  # scientist
            {"scientist/nsubj/1", "telescope/nmod/1", "australian/nsubj+amod/1","large/nmod+amod/1", "with/nmod+case/1"},  # discovers
            set(),  # star
            {"telescope/case/1", "discovers/case+nmod/1", "large/case+amod/1"},  # with
            set(), # very
            {"telescope/amod/1", "discovers/amod+nmod/1", "with/amod+case/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1", "scientist/nmod+nsubj/1"} # telescope
            ] == dep_based_context(entity="form", labels=True, depth=2, deprel_keep=("nmod", "nsubj", "amod"))(sentence)

    assert [{"scientist/amod/1", "discovers/amod+nsubj/1", "telescope/amod+nsubj+nmod/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "telescope/nsubj+nmod/1", "with/nsubj+nmod+case/1", "large/nsubj+nmod+amod/1"},  # scientist
            {"scientist/nsubj/1", "telescope/nmod/1", "australian/nsubj+amod/1","large/nmod+amod/1", "with/nmod+case/1"},  # discovers
            set(),  # star
            {"telescope/case/1", "discovers/case+nmod/1", "large/case+amod/1", "scientist/case+nmod+nsubj/1"},  # with
            set(), # very
            {"telescope/amod/1", "discovers/amod+nmod/1", "with/amod+case/1", "scientist/amod+nmod+nsubj/1"}, # large
            {"discovers/nmod/1", "large/amod/1", "with/case/1", "scientist/nmod+nsubj/1", "australian/nmod+nsubj+amod/1"} # telescope
            ] == dep_based_context(entity="form", labels=True, depth=3, deprel_keep=("nmod", "amod", "nsubj"))(sentence)

    assert [{"scientist/amod/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1"},  # scientist
            {"scientist/nsubj/1", "telescope/case_with/1"},  # discovers
            set(),  # star
            set(),  # with
            set(), # very
            {"telescope/amod/1"}, # large
            {"discovers/case_with/1", "large/amod/1"} # telescope
            ] == dep_based_context(entity="form", labels=True, collapse=True, depth=1, deprel_keep=("nmod", "nsubj", "amod"))(sentence)
 
    assert [{"scientist/amod/1", "discovers/amod+nsubj/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "telescope/nsubj+case_with/1"},  # scientist
            {"scientist/nsubj/1", "telescope/case_with/1", "australian/nsubj+amod/1","large/case_with+amod/1"},  # discovers
            set(),  # star
            set(),  # with
            set(), # very
            {"telescope/amod/1", "discovers/amod+case_with/1"}, # large
            {"discovers/case_with/1", "large/amod/1", "scientist/case_with+nsubj/1"} # telescope
            ] == dep_based_context(entity="form", labels=True, collapse=True, depth=2, deprel_keep=("nmod", "nsubj", "amod"))(sentence)

    assert [{"scientist/amod/1", "discovers/amod+nsubj/1", "telescope/amod+nsubj+case_with/1"},  # australian
            {"discovers/nsubj/1", "australian/amod/1", "telescope/nsubj+case_with/1",  "large/nsubj+case_with+amod/1"},  # scientist
            {"scientist/nsubj/1", "telescope/case_with/1", "australian/nsubj+amod/1","large/case_with+amod/1"},  # discovers
            set(),  # star
            set(),  # with
            set(), # very
            {"telescope/amod/1", "discovers/amod+case_with/1", "scientist/amod+case_with+nsubj/1"}, # large
            {"discovers/case_with/1", "large/amod/1", "scientist/case_with+nsubj/1", "australian/case_with+nsubj+amod/1"} # telescope
            ] == dep_based_context(entity="form", labels=True, collapse=True, depth=3, deprel_keep=("nmod", "amod", "nsubj"))(sentence)
