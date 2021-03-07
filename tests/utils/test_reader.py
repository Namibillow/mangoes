# -*- coding: utf-8 -*-
import mangoes.utils.reader
import pytest


# Raw text, no annotation
@pytest.mark.unittest
def test_raw_reader_from_list_of_strings(raw_source_string, raw_sentences):
    reader = mangoes.utils.reader.TextSentenceGenerator(raw_source_string)
    for i, sentence in enumerate(reader.sentences()):
        assert raw_sentences[i] == sentence
    # TODO : mock mangoes.utils.io.get_reader to make it really unit (idem for the other ones)


@pytest.mark.integration
def test_raw_reader_from_one_file(raw_source_file, raw_sentences):
    reader = mangoes.utils.reader.TextSentenceGenerator(raw_source_file)
    for i, sentence in enumerate(reader.sentences()):
        assert raw_sentences[i] == sentence


@pytest.mark.integration
def test_raw_reader_from_one_dir(raw_source_dir, raw_sentences):
    reader = mangoes.utils.reader.TextSentenceGenerator(raw_source_dir)
    for i, sentence in enumerate(reader.sentences()):
        assert raw_sentences[i] == sentence


# Annotated text : 3 formats : brown, xml or conll
# ### XML
@pytest.mark.unittest
def test_xml_reader_from_xml_string(xml_source_string, fully_annotated_sentences):
    reader = mangoes.utils.reader.XmlSentenceGenerator(xml_source_string)
    for i, sentence in enumerate(reader.sentences()):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_xml_reader_from_xml_file(xml_source_file, fully_annotated_sentences):
    reader = mangoes.utils.reader.XmlSentenceGenerator(xml_source_file)
    for i, sentence in enumerate(reader.sentences()):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_xml_reader_from_xml_dir(xml_source_dir, fully_annotated_sentences):
    reader = mangoes.utils.reader.XmlSentenceGenerator(xml_source_dir)
    for i, sentence in enumerate(reader.sentences()):
        assert fully_annotated_sentences[i] == sentence


# ### BROWN
@pytest.mark.unittest
def test_brown_reader_from_brown_string(brown_source_string, annotated_sentences):
    reader = mangoes.utils.reader.BrownSentenceGenerator(brown_source_string)
    for i, sentence in enumerate(reader.sentences()):
        assert annotated_sentences[i] == sentence


@pytest.mark.integration
def test_brown_reader_from_brown_file(brown_source_file, annotated_sentences):
    reader = mangoes.utils.reader.BrownSentenceGenerator(brown_source_file)
    for i, sentence in enumerate(reader.sentences()):
        assert annotated_sentences[i] == sentence


@pytest.mark.integration
def test_brown_reader_from_brown_dir(brown_source_dir, annotated_sentences):
    reader = mangoes.utils.reader.BrownSentenceGenerator(brown_source_dir)
    for i, sentence in enumerate(reader.sentences()):
        assert annotated_sentences[i] == sentence


# ### CONLL
@pytest.mark.unittest
def test_conll_reader_from_conll_string(conll_source_string, fully_annotated_sentences):
    reader = mangoes.utils.reader.ConllSentenceGenerator(conll_source_string)
    for i, sentence in enumerate(reader.sentences()):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_conll_reader_from_conll_file(conll_source_file, fully_annotated_sentences):
    reader = mangoes.utils.reader.ConllSentenceGenerator(conll_source_file)
    for i, sentence in enumerate(reader.sentences()):
        assert fully_annotated_sentences[i] == sentence


@pytest.mark.integration
def test_conll_reader_from_conll_dir(conll_source_dir, fully_annotated_sentences):
    reader = mangoes.utils.reader.ConllSentenceGenerator(conll_source_dir)
    for i, sentence in enumerate(reader.sentences()):
        assert sentence in fully_annotated_sentences


def test_conllu_reader():
    conll_source_string = """1	australian	_	JJ	_	_	2	amod	_	_
                             2	scientist	_	NN	_	_	3	nsubj	_	_
                             3	discovers	_	VBZ	_	_	0	root	_	_
                             4	star	_	NN	_	_	3	dobj	_	_
                             5	with	_	IN	_	_	3	prep	_	_
                             6	telescope	_	NN	_	_	5	pobj	_	_""".split("\n")

    expected = [[('1', 'australian', '_', 'JJ', '_', '_',  '2', 'amod', '_', '_'),
                 ('2', 'scientist', '_', 'NN', '_', '_', '3', 'nsubj', '_', '_'),
                 ('3', 'discovers', '_', 'VBZ', '_', '_', '0', 'root', '_', '_'),
                 ('4', 'star', '_', 'NN', '_', '_', '3', 'dobj', '_', '_'),
                 ('5', 'with', '_', 'IN', '_', '_', '3', 'prep', '_', '_'),
                 ('6', 'telescope', '_', 'NN', '_', '_', '5', 'pobj', '_', '_')]]

    reader = mangoes.utils.reader.ConllUSentenceGenerator(conll_source_string)

    for i, sentence in enumerate(reader.sentences()):
        assert expected[i] == sentence


def test_xml_reader_with_dependencies():
    xml_string = """<root>
                      <document>
                        <sentences>
                          <sentence id="1">
                            <tokens>
                              <token id="1">
                                <word>australian</word>
                                <lemma>australian</lemma>
                                <POS>JJ</POS>
                              </token>
                              <token id="2">
                                <word>scientist</word>
                                <lemma>scientist</lemma>
                                <POS>NN</POS>
                              </token>
                              <token id="3">
                                <word>discovers</word>
                                <lemma>discover</lemma>
                                <POS>VBZ</POS>
                              </token>
                              <token id="4">
                                <word>star</word>
                                <lemma>star</lemma>
                                <POS>NN</POS>
                              </token>
                              <token id="5">
                                <word>with</word>
                                <lemma>with</lemma>
                                <POS>IN</POS>
                              </token>
                              <token id="6">
                                <word>telescope</word>
                                <lemma>telescope</lemma>
                                <POS>NN</POS>
                              </token>
                            </tokens>
                            <dependencies type="basic-dependencies">
                              <dep type="root">
                                <governor idx="0">ROOT</governor>
                                <dependent idx="3">discovers</dependent>
                              </dep>
                              <dep type="amod">
                                <governor idx="2">scientist</governor>
                                <dependent idx="1">australian</dependent>
                              </dep>
                              <dep type="nsubj">
                                <governor idx="3">discovers</governor>
                                <dependent idx="2">scientist</dependent>
                              </dep>
                              <dep type="dobj">
                                <governor idx="3">discovers</governor>
                                <dependent idx="4">star</dependent>
                              </dep>
                              <dep type="prep">
                                <governor idx="3">discovers</governor>
                                <dependent idx="5">with</dependent>
                              </dep>
                              <dep type="pobj">
                                <governor idx="5">with</governor>
                                <dependent idx="6">telescope</dependent>
                              </dep>
                            </dependencies>
                          </sentence>
                        </sentences>
                      </document>
                    </root>"""

    reader = mangoes.utils.reader.XmlSentenceGenerator(xml_string)

    expected = [[('1', 'australian', 'australian', 'JJ', '_', '2', 'amod'),
                 ('2', 'scientist', 'scientist', 'NN', '_', '3', 'nsubj'),
                 ('3', 'discovers', 'discover', 'VBZ', '_', '0', 'root'),
                 ('4', 'star', 'star', 'NN', '_', '3', 'dobj'),
                 ('5', 'with', 'with', 'IN', '_', '3', 'prep'),
                 ('6', 'telescope', 'telescope', 'NN', '_', '5', 'pobj')]]

    for i, sentence in enumerate(reader.sentences()):
        assert expected[i] == sentence
