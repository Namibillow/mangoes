import pytest

ENCODING = "UTF-8"


@pytest.fixture
def save_temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('tmp')

# #################################################################
#    _____
#   |  __ \
#   | |__) | __ _ __      __     ___   ___   _   _  _ __  ___  ___
#   |  _  / / _` |\ \ /\ / /    / __| / _ \ | | | || '__|/ __|/ _ \
#   | | \ \| (_| | \ V  V /     \__ \| (_) || |_| || |  | (__|  __/
#   |_|  \_\\__,_|  \_/\_/      |___/ \___/  \__,_||_|   \___|\___|


# Raw text, no annotation
RAW_SOURCE = """Beautiful is better than ugly
             Explicit is better than implicit
             Simple is better than complex
             Complex is better than complicated
             Flat is better than nested
             Sparse is better than dense""".split("\n")

RAW_SOURCE_SMALL = ["I'm a new sentence", "For testing purposes"]
RAW_SOURCE_EXTRA_SMALL = "This is a test"
RAW_SOURCE_FUTURE_SUBTOKENS = "A word to split into subtokens is subtokens"

QUESTIONS = ["What is the context for this question?", "What kind of question is this?"]
CONTEXTS = ["This is context for the test questions.", "This is context for the test question."]
ANSWERS = ["This", "test"]
ANSWER_START_INDICES = [CONTEXTS[i].find(ANSWERS[i]) for i in range(len(ANSWERS))]


@pytest.fixture()
def raw_future_subtokens_source_string():
    return RAW_SOURCE_FUTURE_SUBTOKENS


@pytest.fixture()
def extra_small_word_index():
    return {"this": 0, "is": 1, "a": 2, "test": 3}


@pytest.fixture()
def extra_small_word_index_file(tmpdir_factory, extra_small_word_index):
    import json
    source_file = tmpdir_factory.mktemp('data').join('RAW_SOURCE_EXTRA_SMALL_index.txt')
    with open(source_file, "w") as f:
        json.dump(extra_small_word_index, f)
    return str(source_file)


@pytest.fixture()
def extra_small_source_file(tmpdir_factory):
    source_file = tmpdir_factory.mktemp('data').join('RAW_SOURCE_EXTRA_SMALL.txt')
    source_file.write_text(RAW_SOURCE_EXTRA_SMALL+"\n", encoding=ENCODING)
    return str(source_file)


@pytest.fixture()
def raw_source_string():
    return RAW_SOURCE


@pytest.fixture()
def raw_small_source_string():
    return RAW_SOURCE_SMALL


@pytest.fixture()
def raw_source_file(tmpdir_factory):
    source_file = tmpdir_factory.mktemp('data').join('RAW_SOURCE.txt')
    source_file.write_text("\n".join(RAW_SOURCE), encoding=ENCODING)
    return str(source_file)


@pytest.fixture()
def raw_source_file_no_newlines(tmpdir_factory):
    source_file = tmpdir_factory.mktemp('data').join('RAW_SOURCE.txt')
    source_file.write_text(" ".join(RAW_SOURCE), encoding=ENCODING)
    return str(source_file)


@pytest.fixture()
def raw_source_dir(tmpdir_factory):
    source_dir = tmpdir_factory.mktemp('data')
    for i, s in enumerate(RAW_SOURCE):
        f = source_dir.join('source_{}.txt'.format(i))
        f.write_text(s, encoding=ENCODING)
    return str(source_dir)


# From raw source, expected sentences
@pytest.fixture()
def raw_sentences(raw_source_string):
    return [sentence.split() for sentence in raw_source_string]


@pytest.fixture()
def raw_sentences_small(raw_small_source_string):
    return [sentence.split() for sentence in raw_small_source_string]


@pytest.fixture()
def raw_sentences_lowered(raw_source_string):
    return [sentence.lower().split() for sentence in raw_source_string]


@pytest.fixture()
def question_answering_data():
    return QUESTIONS, CONTEXTS, ANSWERS, ANSWER_START_INDICES


# From raw source, return a fake corpus
@pytest.fixture()
def dummy_raw_corpus(raw_source_string):
    class FakeVocabulary:
        def __init__(self):
            self.words = ['is', 'better', 'than', 'beautiful', 'ugly', 'explicit', 'implicit', 'simple', 'complex',
                          'complicated', 'flat', 'nested', 'sparse', 'dense']
            self.word_index = {word: index for index, word in enumerate(self.words)}
            self.entity = None
            self.params = None

        def __len__(self):
            return 14

        def index(self, word):
            return self.words.index(word)

        def __contains__(self, word):
            return word in self.word_index

        def __iter__(self):
            return self.words.__iter__()

    class DummyRawCorpus:
        def __init__(self):
            self.source = raw_source_string
            self.nb_sentences = 6
            self.size = 30
            self.words_count = {'is': 6, 'better': 6, 'than': 6,
                                'beautiful': 1, 'ugly': 1, 'explicit': 1, 'implicit': 1,
                                'simple': 1, 'complex': 2, 'complicated': 1,
                                'flat': 1, 'nested': 1, 'sparse': 1, 'dense': 1}
            self.params = {"name": "dummy"}
            self.annotated = False
            self.reader = self
            self.reader.sentences = self.__iter__

        def __iter__(self):
            for sentence in self.source:
                yield sentence.lower().split()

        def create_vocabulary(self):
            return FakeVocabulary()

    return DummyRawCorpus()


# ###################################################################################################
#                                   _          _             _
#       /\                         | |        | |           | |
#      /  \    _ __   _ __    ___  | |_  __ _ | |_  ___   __| |    ___   ___   _   _  _ __  ___  ___
#     / /\ \  | '_ \ | '_ \  / _ \ | __|/ _` || __|/ _ \ / _` |   / __| / _ \ | | | || '__|/ __|/ _ \
#    / ____ \ | | | || | | || (_) || |_| (_| || |_|  __/| (_| |   \__ \| (_) || |_| || |  | (__|  __/
#   /_/    \_\|_| |_||_| |_| \___/  \__|\__,_| \__|\___| \__,_|   |___/ \___/  \__,_||_|   \___|\___|

# Annotated text
ANNOTATED_SOURCE = """Beautiful/JJ/beautiful is/VBZ/be better/JJR/better than/IN/than ugly/JJ/ugly
                      Explicit/NNP/Explicit is/VBZ/be better/JJR/better than/IN/than implicit/JJ/implicit
                      Simple/NN/simple is/VBZ/be better/JJR/better than/IN/than complex/JJ/complex
                      Complex/NN/complex is/VBZ/be better/JJR/better than/IN/than complicated/VBN/complicate
                      Flat/NNP/Flat is/VBZ/be better/JJR/better than/IN/than nested/JJ/nested
                      Sparse/NNP/Sparse is/VBZ/be better/JJR/better than/IN/than dense/JJ/dense""".split("\n")


# 6 sentences, 30 words
# is, better, than : 6 (-> 18)
# beautiful, ugly, explicit, implicit, simple, complicated, flat, nested, sparse, dense : 1 (-> 10)
# complex : 1 + 1 with capital -> 2
# 14 words if lower cased / 15 if case sensitive

# complex : 1 with pos NN + 1 with pos JJ

# From ANNOTATED_SOURCE, generate inputs for 3 formats : brown, xml or conll
@pytest.fixture()
def brown_source_string():
    return ANNOTATED_SOURCE


@pytest.fixture()
def brown_source_file(tmpdir_factory):
    source_file = tmpdir_factory.mktemp('data').join('BROWN_SOURCE.txt')
    source_file.write_text("\n".join(ANNOTATED_SOURCE), encoding=ENCODING)
    return str(source_file)


@pytest.fixture()
def brown_source_dir(tmpdir_factory):
    source_dir = tmpdir_factory.mktemp('data')
    for i, s in enumerate(ANNOTATED_SOURCE):
        f = source_dir.join('source_{}.txt'.format(i))
        f.write_text(s, encoding=ENCODING)
    return str(source_dir)


@pytest.fixture()
def xml_source_string():
    xml_string = "<root><document><sentences>"
    for sentence in ANNOTATED_SOURCE:
        xml_string += "<sentence><tokens>"
        for i, token in enumerate(sentence.split()):
            word, pos, lemma = token.split("/")
            xml_string += "<token id='{}'>".format(i + 1)
            xml_string += "<word>" + word + "</word>"
            xml_string += "<POS>" + pos + "</POS>"
            xml_string += "<lemma>" + lemma + "</lemma>"
            xml_string += "</token>"
        xml_string += "</tokens></sentence>"
    xml_string += "</sentences></document></root>"

    return xml_string


@pytest.fixture()
def xml_source_file(tmpdir_factory):
    source_file = tmpdir_factory.mktemp('data').join('RAW_SOURCE.xml')
    xml_string = "<root><document><sentences>"
    for sentence in ANNOTATED_SOURCE:
        xml_string += "<sentence><tokens>"
        for i, token in enumerate(sentence.split()):
            word, pos, lemma = token.split("/")
            xml_string += "<token id='{}'>".format(i + 1)
            xml_string += "<word>" + word + "</word>"
            xml_string += "<POS>" + pos + "</POS>"
            xml_string += "<lemma>" + lemma + "</lemma>"
            xml_string += "</token>"
        xml_string += "</tokens></sentence>"
    xml_string += "</sentences></document></root>"
    source_file.write_text(xml_string, encoding="utf-8")
    return str(source_file)


@pytest.fixture()
def xml_source_dir(tmpdir_factory):
    source_dir = tmpdir_factory.mktemp('data')
    for i, sentence in enumerate(ANNOTATED_SOURCE):
        source_file = source_dir.join('source_{}.xml'.format(i))
        xml_string = "<root><document><sentences><sentence><tokens>"
        for i, token in enumerate(sentence.split()):
            word, pos, lemma = token.split("/")
            xml_string += "<token id='{}'>".format(i + 1)
            xml_string += "<word>" + word + "</word>"
            xml_string += "<POS>" + pos + "</POS>"
            xml_string += "<lemma>" + lemma + "</lemma>"
            xml_string += "</token>"
        xml_string += "</tokens></sentence></sentences></document></root>"
        source_file.write_text(xml_string, encoding="utf-8")
    return str(source_dir)


@pytest.fixture()
def conll_source_string():
    conll_str = []
    for sentence in ANNOTATED_SOURCE:
        for i, token in enumerate(sentence.split()):
            word, pos, lemma = token.split("/")
            conll_str.append("{}\t{}\t{}\t{}\t_\t_\t_".format(i + 1, word, lemma, pos))
        conll_str.append("")
    del (conll_str[-1])
    return conll_str


@pytest.fixture()
def conll_source_file(tmpdir_factory, conll_source_string):
    source_file = tmpdir_factory.mktemp('data').join('source.conll')
    source_file.write('\n'.join(conll_source_string))
    return str(source_file)


@pytest.fixture()
def conll_source_dir(tmpdir_factory):
    source_dir = tmpdir_factory.mktemp('data')
    for i, sentence in enumerate(ANNOTATED_SOURCE):
        source_file = source_dir.join('source_{}.xml'.format(i))
        conll_string = ""
        for j, token in enumerate(sentence.split()):
            word, pos, lemma = token.split("/")
            conll_string += "{}\t{}\t{}\t{}\t_\t_\t_\n".format(j + 1, word, lemma, pos)
        source_file.write_text(conll_string + '\n', encoding="utf-8")
    return str(source_dir)


# From annotated source, expected sentences
from collections import namedtuple
SimpleToken = namedtuple("SimpleToken", "form lemma POS")
FullToken = namedtuple("FullToken", "id form lemma POS features head deprel")
FullToken.__new__.__defaults__ = ('_',) * 7


@pytest.fixture()
def annotated_sentences():
    expected = [[SimpleToken('Beautiful', 'beautiful', 'JJ'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('ugly', 'ugly', 'JJ')],

                [SimpleToken('Explicit', 'Explicit', 'NNP'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('implicit', 'implicit', 'JJ')],

                [SimpleToken('Simple', 'simple', 'NN'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('complex', 'complex', 'JJ')],

                [SimpleToken('Complex', 'complex', 'NN'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('complicated', 'complicate', 'VBN')],

                [SimpleToken('Flat', 'Flat', 'NNP'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('nested', 'nested', 'JJ')],

                [SimpleToken('Sparse', 'Sparse', 'NNP'), SimpleToken('is', 'be', 'VBZ'), SimpleToken('better', 'better', 'JJR'),
                 SimpleToken('than', 'than', 'IN'), SimpleToken('dense', 'dense', 'JJ')]]
    return expected


@pytest.fixture()
def fully_annotated_sentences():
    expected = [[FullToken('1', 'Beautiful', 'beautiful', 'JJ'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'ugly', 'ugly', 'JJ')],

                [FullToken('1', 'Explicit', 'Explicit', 'NNP'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'implicit', 'implicit', 'JJ')],

                [FullToken('1', 'Simple', 'simple', 'NN'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'complex', 'complex', 'JJ')],

                [FullToken('1', 'Complex', 'complex', 'NN'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'complicated', 'complicate', 'VBN')],

                [FullToken('1', 'Flat', 'Flat', 'NNP'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'nested', 'nested', 'JJ')],

                [FullToken('1', 'Sparse', 'Sparse', 'NNP'), FullToken('2', 'is', 'be', 'VBZ'), FullToken('3', 'better', 'better', 'JJR'),
                 FullToken('4', 'than', 'than', 'IN'), FullToken('5', 'dense', 'dense', 'JJ')]]
    return expected


# #############################################################################3
# Coreference testing data

coref_documents = [["This is a test sentence. It is short".split(),
                    "This is a test sentence. It is short but not that short".split(),
                    "This is a test sentence. It is short".split()],
                   ["This is a test sentence. It is short".split(),
                    "This is a test sentence. It is short but not that short".split()]]
coref_genres = [1, 1]
coref_speakers = [[[1 for _ in range(len(coref_documents[0][0]))],
                   [2 for _ in range(len(coref_documents[0][1]))],
                   [1 for _ in range(len(coref_documents[0][2]))]],
                  [[2 for _ in range(len(coref_documents[1][0]))],
                   [2 for _ in range(len(coref_documents[1][1]))]]]

coref_cluster_ids = [[[12, -1, -1, 12, -1, 12, -1, -1],
                      [14, -1, -1, 14, -1, 14, -1, -1, -1, -1, -1, -1],
                      [14, -1, -1, 12, -1, 14, -1, -1]],
                     [[13, -1, -1, 13, -1, 13, -1, -1],
                      [15, -1, -1, 15, -1, 15, -1, -1, -1, -1, -1, -1]]]


@pytest.fixture()
def raw_coref_data():
    return coref_documents, coref_cluster_ids, coref_speakers, coref_genres


# #############################################################################3
# multiple choice testing data

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choices = ["It is eaten with a fork and a knife.", "It is eaten while held in the hand."]


@pytest.fixture()
def multiple_choice_example():
    return prompt, choices
