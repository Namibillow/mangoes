# -*- coding: utf-8 -*-

import logging

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from mangoes.modeling import CustomTokenizer, ByteBPETokenizer, CharacterBPETokenizer, WordLevelTokenizer, \
    SentPieceBPETokenizer, BERTWordPieceTokenizer, BERTForMaskedLanguageModeling
from mangoes import Vocabulary

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_wordlevel_tokenizer_dummy_vocab_initialization(extra_small_word_index):
    tokenizer = WordLevelTokenizer(extra_small_word_index)

    assert tokenizer.encode("this is a test").ids == [0, 1, 2, 3]
    assert tokenizer.encode("unknown").ids == [4]


@pytest.mark.unittest
def test_wordlevel_tokenizer_normalizers(extra_small_word_index):
    tokenizer = WordLevelTokenizer(extra_small_word_index, lowercase=True)
    assert tokenizer.encode("This is A Test").ids == [0, 1, 2, 3]

    tokenizer = WordLevelTokenizer(extra_small_word_index, unicode_normalizer='nfc', lowercase=True)
    assert tokenizer.encode("This is A Test").ids == [0, 1, 2, 3]


@pytest.mark.unittest
def test_wordlevel_tokenizer_fromfile(extra_small_word_index_file):

    tokenizer = WordLevelTokenizer.from_file(extra_small_word_index_file)
    assert tokenizer.encode("this is a test").ids == [0, 1, 2, 3]

    tokenizer = WordLevelTokenizer(extra_small_word_index_file)
    assert tokenizer.encode("this is a test").ids == [0, 1, 2, 3]


@pytest.mark.unittest
def test_wordlevel_tokenizer_training(extra_small_source_file):
    tokenizer = WordLevelTokenizer()
    tokenizer.train(extra_small_source_file)

    assert tokenizer.encode("This is a test").ids == [1, 2, 3, 4]

    tokenizer = WordLevelTokenizer(lowercase=True)
    tokenizer.train(extra_small_source_file)

    assert tokenizer.encode("This is A test").ids == [1, 2, 3, 4]


@pytest.mark.unittest
def test_custom_tokenizer_training(raw_source_file):
    # initialize huggingface tokenizer and trainer
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # initialize mangoes tokenizer
    tokenizer = CustomTokenizer(tok)
    tokenizer.train(trainer=trainer, files=raw_source_file)
    output = tokenizer.encode("this is a test")
    assert output.tokens == ['t', 'h', 'is', 'is', 'a', 'te', 's', 't']
    assert output.ids == [26, 17, 34, 34, 10, 30, 25, 26]
    assert output.offsets == [(0, 1), (1, 2), (2, 4), (5, 7), (8, 9), (10, 12), (12, 13), (13, 14)]


@pytest.mark.unittest
def test_custom_tokenizer_from_file(raw_source_file, tmpdir_factory):
    # initialize huggingface tokenizer and trainer
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # train and save tokenizer
    tok.train(trainer=trainer, files=[raw_source_file])
    tok_path = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok.save(str(tok_path))

    # initialize mangoes tokenizer
    tokenizer = CustomTokenizer.from_file(str(tok_path))

    output = tokenizer.encode("this is a test")
    assert output.tokens == ['t', 'h', 'is', 'is', 'a', 'te', 's', 't']
    assert output.ids == [26, 17, 34, 34, 10, 30, 25, 26]
    assert output.offsets == [(0, 1), (1, 2), (2, 4), (5, 7), (8, 9), (10, 12), (12, 13), (13, 14)]


@pytest.mark.unittest
def test_custom_tokenizer_save(raw_source_file, tmpdir_factory):
    # initialize huggingface tokenizer and trainer
    tok = Tokenizer(BPE())
    tok.pre_tokenizer = Whitespace()
    tokenizer = CustomTokenizer(tok)
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    # train and save tokenizer
    tokenizer.train(trainer=trainer, files=raw_source_file)
    tok_dir = tmpdir_factory.mktemp('data')
    tok_path = tokenizer.save(str(tok_dir))
    assert tok_path == tok_dir.join("tokenizer.json")

    tok_path_orig = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok_path = tokenizer.save(str(tok_path_orig))
    assert tok_path_orig == tok_path


@pytest.mark.unittest
def test_char_bpe_tokenizer_from_file(raw_source_file, tmpdir_factory):
    tokenizer = CharacterBPETokenizer()
    tokenizer.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tokenizer.save_model(str(tok_path))

    new_tok = CharacterBPETokenizer.from_file(str(tok_path) + "/vocab.json", str(tok_path) + "/merges.txt")
    output = new_tok.encode("this is test")
    assert output.tokens == ['t', 'h', 'is</w>', 'is</w>', 'te', 's', 't</w>']
    assert output.offsets == [(0, 1), (1, 2), (2, 4), (5, 7), (8, 10), (10, 11), (11, 12)]


@pytest.mark.unittest
def test_char_bpe_tokenizer_save(raw_source_file, tmpdir_factory):
    tokenizer = CharacterBPETokenizer()
    tokenizer.train(raw_source_file)
    tok_dir = tmpdir_factory.mktemp('data')
    tok_path = tokenizer.save(str(tok_dir))
    assert tok_path == tok_dir.join("tokenizer.json")

    tok_path_orig = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok_path = tokenizer.save(str(tok_path_orig))
    assert tok_path_orig == tok_path


@pytest.mark.unittest
def test_byte_bpe_tokenizer_from_file(raw_source_file, tmpdir_factory):
    tokenizer = ByteBPETokenizer()
    tokenizer.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tokenizer.save_model(str(tok_path))

    new_tok = ByteBPETokenizer.from_file(str(tok_path) + "/vocab.json", str(tok_path) + "/merges.txt")
    output = new_tok.encode("this is test")
    assert output.tokens == ['t', 'h', 'i', 's', 'Ġis', 'Ġ', 'te', 's', 't']
    assert output.offsets == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (7, 8), (8, 10), (10, 11), (11, 12)]


@pytest.mark.unittest
def test_byte_bpe_tokenizer_save(raw_source_file, tmpdir_factory):
    tokenizer = ByteBPETokenizer()
    tokenizer.train(raw_source_file)
    tok_dir = tmpdir_factory.mktemp('data')
    tok_path = tokenizer.save(str(tok_dir))
    assert tok_path == tok_dir.join("tokenizer.json")

    tok_path_orig = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok_path = tokenizer.save(str(tok_path_orig))
    assert tok_path_orig == tok_path


@pytest.mark.unittest
def test_sentencepiece_bpe_tokenizer_from_file(raw_source_file_no_newlines, tmpdir_factory):
    tokenizer = SentPieceBPETokenizer()
    tokenizer.train(raw_source_file_no_newlines)
    tok_path = tmpdir_factory.mktemp('data')
    tokenizer.save_model(str(tok_path))

    new_tok = SentPieceBPETokenizer.from_file(str(tok_path) + "/vocab.json", str(tok_path) + "/merges.txt")
    output = new_tok.encode("this is test")
    assert output.tokens == ['▁', 't', 'h', 'i', 's', '▁is', '▁', 'te', 's', 't']
    assert output.offsets == [(0, 1), (0, 1), (1, 2), (2, 3), (3, 4), (4, 7), (7, 8), (8, 10), (10, 11), (11, 12)]


def test_sentencepiece_bpe_tokenizer_save(raw_source_file_no_newlines, tmpdir_factory):
    tokenizer = SentPieceBPETokenizer()
    tokenizer.train(raw_source_file_no_newlines)
    tok_dir = tmpdir_factory.mktemp('data')
    tok_path = tokenizer.save(str(tok_dir))
    assert tok_path == tok_dir.join("tokenizer.json")

    tok_path_orig = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok_path = tokenizer.save(str(tok_path_orig))
    assert tok_path_orig == tok_path


@pytest.mark.unittest
def test_bert_wordpiece_tokenizer_from_file(raw_source_file_no_newlines, tmpdir_factory):
    tokenizer = BERTWordPieceTokenizer()
    tokenizer.train(raw_source_file_no_newlines)
    tok_path = tmpdir_factory.mktemp('data')
    tokenizer.save_model(str(tok_path))

    new_tok = BERTWordPieceTokenizer.from_file(str(tok_path) + "/vocab.txt")
    output = new_tok.encode("this is test")
    assert output.tokens == ['[CLS]', 'th', '##i', '##s', 'is', 't', '##e', '##s', '##t', '[SEP]']
    assert output.offsets == [(0, 0), (0, 2), (2, 3), (3, 4), (5, 7), (8, 9), (9, 10), (10, 11), (11, 12), (0, 0)]


@pytest.mark.unittest
def test_bert_wordpiece_tokenizer_save(raw_source_file_no_newlines, tmpdir_factory):
    tokenizer = BERTWordPieceTokenizer()
    tokenizer.train(raw_source_file_no_newlines)
    tok_dir = tmpdir_factory.mktemp('data')
    tok_path = tokenizer.save(str(tok_dir))
    assert tok_path == tok_dir.join("tokenizer.json")

    tok_path_orig = tmpdir_factory.mktemp('data').join('test_trained_bpe.json')
    tok_path = tokenizer.save(str(tok_path_orig))
    assert tok_path_orig == tok_path

# ###########################################################################################
# ### Integration tests


# Integration with Vocab
@pytest.mark.integration
def test_tokenizer_mixin_class(extra_small_word_index):
    tokenizer = WordLevelTokenizer(extra_small_word_index)
    tokenizer_vocab = tokenizer.make_vocab_object()
    mangoes_vocab = Vocabulary(extra_small_word_index)

    assert tokenizer_vocab == mangoes_vocab


@pytest.mark.integration
def test_tokenizer_bert_integration(raw_source_file_no_newlines, tmpdir_factory):
    tokenizer = BERTWordPieceTokenizer()
    tokenizer.train(raw_source_file_no_newlines)
    tok_dir = tmpdir_factory.mktemp('saved_tokenizer')
    tokenizer.save(str(tok_dir))
    output1 = tokenizer.encode("this is test")
    model = BERTForMaskedLanguageModeling(str(tok_dir), hidden_size=252, num_hidden_layers=1)
    output = model.tokenizer("this is test")
    assert output.input_ids == output1.ids



