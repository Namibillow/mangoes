# -*- coding: utf-8 -*-
"""
This module is for preprocessing text and training subword tokenizer for use in BERT like models in the
mangoes.modeling.bert modules.
It provides an interface into the huggingface tokenizers library.
"""
import os
import fileinput
import json
import warnings

import tokenizers

from mangoes import Vocabulary


class MixinToVocab:
    """
    Mixin class for creating Vocabulary class from tokenizer
    """
    def __init__(self):
        return

    def make_vocab_object(self):
        """
        Returns mangoes.Vocabulary object from tokenizer's vocab
        """
        return Vocabulary(self.get_vocab())


class CustomTokenizer(MixinToVocab, tokenizers.implementations.BaseTokenizer):
    """
    Class for using custom built tokenizer.
    Use this class if the other classes in this module do not contain the tokenizer you would like to use.
    This class inherits from tokenizers.BaseTokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/base_tokenizer.py

    Parameters
    ----------
    tokenizer: tokenizer.Tokenizer()
    """
    def __init__(self, tokenizer):
        tokenizers.implementations.BaseTokenizer.__init__(self, tokenizer)
        MixinToVocab.__init__(self)

    @staticmethod
    def from_file(path):
        tokenizer = tokenizers.Tokenizer.from_file(path)
        return CustomTokenizer(tokenizer)

    def train(self, trainer, files):
        """
        Train the tokenizer using a custom built Trainer

        Parameters
        ----------
        trainer: tokenizers.Trainer().
            Instantiated tokenizers.Trainer(). Make sure the trainer matches the tokenization model in the tokenizer
            attribute.
        files: file or list of files to train on.
        """
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer=trainer, files=files)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.implementations.BaseTokenizer.save(self, path, pretty)
        return path


class WordLevelTokenizer(MixinToVocab, tokenizers.implementations.BaseTokenizer):
    """
    Simple word level tokenization.
    This class inherits from tokenizers.BaseTokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/base_tokenizer.py

    Parameters
    ----------
    vocab: str
        path to saved vocab or dict of [str -> int], if vocab is precomputed
    unk_token: str
        token to use for unknown tokens, defaults to "<unk>"
    unicode_normalizer: str
        optional, one of "nfc", "nfd", "nfkc", "nfkd", or None
    lowercase: Boolean
        whether or not to lowercase all text as part of tokenization pipeline
    """
    def __init__(self, vocab=None, unk_token="<unk>", unicode_normalizer=None, lowercase=False):
        if not vocab:
            index = {unk_token: 0}
        elif isinstance(vocab, dict):
            if unk_token not in vocab:
                vocab[unk_token] = len(vocab)
            index = vocab
        else:
            # have to check if unknown token in vocab file, else causes error
            vocab_file = vocab
            with open(vocab_file, "r") as filename:
                index = json.load(filename)
            if unk_token not in index:
                index[unk_token] = len(index)

        tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(index, unk_token=unk_token))
        normalizers = []
        if unicode_normalizer:
            normalizers += [tokenizers.normalizers.unicode_normalizer_from_str(unicode_normalizer)]
        if lowercase:
            normalizers += [tokenizers.normalizers.Lowercase()]
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = tokenizers.normalizers.Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
        parameters = {
            "model": "WordLevel",
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
            "unk_token": unk_token,
        }
        tokenizers.implementations.BaseTokenizer.__init__(self, tokenizer, parameters)
        MixinToVocab.__init__(self)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.implementations.BaseTokenizer.save(self, path, pretty)
        return path

    @staticmethod
    def from_file(vocab_filename, **kwargs):
        """
        Load a tokenizer from a saved vocab file

        Parameters
        ----------
        vocab_filename: str
            Path to saved vocab file.
        kwargs

        Returns
        -------
        WordLevelTokenizer object
        """
        return WordLevelTokenizer(vocab_filename, **kwargs)

    def train(self, files):
        """
        Train the model using the given file or path to directory containing files

        Parameters
        ----------
        files: str or List[str]
            path or paths to files to train tokenizer on.
        """
        def hook_compressed_encoded_text(filename, mode='r', encoding='utf-8'):
            ext = os.path.splitext(filename)[1]
            if ext == '.gz':
                import gzip
                return gzip.open(filename, mode + 't', encoding=encoding)
            return open(filename, mode, encoding=encoding)

        if isinstance(files, str):
            files = [files]

        with fileinput.input(files, mode="r", openhook=hook_compressed_encoded_text) as input_file:
            for line in input_file:
                if self._tokenizer.normalizer:
                    line = self._tokenizer.normalizer.normalize_str(line)
                words = line.split()
                self._tokenizer.add_tokens(words)


class ByteBPETokenizer(MixinToVocab, tokenizers.ByteLevelBPETokenizer):
    """
    Byte-level BPE, as used by Open-AI's GPT2 model
    This class inherits from tokenizers.ByteLevelBPETokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/byte_level_bpe.py

    Parameters
    ----------
    vocab: str or Dict[str, int]]
        path to saved vocab or dict of [str -> int], if vocab is precomputed
    merges: str or Dict[Tuple[int, int] or Tuple[int, int]]]
        path to saved merges or precomputed merges.
    add_prefix_space: Boolean
    lowercase: Boolean
        Whether or not to lowercase all text as part of tokenization pipeline
    dropout: Float
        The BPE dropout to use. Must be an float between 0 and 1
    unicode_normalizer: str
        Optional, one of "nfc", "nfd", "nfkc", "nfkd", or None
    continuing_subword_prefix: str
        The prefix to attach to subword units that don't represent a beginning of word.
    end_of_word_suffix: str
       The suffix to attach to subword units that represent an end of word.
    trim_offsets: Boolean
        Whether to trim the whitespaces from the produced offsets.
    """
    def __init__(self, vocab=None, merges=None, add_prefix_space=False, lowercase=False, dropout=None,
                 unicode_normalizer=None, continuing_subword_prefix=None, end_of_word_suffix=None, trim_offsets=False):
        tokenizers.ByteLevelBPETokenizer.__init__(self, vocab, merges, add_prefix_space, lowercase, dropout,
                                                  unicode_normalizer, continuing_subword_prefix, end_of_word_suffix,
                                                  trim_offsets)
        MixinToVocab.__init__(self)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.ByteLevelBPETokenizer.save(self, path, pretty)
        return path

    @staticmethod
    def from_file(vocab_filename, merges_filename, **kwargs):
        """
        Load a tokenizer from a saved vocab file

        Parameters
        ----------
        vocab_filename: str
            Path to saved vocab file.
        merges_filename: str
            Path to saved merges file.
        **kwargs

        Returns
        -------
        ByteBPETokenizer object
        """
        vocab, merges = tokenizers.models.BPE.read_file(vocab_filename, merges_filename)
        return ByteBPETokenizer(vocab, merges, **kwargs)


class CharacterBPETokenizer(MixinToVocab, tokenizers.CharBPETokenizer):
    """
    Original character level BPE tokenizer.
    This class inherits from tokenizers.CharBPETokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/char_level_bpe.py

    Parameters
    ----------
    vocab: str or Dict[str, int]]
        path to saved vocab or dict of [str -> int], if vocab is precomputed
    merges: str or Dict[Tuple[int, int] or Tuple[int, int]]]
        path to saved merges or precomputed merges.
    unk_token: str
        token to use for unknown tokens, defaults to "<unk>"
    suffix: str
       The suffix to attach to subword units that represent an end of word.
    dropout: Float
        The BPE dropout to use. Must be an float between 0 and 1
    lowercase: Boolean
        Whether or not to lowercase all text as part of tokenization pipeline
    unicode_normalizer: str
        Optional, one of "nfc", "nfd", "nfkc", "nfkd", or None
    bert_normalizer: Boolean
        Whether to use a bert normalizer, which cleans text and handles Chinese characters. Default is True.
    split_on_whitespace_only: Boolean
        Whether to split on whitespace only. Alternative if to split on whitespace and punctuation. Default is False.
    """
    def __init__(self, vocab=None, merges=None, unk_token="<unk>", suffix="</w>", dropout=None, lowercase=False,
                 unicode_normalizer=None, bert_normalizer=True, split_on_whitespace_only=False):
        tokenizers.CharBPETokenizer.__init__(self, vocab, merges, unk_token, suffix, dropout, lowercase,
                                             unicode_normalizer, bert_normalizer, split_on_whitespace_only)
        MixinToVocab.__init__(self)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.CharBPETokenizer.save(self, path, pretty)
        return path

    @staticmethod
    def from_file(vocab_filename, merges_filename, **kwargs):
        """
        Load a tokenizer from a saved vocab file

        Parameters
        ----------
        vocab_filename: str
            Path to saved vocab file.
        merges_filename: str
            Path to saved merges file.
        **kwargs

        Returns
        -------
        CharacterBPETokenizer object
        """
        vocab, merges = tokenizers.models.BPE.read_file(vocab_filename, merges_filename)
        return CharacterBPETokenizer(vocab, merges, **kwargs)


class BERTWordPieceTokenizer(MixinToVocab, tokenizers.BertWordPieceTokenizer):
    """
    Word piece tokenizer used for BERT.
    This class inherits from tokenizers.BertWordPieceTokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/bert_wordpiece.py

    Parameters
    ----------
    vocab: str or Dict[str, int]]
        path to saved vocab or dict of [str -> int], if vocab is precomputed
    unk_token: str
        token to use for unknown tokens, defaults to "<UNK>"
    sep_token: str
        token to use for separation, defaults to "<SEP>"
    cls_token: str
        token to use for class tokens, defaults to "<CLS>"
    pad_token: str
        token to use for padding, defaults to "<PAD>"
    mask_token: str
        token to use for masking, defaults to "<MASK>"
    handle_chinese_chars: Boolean
        Whether to handle chinese chars by putting spaces around them.
    strip_accents: Boolean
        Whether to strip all accents. If this option is not specified (ie == None), then it will be determined by the
        value for `lowercase` (as in the original Bert).
    lowercase: Boolean
        Whether or not to lowercase all text as part of tokenization pipeline.
    wordpieces_prefix: str
        The prefix to use for subwords that are not a beginning-of-word.
    """
    def __init__(self, vocab=None, unk_token="[UNK]", sep_token="[SEP]", cls_token="[CLS]", pad_token="[PAD]",
                 mask_token="[MASK]", clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True,
                 wordpieces_prefix="##"):
        tokenizers.BertWordPieceTokenizer.__init__(self, vocab, unk_token, sep_token, cls_token, pad_token, mask_token,
                                                   clean_text, handle_chinese_chars, strip_accents, lowercase,
                                                   wordpieces_prefix)
        MixinToVocab.__init__(self)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.BertWordPieceTokenizer.save(self, path, pretty)
        return path

    @staticmethod
    def from_file(vocab, **kwargs):
        """
        Load a tokenizer from a saved vocab file

        Parameters
        ----------
        vocab: str
            Path to saved vocab file.
        kwargs

        Returns
        -------
        BERTWordPieceTokenizer object
        """
        vocab = tokenizers.models.WordPiece.read_file(vocab)
        return BERTWordPieceTokenizer(vocab, **kwargs)


class SentPieceBPETokenizer(MixinToVocab, tokenizers.SentencePieceBPETokenizer):
    """
    BPE tokenizer with Sentence Piece pretokenization
    This class inherits from tokenizers.SentencePieceBPETokenizer() (and as such, tokenizers.Tokenizer()). For more
    information on available methods, see:
    https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    and
    https://github.com/huggingface/tokenizers/blob/master/bindings/python/py_src/tokenizers/implementations/sentencepiece_bpe.py

    Parameters
    ----------
    vocab: str or Dict[str, int]]
        path to saved vocab or dict of [str -> int], if vocab is precomputed
    merges: str or Dict[Tuple[int, int] or Tuple[int, int]]]
        path to saved merges or precomputed merges.
    unk_token: str
        token to use for unknown tokens, defaults to "<unk>"
    replacement: str
        The replacement character for whitespace. Must be exactly one character. Default is the `▁` (U+2581) meta symbol
        (Same as in SentencePiece).
    add_prefix_space: Boolean
        Whether to add a space to the first word if there isn't already one. Default is True.
    dropout: Float
        The BPE dropout to use. Must be an float between 0 and 1
    """

    def __init__(self, vocab=None, merges=None, unk_token="<unk>", replacement="▁", add_prefix_space=True,
                 dropout=None):
        tokenizers.SentencePieceBPETokenizer.__init__(self, vocab, merges, unk_token, replacement, add_prefix_space,
                                                      dropout)
        MixinToVocab.__init__(self)

    def save(self, path, pretty=False):
        """
        Save serialized tokenizer. Alternative to `save_model(path)` that includes attribute/keywords in saved file.

        Parameters
        ----------
        path: str
            Path to saved Tokenizer file. Can be directory or filename. If directory, tokenizer will be saved in
            path/tokenizer.json. Due to how transformers loads tokenizers, filename must be "tokenizer.json" if to be
            loaded by BERT model in mangoes.modeling.bert module.
        pretty: boolean
            Whether the JSON file should be pretty formatted.

        Returns
        --------
        path: str
            path where tokenizer is saved
        """
        if os.path.isdir(path):
            path = os.path.join(path, "tokenizer.json")
        elif not os.path.split(path)[1] == "tokenizer.json":
            warnings.warn("Tokenizer must be saved as `tokenizer.json` to be loaded by BERT model", RuntimeWarning)
        tokenizers.SentencePieceBPETokenizer.save(self, path, pretty)
        return path

    @staticmethod
    def from_file(vocab_filename, merges_filename, **kwargs):
        """
        Load a tokenizer from a saved vocab file

        Parameters
        ----------
        vocab_filename: str
            Path to saved vocab file.
        merges_filename: str
            Path to saved merges file.
        **kwargs

        Returns
        -------
        SentPieceBPETokenizer object
        """
        vocab, merges = tokenizers.models.BPE.read_file(vocab_filename, merges_filename)
        return SentPieceBPETokenizer(vocab, merges, **kwargs)
