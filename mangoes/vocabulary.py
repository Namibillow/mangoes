# -*- coding: utf-8 -*-
""" Class to manage the words to be represented in embeddings or used as contexts.

"""

import logging
import os.path
import json
import collections

import mangoes.utils.decorators
import mangoes.utils.exceptions
from mangoes.constants import ENCODING

_logger = logging.getLogger(__name__)


Bigram = collections.namedtuple('Bigram', 'first second')


class Vocabulary:
    """List of words.

    Vocabulary encapsulates a mapping between words and their ids.
    A Vocabulary can be create from a collection of words.

    Parameters
    ----------
    source: list or dict
        List of words or dict where keys are words and values are their indices

    language: str (optional)

    entity: str or tuple (optional)
        if the words are annotated, attribute(s) of each word

    dependency: bool
        if context for building the cooccurrence matrix uses dependency relation as well

    See Also
    ---------
    :func:`mangoes.corpus.Corpus.create_vocabulary`
    """

    FILE_HEADER_PREFIX = "_$"

    def __init__(self, source, language=None, entity=None, dependency=False):
        self._params = {"language": language, "entity": entity, "dependency": dependency}

        self._index_word = []
        self.word_index = {}
        self._factory(source)

    def __len__(self):
        return len(self._index_word)

    def __eq__(self, other):
        return self._index_word == other._index_word

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return self._index_word.__iter__()

    def __contains__(self, word):
        return word in self.word_index

    def __getitem__(self, index):
        return self._index_word[index]

    def __repr__(self):
        return 'Vocabulary(' + str(self.words) + ')'

    def copy(self):
        v = mangoes.Vocabulary(self.words.copy())
        v._params = self._params
        return v

    @property
    def language(self):
        return self._params["language"]

    @property
    def entity(self):
        return self._params["entity"]

    @property
    def params(self):
        return self._params

    @property
    def words(self):
        """Returns the words of the vocabulary as a list"""
        return self._index_word

    def get_bigrams(self):
        return [w for w in self.words if isinstance(w, Bigram)]

    def index(self, word):
        """Returns the index associated to the word"""
        try:
            return self.word_index[word]
        except KeyError:
            raise mangoes.utils.exceptions.OutOfVocabulary(value=word)

    def indices(self, sentence):
        """Convert words of the sentence to indices

        If a word isn't in the vocabulary, its index is replaced with -1

        Parameters
        ----------
        sentence: list of str

        Returns
        -------
        list of int
        """
        return [self.index(word) if word in self else -1 for word in sentence]
        # TODO: rename encode and deal with entities fields (with a token filter)
        #     filter_word_sentence = mangoes.vocabulary.create_tokens_filter(words_vocabulary.entity)
        #     filter_bigrams_sentence = mangoes.vocabulary.create_bigrams_filter(words_vocabulary.get_bigrams())

    def append(self, word):
        """Append the word to the vocabulary"""
        if word not in self.word_index:
            self.word_index[word] = len(self.words)
            self.words.append(word)
        return self.word_index[word]

    def extend(self, other, inplace=True, return_map=False):
        """Extend the vocabulary with words in other

        Parameters
        ----------
        other: list or Vocabulary
        inplace: boolean
            If False, create a new Vocabulary
        return_map: boolean
            If True, the mapping between the indices of the words from original to merged is returned

        Returns
        -------
        Vocabulary or (Vocabulary, dict)
            Returns the merged vocabulary and, if return_map is True, the mapping between the indices of the words
            from original to merged

        """
        indices_map = {}
        if inplace:
            merged = self
        else:
            merged = self.copy()

        for word in other:
            new_index = merged.append(word)
            indices_map[other.index(word)] = new_index

        if return_map:
            return merged, indices_map

        return merged

    def save(self, path, name="vocabulary"):
        """Save the vocabulary in a file.

        Parameters
        ----------
        path: str
            Local path to the directory where vocabulary should be written

        name: str
            Name of the file to create (without extension)

        Warnings
        ---------
        If the file already exists, it will be overwritten.
        """

        file_path = os.path.join(path, name + '.txt')
        with open(file_path, "w", encoding=ENCODING) as f:
            f.write(self.FILE_HEADER_PREFIX + json.dumps({k: v for k, v in self._params.items() if v},
                                                         allow_nan=False) + "\n")
            if self.entity and not isinstance(self.entity, str):
                f.write("\n".join([self._token_to_string(t[0]) + ' ' + self._token_to_string(t[1])
                                   if isinstance(t, Bigram)
                                   else self._token_to_string(t)
                                   for t in self._index_word]))
            else:
                f.write("\n".join([str(w[0]) + ' ' + str(w[1]) if isinstance(w, Bigram) else str(w) for w in self._index_word]))
        return file_path

    def _token_to_string(self, token):
        return '/'.join([getattr(token, field) for field in self.entity])

    @classmethod
    def load(cls, path, name):
        """Load the vocabulary from its associated file.

        Parameters
        -----------
        path: str
            Local path to the directory where vocabulary file is located

        name: str
            Name of the file (without extension)

        Returns
        --------
        Vocabulary
        """
        temp_words = []
        params = {}

        def parse_words(line):
            line = line.strip()
            if ' ' in line:
                return Bigram(*line.split(' '))
            else:
                return line

        def create_token_parser(token_fields):
            Token = collections.namedtuple('Token', token_fields)

            def parse_tokens(line):
                """Parse tokens with additional information (such as annotation).
                """
                line = line.strip()
                if ' ' in line:
                    bigram = line.split(' ')
                    return Bigram(Token(*bigram[0].split('/')), Token(*bigram[1].split('/')))
                else:
                    return Token(*line.split('/'))  # simple token
            return parse_tokens

        with open(os.path.join(path, name + '.txt'), "r", encoding=ENCODING) as f:
            first_line = f.readline()
            if first_line.startswith(cls.FILE_HEADER_PREFIX):
                params = json.loads(first_line[len(cls.FILE_HEADER_PREFIX):].strip())
                if 'entity' in params and not isinstance(params['entity'], str):
                    parse_line = create_token_parser(params['entity'])
                else:
                    parse_line = parse_words
            else:
                parse_line = parse_words
                f.seek(0)

            for line in f:
                temp_words.append(parse_line(line))

        return Vocabulary(temp_words, **params)

    def _factory(self, words):
        if isinstance(words, self.__class__):
            self._params = words._params
            self._from_list(words._index_word)
        elif isinstance(words, dict):
            self._from_dict(words)
        elif isinstance(words, list):
            self._from_list(words)
        else:
            try:
                self._from_list(words.words)
            except:
                error_message = "{} can't be used as input to create a Vocabulary. " \
                                "A Vocabulary, a dict or a list is expected".format(type(words))
                raise mangoes.utils.exceptions.UnsupportedType(error_message)

    def _from_dict(self, word_index):
        words_list = sorted(word_index, key=word_index.get)
        self._from_list(words_list)

    def _from_list(self, words):
        if not self._params["dependency"]:
            self._check_bigrams(words)
        self._index_word = words # list
        self.word_index = {word: index for index, word in enumerate(self._index_word)}

    def _check_bigrams(self, words):
        """Detect and replace bigrams with a Vocabulary.Bigram instance"""
        for i, w in enumerate(words):
            if isinstance(w, str) and ' ' in w:
                # a two words string
                words[i] = Bigram(*w.split())
            elif isinstance(w, tuple) and len(w) == 2:
                # a tuple of length 2 can be :
                # - a tuple of string -> a bigram
                # - a tuple of Tokens -> a bigram
                # - a Token with 2 fields : not a bigram
                if not getattr(w, '_fields', None):
                    # not a namedtuple so (probably ?) not a Token
                    words[i] = Bigram(*w)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.entity and not isinstance(self.entity, str):
            state['_index_word'] = [tuple(t) for t in self.words]
            del state['word_index']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.entity and not isinstance(self.entity, str):
            Token = collections.namedtuple('Token', self.entity)
            self._from_list([Token(*t) for t in self._index_word])


class DynamicVocabulary(Vocabulary):
    """Extensible list of words.

    A DynamicVocabulary can be created from a collection of words or empty and each new encountered word will be
    added to it either explicitly (with :func:`.add()` or implicitly when testing if the word is in the vocabulary
    (always returns True) or getting its index

    Examples
    --------
    >>> v = mangoes.vocabulary.DynamicVocabulary()
    >>> print(v.words)
    []
    >>> v.append('a')
    0
    >>> print(v.words)
    ['a']
    >>> 'b' in v
    True
    >>> v.index('b')
    1
    >>> v.index('c')
    2


    Parameters
    ----------
    source: list or dict (optional)
        List of words or dict where keys are words and values are their indices

    language: str (optional)

    entity: str or tuple (optional)
        if the words are annotated, attribute(s) of each word

    See Also
    ---------
    :func:`mangoes.corpus.Corpus.create_vocabulary`
    """
    def __init__(self, source=None, *args, **kwargs):
        if not source:
            source = []
        super().__init__(source, *args, **kwargs)

    def index(self, word):
        """Returns the index associated to the word, adding it to the vocabulary if not yet"""
        try:
            return self.word_index[word]
        except KeyError:
            return self.append(word)

    def __contains__(self, word):
        self.append(word)
        return True


def merge(*vocabularies, keys=None, concat=lambda key, word: key + '_' + word, return_map=False):
    """Merge a list of Vocabulary into one

    Examples
    --------
    >>> import mangoes.vocabulary
    >>> v1 = mangoes.Vocabulary(['a', 'b', 'c'], language='l1')
    >>> v2 = mangoes.Vocabulary(['a', 'd'], language='l2')
    >>> merge(v1, v2)
    Vocabulary(['a', 'b', 'c', 'd'])
    >>> merge(v1, v2, keys=True)
    Vocabulary(['l1_a', 'l1_b', 'l1_c', 'l2_a', 'l2_d'])
    >>> merge(v1, v2, keys=['v1', 'v2'])
    Vocabulary(['v1_a', 'v1_b', 'v1_c', 'v2_a', 'v2_d'])

    With tokens :
    >>> import collections
    >>> Token = collections.namedtuple('Token', 'lemma POS')
    >>> v3 = mangoes.Vocabulary([Token('a', 'X'), Token('b', 'Y'), Token('c', 'X')], language='l1')
    >>> v4 = mangoes.Vocabulary([Token('a', 'X'), Token('d', 'Y')], language='l2')
    >>> LangToken = collections.namedtuple('LangToken', 'lemma POS lang')
    >>> merge(v3, v4, keys=True, concat=lambda lang, token: LangToken(*token, lang)) #doctest: +ELLIPSIS
    Vocabulary([LangToken(lemma='a', POS='X', lang='l1'), ..., LangToken(lemma='d', POS='Y', lang='l2')])

    Parameters
    ----------
    vocabularies: list of vocabularies
    keys: None (default), or bool or list of str
        If None or False, words that are common to several vocabularies are considered the same and will appear
        only once in resulting Vocabulary
        If keys is a list of string, of same size as counts, all words are prefixes with these keys.
        If keys is True, the languages of the vocabularies are used as keys
    concat: callable, optional
        Function that takes a key and a word (or a token) as input and returns a new word (or token)
        If keys are given, this function is called to create the word of the merged vocabulary from the given keys and
        the original words
        Default is '{key}_{word}' that prefixes each word with their key and is only valid from simple string words
        vocabularies.
        Bigrams are transformed applying these function to both of their part


    Returns
    -------
    Vocabulary

    """
    indices_map = []

    def _concat(k, w):
        if isinstance(w, mangoes.vocabulary.Bigram):
            return mangoes.vocabulary.Bigram(concat(k, w[0]), concat(k, w[1]))
        return concat(k, w)

    if not keys:
        merged = Vocabulary([])
        for v in vocabularies:
            _, m = merged.extend(v, inplace=True, return_map=True)
            if return_map:
                indices_map.append(m)
        if return_map:
            return merged, indices_map
        return merged
    else:
        if keys == True:
            keys = [v.language for v in vocabularies]
        merged = mangoes.Vocabulary([_concat(k, w) for v, k in zip(vocabularies, keys) for w in v])
        merged._params = {k: v._params for k, v in zip(keys, vocabularies)}
        if return_map:
            for v, k in zip(vocabularies, keys):
                indices_map.append({i: merged.index(_concat(k, w))
                                    for w, i in v.word_index.items()})
            return merged, indices_map
    return merged


def create_token_filter(fields):
    """Returns a function to filter the given fields from a token

    Examples
    --------
    >>> Token = mangoes.corpus.BROWN.Token
    >>> cat_token = Token(form="cat", lemma="cat", POS="NOUN")
    >>> mangoes.vocabulary.create_token_filter("lemma")(cat_token)
    'cat'
    >>> mangoes.vocabulary.create_token_filter(("lemma", "POS"))(cat_token)
    Token(lemma='cat', POS='NOUN')

    Parameters
    ----------
    fields: str or tuple
        name of the fields(s) to keep

    Returns
    -------
    callable


    """
    if fields:
        if isinstance(fields, str):
            def filter_token(token):
                if isinstance(token, Bigram):
                    return Bigram(getattr(token[0], fields), getattr(token[1], fields))
                else:
                    return getattr(token, fields)
        else:
            Token = collections.namedtuple('Token', fields)

            def filter_token(token):
                if isinstance(token, Bigram):
                    return Bigram(Token(*[getattr(token[0], attr) for attr in fields]),
                                  Token(*[getattr(token[1], attr) for attr in fields]))
                else:
                    return Token(*[getattr(token, attr) for attr in fields])
    else:
        def filter_token(token):
            return token
    return filter_token


def create_tokens_filter(fields):
    """Returns a function to filter the given fields from a list of tokens"""
    if fields:
        filter_token = create_token_filter(fields)

        def filter_tokens(sentence):
            return [filter_token(token) for token in sentence]
    else:
        def filter_tokens(sentence):
            return sentence

    return filter_tokens


def create_bigrams_filter(bigrams=None):
    """Returns a function to find expected bigrams within a sentence"""
    if bigrams:
        if getattr(bigrams[0][0], '_fields', None):
            # bigrams of Tokens
            fields = bigrams[0][0]._fields
            Token = collections.namedtuple('Token', fields)

            def filter_bigrams(sentence):
                filtered_sentence = []
                i = 0
                while i < len(sentence):
                    try:
                        x = Token(*[getattr(sentence[i], f) for f in fields]), Token(*[getattr(sentence[i + 1], f) for f in fields])
                        if x in bigrams:
                            filtered_sentence.append(Bigram(sentence[i], sentence[i + 1]))
                            i += 1
                        else:
                            filtered_sentence.append(sentence[i])
                    except IndexError:
                        # sentence[i] or sentence[i + 1] is None
                        filtered_sentence.append(sentence[i])
                    i += 1
                return filtered_sentence
        else:
            # bigrams of strings
            def filter_bigrams(sentence):
                filtered_sentence = []
                i = 0
                while i < len(sentence):
                    try:
                        if (sentence[i], sentence[i + 1]) in bigrams:
                            filtered_sentence.append(Bigram(sentence[i], sentence[i + 1]))
                            i += 1
                        else:
                            filtered_sentence.append(sentence[i])
                    except (TypeError, IndexError):
                        # sentence[i] or sentence[i + 1] is None
                        filtered_sentence.append(sentence[i])
                    i += 1
                return filtered_sentence
    else:
        def filter_bigrams(sentence):
            return sentence

    return filter_bigrams
