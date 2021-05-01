# -*- coding: utf-8 -*-
"""Functions to count the words co-occurrence within a corpus.

This module provides the main function count_cooccurrence to construct a CountBasedRepresentation.
"""
from  collections import defaultdict, namedtuple
import logging
import math
import multiprocessing
import random

import numpy as np

import mangoes.context
import mangoes.utils.decorators
import mangoes.utils.exceptions
import mangoes.utils.multiproc
import mangoes.utils.counting
from mangoes.utils.options import ProgressBar

logger = logging.getLogger(__name__)


##########
# Computing / Building a cooccurrence count instance
##########
@mangoes.utils.decorators.timer(display=logger.info)
def count_cooccurrence(corpus, words,
                       context=mangoes.context.Window(),
                       subsampling=False, nb_workers=None, batch=1000000):
    """Build a CountBasedRepresentation where rows are the words in `words`, counting co-occurrences from the `corpus`.

    Examples
    --------
    >>> import mangoes.counting
    >>> window_5 = mangoes.context.Window(window_half_size=5)
    >>> counts_matrix = mangoes.counting.count_cooccurrence(corpus, vocabulary, context=window_5)

    Parameters
    -----------
    corpus: mangoes.Corpus
    words: mangoes.Vocabulary
        words represented as vectors (rows of the matrix)
    context: mangoes.context.Context or mangoes.Vocabulary
        A Vocabulary or context defining function such as defined in the :mod:`mangoes.context` module.
        Default is a window of size 1-x-1 : count the co-occurrences between the words in `words_vocabulary` and the
        words surrounding it.
        If `context` is a Vocabulary, only consider the words of this vocabulary in the window.
    nb_workers: int
        number of subprocess to use;
    subsampling: boolean or dict
        to apply subsampling on frequent words. Value can be False (default), True or a frequency
        threshold. If True, the default value of `create_subsampler()` function is used

    Returns
    -------
    mangoes.CountBasedRepresentation

    """
    if words is None:
        raise mangoes.utils.exceptions.RequiredValue("'words' is required to count cooccurrences")

    if isinstance(context, mangoes.Vocabulary):
        context = mangoes.context.Window(context)

    if not nb_workers:
        nb_workers = multiprocessing.cpu_count() - 1

    kwargs = {}
    if subsampling:
        kwargs['subsampler'] = create_subsampler(corpus) if subsampling is True else create_subsampler(corpus,
                                                                                                       subsampling)
    if corpus.nb_sentences:
        kwargs['nb_sentences'] = _estimate_nb_sentences_per_worker(corpus.nb_sentences, nb_workers)

    if nb_workers == 1:
        result = _count_context_cooccurrence(corpus.reader.sentences(),
                                             context, words, context.vocabulary, **kwargs).__next__()
        matrix, contexts_words = result
    else:
        data_parallel = mangoes.utils.multiproc.DataParallel(_count_context_cooccurrence,
                                                             mangoes.utils.counting._reduce_counter,
                                                             nb_workers=nb_workers, batch=batch)


        matrix, contexts_words = data_parallel.run(corpus,
                                                   *(context, words, context.vocabulary),
                                                   **kwargs)

    features = mangoes.Vocabulary(contexts_words)

    hyperparameters = {"corpus": corpus.params,
                       "context_definition": context.params,
                       "subsampling": subsampling}

    return mangoes.CountBasedRepresentation(words, features, matrix, hyperparameters)


def _count_context_cooccurrence(sentences, context, words_vocabulary, contexts_vocabulary,
                                nb_sentences=None, subsampler=None, batch=1000000):
    """Parallelizable function to count cooccurrence

    Count cooccurrence between words and contexts as defined by the 'context_definition' function, and on the
    sentences given by the 'sentences_enumerator'.

    Parameters
    -----------
    sentences: mangoes.utils.reader.SentenceGenerator
        an iterator over sentences
    context: callable
        a context defining function such as defined in 'mangoes.context'
    words_vocabulary: mangoes.Vocabulary
        words to represent as vectors (rows of the matrix)
    contexts_vocabulary: mangoes.Vocabulary
        words to use as features (columns of the matrix)
    nb_sentences: int, optional
        number of sentences. Only used for the progress bar
    subsampler: dict, optional
        a dictionary with probabilities of removal of frequent words

    Yields
    --------
    (collections.Counter, mangoes.Vocabulary)
    """
    counter = defaultdict(float)

    build_contexts_vocabulary = contexts_vocabulary is None
    
    dependency = isinstance(context, mangoes.context.DependencyBasedContext)
    
    if build_contexts_vocabulary:
        contexts_vocabulary = mangoes.vocabulary.DynamicVocabulary(dependency=dependency)

    same_vocabulary = (words_vocabulary == contexts_vocabulary) and not dependency

    filter_word_sentence = mangoes.vocabulary.create_tokens_filter(words_vocabulary.entity)
    filter_bigrams_sentence = mangoes.vocabulary.create_bigrams_filter(words_vocabulary.get_bigrams())

    context_accept_encoded_sentence = hasattr(context, 'call_on_encoded')
    if context_accept_encoded_sentence and not build_contexts_vocabulary:
        get_contexts = context.call_on_encoded
    else:
        get_contexts = context


    with ProgressBar(total=nb_sentences) as progress_bar:
        while True:
            try:
                for _ in range(batch):
                    sentence = sentences.__next__()
                    sentence = filter_bigrams_sentence(sentence)
                    if subsampler:
                        sentence = _subsample(sentence, subsampler)

                    word_sentence = filter_word_sentence(sentence)
                    word_sentence = words_vocabulary.indices(word_sentence)
                    # (NOTE) -1 if words in sentence is not in vocabulary (because of filtering by max_frequency, stopwords etc...)
                    word_sentence_mask = [True if w > -1 else False for w in word_sentence]

                    if build_contexts_vocabulary:
                        contexts_sentence = context.filter_sentence(sentence)

                        for word, word_contexts in zip(word_sentence, get_contexts(contexts_sentence)):
                            if word > -1:
                                for context_word in word_contexts:
                                    if dependency: # put weight 
                                        context_word_split = context_word.rsplit('/', 1)
                                        try:
                                            new_tok = convert(eval(context_word_split[0]))
                                        except:
                                            new_tok = context_word_split[0]
                                        counter[(word, contexts_vocabulary.index(new_tok))]+=float(context_word_split[-1])
                                    else:
                                        counter[(word, contexts_vocabulary.index(context_word))]+=1

                    else:
                        if same_vocabulary:
                            for word, word_contexts in zip(word_sentence, get_contexts(word_sentence, word_sentence_mask)):
                                if word > -1:
                                    for context_word in word_contexts:
                                        if dependency: # weight is put 
                                            context_word_split = context_word.rsplit('/', 1)
                                            counter[(word, context_word_split[0])]+= float(context_word_split[-1])
                                        else:
                                            counter[(word, context_word)]+=1

                        elif context_accept_encoded_sentence:
                            contexts_sentence = contexts_vocabulary.indices(context.filter_sentence(sentence))

                            for word, word_contexts in zip(word_sentence, get_contexts(contexts_sentence)):
                                if word > -1:
                                    for context_word in word_contexts:
                                        if dependency: # weight is put 
                                            context_word_split = context_word.rsplit('/', 1)
                                            counter[(word, context_word_split[0])]+=float(context_word_split[-1])
                                        else:
                                            counter[(word, context_word)]+=1

                        else:
                            contexts_sentence = sentence

                            for word_id, word_contexts in zip(word_sentence, get_contexts(contexts_sentence)):
                                if word_id > -1:
                                    for context_word in word_contexts:
                                        if dependency: # put weight 
                                            context_word_split = context_word.rsplit('/', 1)
                                            try:
                                                new_tok = convert(eval(context_word_split[0]))
                                            except: 
                                                new_tok = context_word_split[0]
                                            counter[(word_id, contexts_vocabulary.index(new_tok))]+=float(context_word_split[-1])
                                        else:
                                            counter[(word_id, contexts_vocabulary.index(context_word))]+=1

                    progress_bar.update()

                csr = mangoes.utils.counting._counter_to_csr(counter,
                                                             shape=(len(words_vocabulary), len(contexts_vocabulary)))
                yield (csr, contexts_vocabulary)

                counter = defaultdict(int)
                if build_contexts_vocabulary:
                    contexts_vocabulary = mangoes.vocabulary.DynamicVocabulary(dependency=dependency)
            except StopIteration:
                break

    csr = mangoes.utils.counting._counter_to_csr(counter,
                                                 shape=(len(words_vocabulary), len(contexts_vocabulary)))
    yield (csr, contexts_vocabulary)

def convert(dictionary):
    return namedtuple('Token', dictionary.keys())(**dictionary)

def _subsample(sentence, subsampler):
    return [word if word not in subsampler or random.random() > subsampler[word] else None for word in sentence]


def create_subsampler(corpus, threshold=10 ** -5):
    """Compute probabilities of removal of frequent words

    For each word appearing with a frequency higher than the threshold in the corpus, a probabilty of removal
    is computed following the formula :

    .. math::
            p = 1 - \sqrt{\\frac{t}{f}}

    where :math:`t` is the threshold and :math:`f` the frequency of the word in the corpus.

    Parameters
    ----------
    corpus: mangoes.Corpus
        Frequencies come from corpus.words_count
    threshold: float, optional
        Words appearing more than this threshold appear in the subsampler (default : :math:`10^{-5}`)

    Returns
    --------
    dict
        a dictionary associating each frequent word with a removal probability
    """
    threshold *= corpus.size
    return {word: 1 - math.sqrt(threshold / count) for word, count in corpus.words_count.items() if count > threshold}


def _estimate_nb_sentences_per_worker(nb_sentences, nb_workers):
    if nb_workers > 1:
        return nb_sentences // (nb_workers - 1)
    return nb_sentences


def merge(*counts, word_keys=None, context_keys=None, concat=lambda key, word: key + '_' + word):
    """Merge cooccurrence counts into one, providing parameters to handle how words and contexts should be merged

    Examples
    --------
    First example is a use case where your counts cooccurrences of words and POS tags from different languages
    >>> import mangoes
    >>> english_words = mangoes.Vocabulary(['can', 'car', 'cap'], language='en')
    >>> french_words = mangoes.Vocabulary(['car', 'cap'], language='fr')
    >>> pos_contexts = mangoes.Vocabulary(['ADJ', 'NOUN', 'VERB'])
    >>> en_count = mangoes.CountBasedRepresentation(english_words, pos_contexts, np.array(range(9)).reshape((3,3)))
    >>> print(en_count.to_df()) # doctest: +NORMALIZE_WHITESPACE
             ADJ  NOUN  VERB
        can    0     1     2
        car    3     4     5
        cap    6     7     8
    >>> fr_count = mangoes.CountBasedRepresentation(french_words, pos_contexts, np.array(range(6)).reshape((2,3)))
    >>> print(fr_count.to_df()) # doctest: +NORMALIZE_WHITESPACE
             ADJ  NOUN  VERB
        car    0     1     2
        cap    3     4     5
    >>> print(mangoes.counting.merge(en_count, fr_count, word_keys=True).to_df()) # doctest: +NORMALIZE_WHITESPACE
               ADJ  NOUN  VERB
        en_can   0     1     2
        en_car   3     4     5
        en_cap   6     7     8
        fr_car   0     1     2
        fr_cap   3     4     5

    A second example where contexts are keyed:
    >>> import mangoes
    >>> words = mangoes.Vocabulary(['a', 'b', 'c'])
    >>> contexts1 = mangoes.Vocabulary(['x', 'y', 'z'])
    >>> contexts2 = mangoes.Vocabulary(['x', 'y'])
    >>> count1 = mangoes.CountBasedRepresentation(words, contexts1, np.array(range(9)).reshape((3,3)))
    >>> print(count1.to_df()) # doctest: +NORMALIZE_WHITESPACE
             x     y     z
        a    0     1     2
        b    3     4     5
        c    6     7     8
    >>> count2 = mangoes.CountBasedRepresentation(words, contexts2, np.array(range(6)).reshape((3, 2)))
    >>> print(count2.to_df()) # doctest: +NORMALIZE_WHITESPACE
             x     y
        a    0     1
        b    2     3
        c    4     5
    >>> print(mangoes.counting.merge(count1, count2, context_keys=['c1', 'c2']).to_df()) # doctest: +NORMALIZE_WHITESPACE
            c1_x    c1_y    c1_z    c2_x    c2_y
        a      0       1       2       0       1
        b      3       4       5       2       3
        c      6       7       8       4       5
    >>> print(mangoes.counting.merge(count1, count2, context_keys=['c1', 'c2'], \
                                     concat=lambda k,w : '{}({})'.format(w, k)).to_df()) # doctest: +NORMALIZE_WHITESPACE
             x(c1)   y(c1)   z(c1)   x(c2)   y(c2)
        a      0       1       2       0       1
        b      3       4       5       2       3
        c      6       7       8       4       5

    And a third example with tokens
    >>> import collections
    >>> Token = collections.namedtuple('Token', 'lemma POS')
    >>> english_tokens = mangoes.Vocabulary([Token('can', 'N'), Token('can', 'V'), \
                                             Token('car', 'N'), Token('cap', 'N')], language='en')
    >>> french_tokens = mangoes.Vocabulary([Token('car', 'C'), Token('cap', 'N')], language='fr')
    >>> en_tok_count = mangoes.CountBasedRepresentation(english_tokens, pos_contexts, \
                                                        np.array(range(12)).reshape((4,3)))
    >>> print(en_tok_count.to_df()) # doctest: +NORMALIZE_WHITESPACE
                ADJ  NOUN  VERB
        can N    0     1     2
            V    3     4     5
        car N    6     7     8
        cap N    9    10    11
    >>> fr_tok_count = mangoes.CountBasedRepresentation(french_tokens, pos_contexts, np.array(range(6)).reshape((2,3)))
    >>> print(fr_tok_count.to_df()) # doctest: +NORMALIZE_WHITESPACE
                      ADJ  NOUN  VERB
        car C    0     1     2
        cap N    3     4     5
    >>> print(mangoes.counting.merge(en_tok_count, fr_tok_count, word_keys=True, \
                                     concat=lambda k, w: (*w, k)).to_df()) # doctest: +NORMALIZE_WHITESPACE
                   ADJ  NOUN  VERB
        can N en    0     1     2
            V en    3     4     5
        car N en    6     7     8
        cap N en    9    10    11
        car C fr    0     1     2
        cap N fr    3     4     5

    Parameters
    ----------
    counts: list of mangoes.CountBasedRepresentation
        List of cooccurrence counts to be merged
    word_keys: None (default), or bool or list of str
        If None or False, words that are common to several vocabularies are considered the same and their counts are
        summed.
        If word_keys is a list of string, of same size as counts, words are prefixed with these keys (prefixing is
        default but you can change that with format_str parameter).
        If word_keys is True, the languages of the vocabularies are used as keys
    context_keys: None (default), or bool or list of str
        If None or False, context words that are common to several context vocabularies are considered the same and
        their counts are summed.
        If context_keys is a list of string, of same size as counts, context words are prefixes with these keys.
        If context_keys is True, the languages of the context vocabularies are used as keys
    concat: callable, optional
        Function that takes a key and a word (or a token) as input and returns a new word (or token)
        If keys are given, this function is called to create the word of the merged vocabulary from the given keys and
        the original words
        Default is '{key}_{word}' that prefixes each word with their key and is only valid from simple string words
        vocabularies.
        Bigrams are transformed applying this function to both of their part

    Returns
    --------
    mangoes.CountBasedRepresentation
        a cooccurrence count with merged vocabulary, context and counts
    """
    mrg_vocabulary, vocabulary_mappings = mangoes.vocabulary.merge(*[c.words for c in counts], keys=word_keys,
                                                                   return_map=True, concat=concat)

    mrg_context, context_mappings = mangoes.vocabulary.merge(*[c.contexts_words for c in counts], keys=context_keys,
                                                             return_map=True, concat=concat)

    mrg_matrix = counts[0].matrix.copy()
    for count, row_mapping, context_mapping in zip(counts[1:], vocabulary_mappings[1:], context_mappings[1:]):
        mrg_matrix = mrg_matrix.combine(count.matrix, new_shape=(len(mrg_vocabulary), len(mrg_context)),
                                        row_indices_map=row_mapping, col_indices_map=context_mapping)

    return mangoes.CountBasedRepresentation(mrg_vocabulary, mrg_context, mrg_matrix)

