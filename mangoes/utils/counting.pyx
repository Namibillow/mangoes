import collections
import random
import numpy as np
from scipy import sparse


from mangoes.utils.options import ProgressBar

Token = collections.namedtuple('Token', ("form", "lemma", "POS"), module='mangoes.utils.counting')

cdef list ngrams(list sentence, int n):
    cdef int i, j
    cdef int sentence_len = len(sentence)
    cdef num_ngrams = sentence_len - n + 1
    cdef list ngrams_results = [_ for _ in range(num_ngrams)]

    for i in range(num_ngrams):
        ngrams_results[i] = tuple([sentence[j] for j in range(i, i + n)])
    return ngrams_results

cpdef (int, int) _get_random_size(int max_size_before, int max_size_after):
    cdef int size_before, size_after
    # TODO: replace python randint with cython equivalent for speed up
    size_before = random.randint(1, max_size_before) if max_size_before > 0 else 0
    if max_size_after == max_size_before:
        size_after = size_before
    else:
        size_after = random.randint(1, max_size_after) if max_size_after > 0 else 0
    return size_before, size_after

def _counter_to_csr(counter, (int, int) shape):
    """Build a sparse.csr_matrix from a collection.Counter built with count_cooccurrence.

    Parameters
    -----------
    counter: Counter
        Counter of ((i,j), count) key-values pairs
    shape: tuple
        shape of the resulting scr matrix

    Returns
    --------
    sparse.csr_matrix
    """
    cdef Py_ssize_t counter_len = len(counter)
    cdef int i

    data = np.empty(shape=(counter_len, 3), dtype=np.intc)
    cdef int[:, ::1] data_view = data
    for i, ((word_index, context_index), count) in enumerate(counter.items()):
        data_view[i, 0] = word_index
        data_view[i, 1] = context_index
        data_view[i, 2] = count
    return sparse.csr_matrix((data_view[:, 2], (data_view[:, 0], data_view[:, 1])), shape=shape)


def _reduce_counter(tuple total, tuple part):
    """ Reduces multiprocessing counters

    Parameters
    -----------
    total: tuple
        tuple containing current total counter in scipy scr sparse matrix, and total vocabulary
    part: tuple
        tuple containing new counter in scipy scr sparse matrix, and new vocabulary.
        Will be added to total

    Returns
    --------
    tuple of form (sparse.csr_matrix, Vocabulary) containing combined counts and vocabulary
    """
    total_counter, total_vocabulary = total
    part_counter, part_vocabulary = part

    if total_vocabulary == part_vocabulary:
        return total_counter + part_counter, total_vocabulary

    cdef dict part_to_total_indices_map = {}
    cdef int i

    for i in range(len(part_vocabulary)):
        part_to_total_indices_map[i] = total_vocabulary.index(part_vocabulary[i])

    # during the mapping, words of part_vocabulary are added to total_vocabulary
    new_shape = (total_counter.shape[0], len(total_vocabulary))

    # update the indices in part_counter to map them to total_counter
    cdef int num_new_indices = part_counter.indices.shape[0]
    new_indices = np.empty(shape=(num_new_indices,), dtype=np.intc)
    cdef int[::1] new_indices_view = new_indices
    cdef int[:] part_indices_view = part_counter.indices

    for i in range(num_new_indices):
        new_indices_view[i] = part_to_total_indices_map[part_indices_view[i]]

    new_part = sparse.csr_matrix((part_counter.data, new_indices, part_counter.indptr), shape=new_shape)
    total_counter.resize(new_shape)

    return total_counter + new_part, total_vocabulary


def count_words_raw(sentences, nb_sentences=None):
    cdef int real_nb_sentences = 0
    cdef list sentence
    cdef dict words_count = {}

    for sentence in ProgressBar(sentences, total=nb_sentences, desc="Counting words"):
        for word in sentence:
            try:
                words_count[word] += 1
            except KeyError:
                words_count[word] = 1
        real_nb_sentences += 1

    return collections.Counter(words_count), real_nb_sentences


def count_words_annotated(sentences, nb_sentences=None):
    cdef int real_nb_sentences = 0
    cdef list sentence
    cdef dict words_count = {}
    cdef tuple token

    for sentence in ProgressBar(sentences, total=nb_sentences, desc="Counting words"):
        for tok in sentence:
            token = tuple((tok.form, tok.lemma, tok.POS))
            try:
                words_count[token] += 1
            except KeyError:
                words_count[token] = 1
        real_nb_sentences += 1

    return collections.Counter({Token(k[0], k[1], k[2]):v for k,v in words_count.items()}), real_nb_sentences


def count_bigrams_raw(sentences, nb_sentences=None):
    cdef int real_nb_sentences = 0
    cdef list sentence
    cdef dict bigrams_count = {}
    cdef int i, n
    cdef tuple bigram

    for sentence in ProgressBar(sentences, total=nb_sentences, desc="Counting words"):
        n = len(sentence)
        if n > 1:
            for i in range(n-1):
                bigram = tuple((sentence[i], sentence[i+1]))
                try:
                    bigrams_count[bigram] += 1
                except KeyError:
                    bigrams_count[bigram] = 1
        real_nb_sentences += 1

    return collections.Counter(bigrams_count), real_nb_sentences


def count_bigrams_annotated(sentences, nb_sentences=None):
    cdef int real_nb_sentences = 0
    cdef list sentence
    cdef dict bigrams_count = {}
    cdef tuple token, bigram
    cdef int i, n

    for sentence in ProgressBar(sentences, total=nb_sentences, desc="Counting words"):
        n = len(sentence)
        if n > 1:
            for i in range(n-1):
                bigram = tuple((tuple((sentence[i].form, sentence[i].lemma, sentence[i].POS)),
                                tuple((sentence[i+1].form, sentence[i+1].lemma, sentence[i+1].POS))))
                try:
                    bigrams_count[bigram] += 1
                except KeyError:
                    bigrams_count[bigram] = 1
        real_nb_sentences += 1

    return collections.Counter({tuple((Token(k[0][0], k[0][1], k[0][2]), Token(k[1][0], k[1][1], k[1][2]))):v for k,v
                                in bigrams_count.items()}), real_nb_sentences


def get_window_function(int size_before, int size_after, bint dirty, bint dynamic, int n, bint distance):
    # if distance is True, the items returned by _window are tuples : (token, distance from target)
    format_output = (lambda token, d: (token, d)) if distance else (lambda token, _: token)

    # if dynamic, the size of the window is sampled between 1 and the size passed as parameter
    get_size = (lambda: _get_random_size(size_before, size_after)) if dynamic else (lambda: (size_before, size_after))

    # if n > 1, _window returns n-grams i.e. tuples of n tokens
    get_items_from = (lambda sentence: ngrams(sentence, n)) if n > 1 else (lambda sentence: sentence)

    # if n > 1, all tokens in n-grams have to be checked if in vocabulary
    ignored = {None, -1}
    check_in_vocabulary = (lambda ng: not any((t in ignored for t in ng))) if n > 1 else (lambda t: t not in ignored)

    if not dirty:
        if not dynamic and size_before == size_after and n == 1 and not distance:
            def _window(list sentence, list mask):
                cdef int sentence_len = len(sentence)
                cdef list result = [[] for _ in range(sentence_len)]
                cdef int i, j

                for i in range(sentence_len):
                    if mask[i]:
                        result[i].extend([sentence[j] for j in range(max(i - size_before, 0),
                                                                     min(i + size_after + 1, sentence_len))
                                          if not i == j and sentence[j] not in ignored])
                    else:
                        result[i] = []
                return result
        else:
            def _window(list sentence, list mask):
                cdef int sentence_len = len(sentence)
                cdef list result = [[] for _ in range(sentence_len)]
                cdef list before_list, after_list
                cdef int i, j, start_index, start_distance, end_index, before, after

                for i in range(sentence_len):
                    if mask[i]:
                        before, after = get_size()
                        if i - before < 0:
                            start_index = 0
                            start_distance = i - before + n - 1
                        else:
                            start_index = i - before
                            start_distance = - before + n - 1

                        # distance_start is the distance from the first token in before list to the target token (i)
                        before_list = [format_output(w, d)
                                       for d, w in enumerate(get_items_from(sentence[start_index:i]),
                                                             start=start_distance)
                                       if check_in_vocabulary(w)]

                        end_index = min(i + after + 1, len(sentence))
                        after_list = [format_output(w, d)
                                      for d, w in enumerate(get_items_from(sentence[i + 1:end_index]),
                                                            start=1)
                                      if check_in_vocabulary(w)]

                        result[i] = before_list + after_list
                    else:
                        result[i] = []
                return result
    else:
        def _window(list sentence, list mask):
            cdef int sentence_len = len(sentence)
            cdef list result = [[] for _ in range(sentence_len)]
            cdef list before_list, after_list
            cdef int i, j, d, before, after

            for i in range(len(sentence)):
                if mask[i]:
                    before, after = get_size()

                    before_list = []
                    d = -1
                    while i + d >= 0 and len(before_list) < before:
                        items = get_items_from(sentence[i + d + 1 - n:i + d + 1])
                        if len(items) > 0 and check_in_vocabulary(items[0]):
                            # items cannot have more than one element
                            before_list.append(format_output(items[0], d))
                        d -= 1
                    before_list = list(reversed(before_list))

                    after_list = []
                    d = 1
                    while i + d < len(sentence) and len(after_list) < after:
                        items = get_items_from(sentence[i + d:i + d + n])
                        if len(items) > 0 and check_in_vocabulary(items[0]):
                            # items cannot have more than one element
                            after_list.append(format_output(items[0], d))
                        d += 1

                    result[i] = before_list + after_list
                else:
                    result[i] = []
            return result

    return _window

