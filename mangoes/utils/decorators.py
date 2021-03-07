# -*- coding: utf-8 -*-
import functools
import inspect
import time

import mangoes.utils.persist


def timer(display=print, label=None):
    """Decorator to measure the execution time of a function

    Parameters
    ----------
    display: callable
        function to use to display the execution time. Defaut = print but a logger should be used
    label
        label to give to the function in the display. If None (default), the name of the function will be used.

    """
    def decorated(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if label:
                label_to_display = label
            else:
                label_to_display = func.__name__

            display("Starting {} ...".format(label_to_display))
            start = time.time()
            response = func(*args, **kwargs)
            display("Finished {} in {}s".format(label_to_display, time.time() - start))
            return response
        return wrapper
    return decorated


def counter_filter(func):
    """Decorator to transform a function on a collections.Counter into a filter callable on this collections.Counter

    Filters are meant to be applied in the :func:`mangoes.corpus.Corpus.create_vocabulary` method to create a Vocabulary
    from a Corpus.
    The Counter object has to be the last parameter of the decorated function.
    So, this function, called without `words_count` parameter, provides a parametrized
    filter to be called on a collections.Counter.


    Examples
    ---------

    These 2 lines are equivalent :

        >>> mangoes.vocabulary.remove_most_frequent(min_frequency, words_count)
        >>> mangoes.vocabulary.remove_most_frequent(min_frequency)(words_count)


    This parametrized function is meant to be used to create a Vocabulary from a Corpus :

        >>> vocabulary = mangoes.Vocabulary(corpus,
        >>>                                filters=[mangoes.vocabulary.remove_most_frequent(max_frequency)])


    See Also
    --------
    :func:`mangoes.corpus.Corpus.create_vocabulary`
    """
    @functools.wraps(func)
    def _inner_filter(*args, **kwargs):
        try:
            if len(args) < len(inspect.signature(func).parameters):
                return functools.partial(func, *args, **kwargs)
            return func(*args, **kwargs)
        except mangoes.utils.exceptions.NotAllowedValue as e:
            raise e
    return _inner_filter
