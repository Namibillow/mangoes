# -*- coding: utf-8 -*-
"""Classes and functions to evaluate embeddings according to the "Outlier Detection" task.


This module implements the evaluation task defined in [1]_

Datasets available in this module :

* `OD_8_8_8` [1]_
* `WIKI_SEM_500` [2]_

References
----------
.. [1] José Camacho-Collados and Roberto Navigli. Find the word that does not belong: A Framework for an Intrinsic
        Evaluation of Word Vector Representations. In Proceedings of the ACL Workshop on Evaluating Vector Space
        Representations for NLP, Berlin, Germany, August 12, 2016.
.. [2]


"""
from collections import namedtuple

import numpy as np

import mangoes
import mangoes.utils.arrays
import mangoes.utils.metrics
from mangoes.evaluation.base import PrintableReport, BaseEvaluation, BaseEvaluator


# ####################################################################################################################
# DATASETS

class Dataset(mangoes.evaluation.base.BaseDataset):
    """Class to create a Dataset for outlier detection task, to be used in Evaluation class

    The outlier is the last word of the group

    Examples
    --------
    >>> from mangoes.evaluation.outlier import Dataset
    >>> user_dataset = Dataset("user dataset", ['january february march saturn', 'monday tuesday friday phone'])
    >>> cats_dataset = Dataset("cats", "../resources/en/outlier_detection/8-8-8/Big_cats.txt")

    2 analogy datasets are available in this module:

    - the 8-8-8 dataset :
    >>> import mangoes.evaluation.outlier
    >>> _8_8_8 = mangoes.evaluation.outlier._8_8_8

    - the Wiki Sem 500 dataset :
    >>> import mangoes.evaluation.outlier
    >>> msr = mangoes.evaluation.outlier.WIKI_SEM_500
    """

    @classmethod
    def parse_question(cls, question):
        """
        Examples
        --------
        >>> Dataset.parse_question('january february march saturn')
        'january february march saturn'

        Parameters
        ----------
        question: str
            A splittable string with the group of words, outlier in last position

        Returns
        -------
        namedtuple

        """
        return ' '.join(question.strip().split())

    @classmethod
    def parse_file(cls, file_content):
        # a standard outlier detection dataset files contains first a cluster of words, one per line, then a blank line,
        # then a list of outliers for the cluster
        cluster = []
        outliers = []

        current = cluster
        for line in file_content:
            if not line.strip():
                current = outliers
                continue
            current.append(line.strip())

        return [' '.join(cluster) + ' ' + outlier for outlier in outliers]


_8_8_8 = Dataset("8-8-8", "../resources/en/outlier_detection/8-8-8")
WIKI_SEM_500 = Dataset("wiki-sem-500", "../resources/en/outlier_detection/wiki-sem-500-tokenized.zip")

ALL_DATASETS = [_8_8_8, WIKI_SEM_500]


# ####################################################################################################################
# EVALUATOR

class Evaluator(BaseEvaluator):
    """Evaluator to detect outliers in a group of words according to the given representation

    Parameters
    ----------
    representation: mangoes.Representation
        The Representation to use
    """

    def __init__(self, representation):
        self.representation = representation

    def predict(self, data):
        """Given a group of words or a set of group of words, predict the "outlier position" within each group

        The "outlier position" (OP) refers to [1]_ :

        Given a set W of n + 1 words, OP is defined as the position of the outlier w_{n+1} according to the compactness
        score, which ranges from 0 to n (position 0 indicates the lowest overall score among all words in W, and
        position n indicates the highest overall score).

        References
        ----------
        .. [1] José Camacho-Collados and Roberto Navigli. Find the word that does not belong: A Framework for an
               Intrinsic Evaluation of Word Vector Representations. In Proceedings of the ACL Workshop on Evaluating
               Vector Space Representations for NLP, Berlin, Germany, August 12, 2016.

        Examples
        --------
        >>> # create a representation
        >>> import numpy as np
        >>> import mangoes
        >>> vocabulary = mangoes.Vocabulary(['january', 'february', 'march', 'pluto', 'mars', 'saturn'])
        >>> matrix = np.array([[1.0, 0.2], [0.9, 0.1], [1.1, 0.1], [0.3, 0.9], [0.2, 1.0], [0.1, 0.9]])
        >>> representation = mangoes.Embeddings(vocabulary, matrix)
        >>> # predict
        >>> import mangoes.evaluation.outlier
        >>> evaluator = mangoes.evaluation.outlier.Evaluator(representation)
        >>> evaluator.predict('january february march saturn')
        4
        >>> evaluator.predict(['january february march saturn', 'pluto saturn march'])
        {'january february march saturn': 4, 'pluto saturn march': 3}

        Parameters
        ----------
        data: str or iterable of str

        Returns
        -------
        int or dict
            If a string was given, the outlier position according to the compactness score.
            If a list of string was given, a dict with strings as keys and outlier positions as values

        """
        if isinstance(data, str):
            compactness_scores = []
            group = data.split()
            for w in group:
                all_except_w = self.representation.matrix[[self.representation.words.index(word)
                                                           for word in group
                                                           if word is not w]]
                all_except_w = mangoes.utils.arrays.Matrix.factory(all_except_w)
                compactness_scores.append(_pseudo_inversed_compactness_score(all_except_w, self.representation[w]))
            sorted_indices = reversed(np.asarray(compactness_scores, dtype=float).argsort())
            return list(sorted_indices).index(len(group) - 1) + 1
        else:
            return {cluster: self.predict(cluster) for cluster in data}


def _pseudo_inversed_compactness_score(vectors, w):
    similarities = mangoes.utils.metrics.pairwise_cosine_similarity(w, vectors)
    return similarities.sum()


# ####################################################################################################################
# EVALUATION

class Evaluation(BaseEvaluation):
    """
    Examples
    --------
    >>> # create a representation
    >>> import numpy as np
    >>> import mangoes
    >>> vocabulary = mangoes.Vocabulary(['january', 'february', 'march', 'pluto', 'mars', 'saturn'])
    >>> matrix = np.array([[1.0, 0.2], [0.9, 0.1], [1.1, 0.1], [0.3, 0.9], [0.2, 1.0], [0.1, 0.9]])
    >>> representation = mangoes.Embeddings(vocabulary, matrix)
    >>> import mangoes.evaluation.outlier
    >>> # evaluate
    >>> dataset = Dataset("test", ['january february march pluto', 'mars saturn pluto march'])
    >>> evaluation = mangoes.evaluation.outlier.Evaluation(representation, dataset)
    >>> print(evaluation.get_score())
    Score(opp=1.0, accuracy=1.0, nb=2)
    >>> print(evaluation.get_report()) # doctest: +NORMALIZE_WHITESPACE
                                                                Nb questions         OPP    accuracy
    ================================================================================================
    test                                                                 2/2     100.00%     100.00%
    ------------------------------------------------------------------------------------------------
    """
    _Score = namedtuple("Score", "opp accuracy nb")

    class _Report(PrintableReport):
        HEADER = [(("Nb questions", "OPP", "accuracy"), (3, 3, 3))]
        PREDICTION_HEADER = [(("outlier position",), (6,))]

        def _print_score(self, score):
            if score.nb:
                return "{:>{width}}{:>{width}}".format("{:.2%}".format(score.opp), "{:.2%}".format(score.accuracy),
                                                       width=3 * self.COL)
            else:
                return "{:>{width}}{:>{width}}".format('NA', 'NA', width=3 * self.COL)

        def _print_prediction(self, question, indent):
            outlier_position = self.evaluation.predictions[question]

            line = "{:{width}}".format('    ' * indent + question, width=(self.LINE_LENGTH - 1 * self.COL))
            line += "{:>{width}}".format(outlier_position, width=1 * self.COL)
            return line + "\n"

    def __init__(self, representation, *datasets, lower=True):
        super(Evaluation, self).__init__(Evaluator(representation), *datasets, lower=lower)

    def _filter_list_of_questions(self, list_of_questions):
        subset_clusters = []
        unique_clusters = set()
        nb_oov = 0
        nb_duplicates = 0

        for group in list_of_questions:
            group = group.lower() if self.lower else group

            if all(w in self.evaluator.representation.words for w in group.split()):
                subset_clusters.append(group)

                if group in unique_clusters:
                    nb_duplicates += 1
                else:
                    unique_clusters.add(group)
            else:
                nb_oov += 1

        result = self._FilteredSubset(subset_clusters, len(list_of_questions), nb_oov, nb_duplicates)
        return result, unique_clusters, unique_clusters

    def _score(self, clusters):
        if not clusters:
            return np.nan, np.nan

        predictions = [self.predictions[cluster] for cluster in clusters]
        opp_score = sum([op / len(c.split()) for c, op in zip(clusters, predictions)]) / len(clusters)

        outlier_detections = [op == len(c.split()) for op, c in zip(predictions, clusters)]
        accuracy = sum(outlier_detections) / len(predictions)

        return opp_score, accuracy
