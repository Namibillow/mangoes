# -*- coding: utf-8 -*-
"""Classes and functions to evaluate embeddings according to the "Analogy" task.

The Analogy task tries to predict the answer of the question of the form : a is to b as c is to ...
It uses both 3CosAdd [2]_ and 3CosMul [3]_ methods to solve them

Datasets available in this module :

* `GOOGLE` for the Mikolov et al.'s (2013) Google dataset [1]_ . Also partitionned into :

    * `GOOGLE_SEMANTIC` for semantic analogies
    * `GOOGLE_SYNTACTIC` for syntactic analogies

* `MSR` for the Mikolov et al.'s (2013) Microsoft Research dataset [2]_

References
----------
.. [1] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector
       space. arXiv preprint arXiv:1301.3781.
.. [2] Mikolov, T., Yih, W. T., & Zweig, G. (2013, June). Linguistic regularities in continuous space word
       representations. In hlt-Naacl (Vol. 13, pp. 746-751).
.. [3] Levy, O., Goldberg, Y., & Ramat-Gan, I. (2014). Linguistic Regularities in Sparse and Explicit Word
       Representations. In CoNLL (pp. 171–180).


"""
import heapq
from collections import namedtuple

import numpy as np

import mangoes
import mangoes.utils.arrays
import mangoes.utils.metrics
from mangoes.evaluation.base import PrintableReport, BaseEvaluation, BaseEvaluator

_Analogy = namedtuple("Analogy", "abc gold")
_Analogy.__doc__ = """Analogy with expected answer
        
        Parameters
        ----------
        abc: str
            The 3 terms of the analogy (a is to b as c is to ...) as a single splittable string. 
        gold: str
            The expected answer
            
        Examples
        --------
        >>> Analogy('paris france london', 'england')
        Analogy(abc='paris france london', gold='england')
        >>> Dataset.parse_question('paris france london england')
        Analogy(abc='paris france london', gold='england')
        """


# ####################################################################################################################
# DATASETS

class Dataset(mangoes.evaluation.base.BaseDataset):
    """Class to create a Dataset of analogies, to be used in Evaluation class

    Examples
    --------
    >>> from mangoes.evaluation.analogy import Dataset
    >>> user_dataset = Dataset("user dataset", ['paris france london england', 'get gets do does'])
    >>> capitals = Dataset("google", "../resources/en/analogy/google/semantic/capital-world.txt")

    2 analogy datasets are available in this module:

    - the GOOGLE dataset, also split in GOOGLE_SEMANTIC and GOOGLE_SYNTACTIC :
    >>> import mangoes.evaluation.analogy
    >>> google = mangoes.evaluation.analogy.GOOGLE
    >>> google_sem = mangoes.evaluation.analogy.GOOGLE_SEMANTIC
    >>> google_syn = mangoes.evaluation.analogy.GOOGLE_SYNTACTIC

    - the MSR dataset :
    >>> import mangoes.evaluation.analogy
    >>> msr = mangoes.evaluation.analogy.MSR

    """

    @classmethod
    def parse_question(cls, question):
        """
        Examples
        --------
        >>> Dataset.parse_question('paris france london england')
        Analogy(abc='paris france london', gold='england')

        Parameters
        ----------
        question: str
            A splittable string with the 4 terms of the analogies

        Returns
        -------
        namedtuple

        """
        if isinstance(question, _Analogy):
            return question
        a, b, c, d = question.strip().split()
        return _Analogy(' '.join((a, b, c)), d)


GOOGLE = Dataset("Google", "../resources/en/analogy/google")
GOOGLE_SEMANTIC = Dataset("Google Semantic", "../resources/en/analogy/google/semantic")
GOOGLE_SYNTACTIC = Dataset("Google Syntactic", "../resources/en/analogy/google/syntactic")
MSR = Dataset("MSR", "../resources/en/analogy/msr")

ALL_DATASETS = (GOOGLE, MSR)


# ####################################################################################################################
# EVALUATOR

class Evaluator(BaseEvaluator):
    _Prediction = namedtuple("Prediction", "using_cosadd using_cosmul")

    def __init__(self, representation, threshold=300000):
        """Evaluator to resolve analogies according to the given representation

        Parameters
        ----------
        representation: mangoes.Representation
            The Representation to use
        threshold: int
            A threshold to reduce the size of vocabulary of the representation for fast approximate evaluation
            (default is 300000 as in word2vec)
        """
        self.threshold = min(threshold, len(representation.words))

        self.representation = representation
        self.vocabulary = representation.words[:threshold]

        if self.representation.matrix.all_positive():
            self.cosine_similarity = mangoes.utils.metrics.pairwise_cosine_similarity
        else:
            self.cosine_similarity = mangoes.utils.metrics.pairwise_non_negative_cosine_similarity

    def _prepare(self, abc_vocabulary):
        abc_matrix = self.representation.matrix[[self.representation.words.index(word) for word in abc_vocabulary]]
        abc_matrix = mangoes.utils.arrays.Matrix.factory(abc_matrix)

        return self.cosine_similarity(abc_matrix, self.representation.matrix[:self.threshold])

    def predict(self, analogies, allowed_answers=1, epsilon=0.001, batch=1000):
        """Predict the answer for the given analogy question(s).

        Examples
        --------
        >>> # create a representation
        >>> import numpy as np
        >>> import mangoes
        >>> vocabulary = mangoes.Vocabulary(['paris', 'france', 'london', 'england', 'belgium', 'germany'])
        >>> matrix = np.array([[1, 0], [1, 0.2], [0, 1], [0, 1.2], [0.7, 0.7], [0.7, 0.8]])
        >>> representation = mangoes.Embeddings(vocabulary, matrix)
        >>> # predict
        >>> import mangoes.evaluation.analogy
        >>> evaluator = mangoes.evaluation.analogy.Evaluator(representation)
        >>> evaluator.predict('paris france london')
        Prediction(using_cosadd=['england'], using_cosmul=['england'])

        Parameters
        ----------
        analogies: str or list of str
            an analogy or a list of analogies to resolve in the form 'a b c' : a is to b as c is to ...
        allowed_answers
            number of answers to predict
        epsilon
            value to use as epsilon when computing 3CosMul
        batch
            As this function needs to compute the similarities between all the words in the analogies and all the
            words in the vocabulary, it can be memory-consuming. This parameter allowed to slice the list in batches.
            You can increase it to run faster or decrease it if you run out of memory.

        Returns
        -------
        namedtuple or dict
            If the input is a single analogy, returns a tuple with both predictions using cosadd and cosmul.
            If the input is a list of analogies, returns a dictionary with analogies as keys and
            the predictions as values.

        """
        if not analogies:
            return {}
        if isinstance(analogies, str):
            return self._predict([analogies], allowed_answers, epsilon)[analogies]

        analogies = list(analogies)
        start = 0
        result = {}

        while start < len(analogies):
            batch_analogies = analogies[start:min(start + batch, len(analogies))]
            result.update(self._predict(batch_analogies, allowed_answers, epsilon))
            start += batch

        return result

    def _predict(self, analogies, allowed_answers, epsilon):
        # get the words in the analogies and compute the similarities between them and all the words in the vocabulary
        analogies_terms = {w for q in analogies for w in q.split()}
        analogies_terms = mangoes.Vocabulary([w for w in analogies_terms if w in self.vocabulary])

        if not analogies_terms: # asked analogy words do not exist in embedding target word vocabulary 
            # TODO(nami) better error exception
            msg = ", ".join(w for q in analogies for w in q.split())
            raise mangoes.utils.exceptions.OutOfVocabulary(value=msg) 

        similarities = self._prepare(analogies_terms)
        analogies_indices = np.array([[analogies_terms.index(w) for w in analogy.split()] for analogy in analogies])

        # to remove the terms of the analogies from the result, we'll affect them -inf score
        terms_indices_in_candidates_vocabulary = [[self.vocabulary.index(word)
                                                   for word in abc.split() if word in self.vocabulary]
                                                  for abc in analogies]

        # compute scores with cosadd
        scores = self._3cosadd(similarities[analogies_indices[:, 0]],
                               similarities[analogies_indices[:, 1]],
                               similarities[analogies_indices[:, 2]])
        # exclude the terms of the questions from possible answers
        for i, qti in enumerate(terms_indices_in_candidates_vocabulary):
            scores[i, qti] = np.NINF

        best_answer_add = [[self.vocabulary[i] for i in _get_n_best(allowed_answers, scores[j])]
                           for j in range(len(analogies))]

        # compute scores with cosmul
        scores = self._3cosmul(similarities[analogies_indices[:, 0]],
                               similarities[analogies_indices[:, 1]],
                               similarities[analogies_indices[:, 2]],
                               epsilon)
        for i, qti in enumerate(terms_indices_in_candidates_vocabulary):
            scores[i, qti] = np.NINF
        best_answer_mul = [[self.vocabulary[i] for i in _get_n_best(allowed_answers, scores[j])]
                           for j in range(len(analogies))]

        return {abc: self._Prediction(cosadd, cosmul)
                for abc, cosadd, cosmul in zip(analogies, best_answer_add, best_answer_mul)}

    @staticmethod
    def _3cosadd(sim_with_a, sim_with_b, sim_with_c):
        return sim_with_b - sim_with_a + sim_with_c

    @staticmethod
    def _3cosmul(sim_with_a, sim_with_b, sim_with_c, epsilon=0.001):
        return sim_with_b * sim_with_c / (sim_with_a + epsilon)


# ####################################################################################################################
# EVALUATOR

class Evaluation(BaseEvaluation):
    """Class to evaluate a representation on a dataset or a list of datasets

    Examples
    --------
    >>> # create a representation
    >>> import numpy as np
    >>> import mangoes
    >>> vocabulary = mangoes.Vocabulary(['paris', 'france', 'london', 'england', 'berlin', 'germany'])
    >>> matrix = np.array([[1, 0], [1, 0.2], [0, 1], [0, 1.2], [0.7, 0.7], [0.7, 0.8]])
    >>> representation = mangoes.Embeddings(vocabulary, matrix)
    >>> # evaluate
    >>> import mangoes.evaluation.analogy
    >>> dataset = Dataset("test", ['paris france london england', 'paris france berlin germany'])
    >>> evaluation = mangoes.evaluation.analogy.Evaluation(representation, dataset)
    >>> evaluation.get_score()
    Score(cosadd=1.0, cosmul=0.5, nb=2)
    >>> print(evaluation.get_report()) # doctest: +NORMALIZE_WHITESPACE
                                                                Nb questions      cosadd      cosmul
    ================================================================================================
    test                                                                 2/2     100.00%      50.00%
    ------------------------------------------------------------------------------------------------


    Parameters
    ----------
    representation: mangoes.Representation
        The representation to evaluate
    datasets: Dataset
        The dataset(s) to use
    lower: bool
        Whether or not the analogies in the dataset should be lowered
    allowed_answers: int
        Nb of answers to consider when predicting an analogy (the analogy will be considered as correct if the
        expected answer is among the `allowed_answers` best answers)
    epsilon: float
        Value to be used as epsilon when computing 3CosMul
    threshold: int
        A threshold to reduce the size of vocabulary of the representation for fast approximate evaluation
        (default is 300000 as in word2vec)
    """

    class _Score:
        def __init__(self, nb_correct_cosadd, nb_correct_cosmul, nb_evaluated):
            self.nb = nb_evaluated

            if nb_evaluated:
                self.cosadd = nb_correct_cosadd / nb_evaluated
                self.cosmul = nb_correct_cosmul / nb_evaluated
            else:
                self.cosadd = np.nan
                self.cosmul = np.nan

        def __repr__(self):
            return "Score(cosadd={}, cosmul={}, nb={})".format(self.cosadd, self.cosmul, self.nb)

    class _Report(PrintableReport):
        HEADER = [(("Nb questions", "cosadd", "cosmul"), (3, 3, 3))]
        PREDICTION_HEADER = []

        def _print_score(self, score):
            if score.nb:
                return "{:>{width}}{:>{width}}".format("{:.2%}".format(score.cosadd), "{:.2%}".format(score.cosmul),
                                                       width=3 * self.COL)
            else:
                return "{:>{width}}{:>{width}}".format('NA', 'NA', width=3 * self.COL)

        def _print_prediction(self, analogy, indent):
            cosadd, cosmul = self.evaluation.predictions[analogy.abc]
            line = "{:{width}}".format('    ' * indent + ' '.join(analogy), width=(self.LINE_LENGTH - 6 * self.COL))
            line += "{:>{width}}".format(", ".join(cosadd), width=3 * self.COL)
            line += "{:>{width}}".format(", ".join(cosmul), width=3 * self.COL)
            return line + "\n"

    def __init__(self, representation, *datasets, lower=True, allowed_answers=1, epsilon=0.001, threshold=30000):
        super(Evaluation, self).__init__(Evaluator(representation, threshold=threshold),
                                         *datasets, lower=lower, evaluator_kwargs={'allowed_answers': allowed_answers,
                                                                                   'epsilon': epsilon})

    def _filter_list_of_questions(self, list_of_questions):
        subset_analogies = []  # analogies that can be predicted, respecting duplicates and order from original
        unique_abc = set()  # unique analogies to resolve
        unique_abcd = set()  # unique analogies with expected answers to detect duplicates
        nb_oov = 0  # number of ignored analogies due to OOV terms
        nb_duplicates = 0  # number of duplicates among the filtered analogies

        for analogy in list_of_questions:
            analogy = _Analogy(analogy.abc.lower(), analogy.gold.lower()) if self.lower else analogy
            (a, b, c), d = analogy.abc.split(), analogy.gold

            if all(w in self.evaluator.vocabulary for w in [a, b, c, d]):
                subset_analogies.append(analogy)

                if analogy in unique_abcd:
                    nb_duplicates += 1
                else:
                    unique_abcd.add(analogy)
                unique_abc.add(analogy.abc)
            else:
                nb_oov += 1

        result = self._FilteredSubset(subset_analogies, len(list_of_questions), nb_oov, nb_duplicates)
        return result, unique_abc, unique_abcd

    def _score(self, analogies):
        nb_correct_cosadd = sum([analogy.gold in self.predictions[analogy.abc].using_cosadd for analogy in analogies])
        nb_correct_cosmul = sum([analogy.gold in self.predictions[analogy.abc].using_cosmul for analogy in analogies])
        return nb_correct_cosadd, nb_correct_cosmul


# ####################################################################################################################
# HELPER FUNCTIONS
def _get_n_best(nb_best, scores):
    """From a vector of scores, returns the indices of the n best ones"""
    if nb_best == 1:
        return [np.nanargmax(scores)]

    return heapq.nlargest(nb_best, range(len(scores)), key=lambda i: scores[i])
