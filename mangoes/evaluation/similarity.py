# -*- coding: utf-8 -*-
"""Classes and functions to evaluate embeddings according to the "Similarity" task.

The Similarity task computes the correlation between the similarities of word pairs according to their representation
and according to human-assigned scores.

Datasets available in this module :

* `WS353` for the `WordSim353 <http://alfonseca.org/eng/research/wordsim353.html>`_ dataset
  (Finkelstein et al., 2002) [1]_.
  Also partitioned by [2]_ into :
    * `WS_SIM` : WordSim Similarity
    * `WS_REL` : WordSim Relatedness
* `RG65` for Rubenstein and Goodenough (1965) dataset [3]_
* `RAREWORD` for the Luong et al.'s (2013)
  `Rare Word (RW) Similarity Dataset <https://nlp.stanford.edu/~lmthang/morphoNLM/>`_ [4]_
* `MEN` for the Bruni et al.'s (2012) MEN dataset [5]_
* `MTURK` for the `Radinsky et al.'s (2011) Mechanical Turk dataset` [6]_

References
----------
.. [1] Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., & Ruppin, E. (2001, April).
       Placing search in context: The concept revisited. In Proceedings of the 10th international conference on World
       Wide Web (pp. 406-414). ACM.
.. [2] Eneko Agirre, Enrique Alfonseca, Keith Hall, Jana Kravalova, Marius Pasca, Aitor Soroa, A Study on Similarity
       and Relatedness Using Distributional and WordNet-based Approaches, In Proceedings of NAACL-HLT 2009.
.. [3] Rubenstein, Herbert, and John B. Goodenough. Contextual correlates of synonymy. Communications of the ACM,
       8(10):627–633, 1965.
.. [4] Luong, T., Socher, R., & Manning, C. D. (2013, August). Better word representations with recursive neural
       networks for morphology. In CoNLL (pp. 104-113).
.. [5] Bruni, E., Boleda, G., Baroni, M., & Tran, N. K. (2012, July). Distributional semantics in technicolor. In
       Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Long Papers-Volume 1
       (pp. 136-145). Association for Computational Linguistics.
.. [6] Radinsky, K., Agichtein, E., Gabrilovich, E., & Markovitch, S. (2011, March). A word at a time: computing word
       relatedness using temporal semantic analysis. In Proceedings of the 20th international conference on
       World wide web (pp. 337-346). ACM.


"""
import numpy as np
import scipy.stats
from collections import namedtuple

import mangoes
import mangoes.utils.metrics
from mangoes.evaluation.base import PrintableReport, BaseEvaluation, BaseEvaluator

_Similarity = namedtuple("Similarity", "word_pair gold")


# ####################################################################################################################
# DATASETS

class Dataset(mangoes.evaluation.base.BaseDataset):
    """Class to create a Dataset of word pairs similarities, to be used in Evaluation class

    Examples
    --------
    >>> from mangoes.evaluation.similarity import Dataset
    >>> user_dataset = Dataset("user dataset", ['lion tiger 0.8', 'sun phone 0.1'])

    Predefined datasets are available in this module:

    >>> import mangoes.evaluation.similarity
    >>> ws353 = mangoes.evaluation.similarity.WS353

    """

    @classmethod
    def parse_question(cls, question):
        """
        Examples
        --------
        >>> Dataset.parse_question('lion tiger 0.8')
        Similarity(word_pair=('lion', 'tiger'), gold=0.8)

        Parameters
        ----------
        question: str
            A splittable string with the word pair and a score

        Returns
        -------
        namedtuple

        """
        if isinstance(question, _Similarity):
            return question
        a, b, gold = question.strip().split()
        return _Similarity((a, b), float(gold))


WS353 = Dataset("WS353", "../resources/en/similarity/wordsim353.txt")
WS353_RELATEDNESS = Dataset("WS353 relatedness", "../resources/en/similarity/ws353_relatedness.txt")
WS353_SIMILARITY = Dataset("WS353 similarity", "../resources/en/similarity/ws353_similarity.txt")
MEN = Dataset("MEN", "../resources/en/similarity/men.txt")
MTURK = Dataset("M. Turk", "../resources/en/similarity/mturk.txt")
RAREWORD = Dataset("Rareword", "../resources/en/similarity/rareword.txt")
RG65 = Dataset("RG65", "../resources/en/similarity/rg65.txt")

ALL_DATASETS = (WS353, WS353_RELATEDNESS, WS353_SIMILARITY, MEN, MTURK, RAREWORD, RG65)


# ####################################################################################################################
# EVALUATOR

class Evaluator(BaseEvaluator):
    def __init__(self, representation):
        """Evaluator to predict similarity scores for word pairs according to the given representation

        Parameters
        ----------
        representation: mangoes.Representation
            The Representation to use
        """
        self.representation = representation

    def predict(self, word_pairs, metric=mangoes.utils.metrics.rowwise_cosine_similarity):
        """Predict the similarity scores for the given word pair(s).

        Examples
        --------
        >>> # create a representation
        >>> import numpy as np
        >>> import mangoes
        >>> vocabulary = mangoes.Vocabulary(['lion', 'tiger', 'sun', 'moon', 'phone', 'germany'])
        >>> matrix = np.array([[1, 0], [1, 0.2], [0, 1], [0, 1.2], [0.7, 0.7], [0.7, 0.8]])
        >>> representation = mangoes.Embeddings(vocabulary, matrix)
        >>> # predict
        >>> import mangoes.evaluation.similarity
        >>> evaluator = mangoes.evaluation.similarity.Evaluator(representation)
        >>> evaluator.predict(('lion', 'tiger'))
        array([ 0.98058068])
        >>> evaluator.predict([('lion', 'tiger'), ('sun', 'phone')])
        {('lion', 'tiger'): 0.98058067569092011, ('sun', 'phone'): 0.70710678118654757}

        Parameters
        ----------
        word_pairs: tuple of 2 str or list of tuples of 2 str
            a word pair or a list of word pairs
        metric
            the metric to use to compute the similarity (default : cosine)

        Returns
        -------
        dict
            A dictionary with analogies as keys and the Predictions as values

        """
        if isinstance(word_pairs, tuple):
            w1, w2 = word_pairs
            return metric(self.representation[w1], self.representation[w2])[0]

        words_pairs_indices = np.array([[self.representation.words.index(q[0]),
                                         self.representation.words.index(q[1])]
                                        for q in word_pairs])

        first_terms = self.representation.matrix[words_pairs_indices[:, 0], :]
        second_terms = self.representation.matrix[words_pairs_indices[:, 1], :]

        return {q: p for q, p in zip(word_pairs, metric(first_terms, second_terms))}


# ####################################################################################################################
# EVALUATION
class Evaluation(BaseEvaluation):
    """Class to evaluate a representation on a dataset or a list of datasets

    Both Pearson and Spearman coefficient are given.

    Examples
    --------
    >>> # create a representation
    >>> import numpy as np
    >>> import mangoes
    >>> vocabulary = mangoes.Vocabulary(['lion', 'tiger', 'sun', 'moon', 'phone', 'germany'])
    >>> matrix = np.array([[1, 0], [1, 0.2], [0, 1], [0, 1.2], [0.7, 0.7], [0.7, 0.8]])
    >>> representation = mangoes.Embeddings(vocabulary, matrix)
    >>> # evaluate
    >>> import mangoes.evaluation.similarity
    >>> dataset = Dataset("test", ['lion tiger 0.8', 'sun moon 0.8', 'phone germany 0.3'])
    >>> evaluation = mangoes.evaluation.similarity.Evaluation(representation, dataset)
    >>> evaluation.get_score() # doctest: +NORMALIZE_WHITESPACE
    Score(pearson=Coeff(coeff=-0.40705977800644011, pvalue=0.73310813349301363),
          spearman=Coeff(coeff=0.0, pvalue=1.0), nb=3)
    >>> print(evaluation.get_report()) # doctest: +NORMALIZE_WHITESPACE
                                                                              pearson       spearman
                                                          Nb questions        (p-val)        (p-val)
    ================================================================================================
    test                                                           3/3  -0.407(7e-01)     0.0(1e+00)
    ------------------------------------------------------------------------------------------------


    Parameters
    ----------
    representation: mangoes.Representation
        The representation to evaluate
    datasets: Dataset
        The dataset(s) to use
    lower: bool
        Whether or not the analogies in the dataset should be lowered
    metric
        the metric to use to compute the similarity (default : cosine)
    """
    _Coeff = namedtuple("Coeff", ["coeff", "pvalue"])
    _Score = namedtuple("Score", "pearson spearman nb")

    class _Report(PrintableReport):
        COL = 5
        HEADER = [(("pearson", "spearman"), (3, 3)),
                  (("Nb questions", "(p-val)", "(p-val)"), (3, 3, 3))]
        PREDICTION_HEADER = [(("gold", "score", ""), (3, 3, 6))]

        def _print_score(self, score):
            string = "{:>{width}}".format("{:.3}({:.0e})".format(*score.pearson), width=3 * self.COL)
            string += "{:>{width}}".format("{:.3}({:.0e})".format(*score.spearman), width=3 * self.COL)
            return string

        def _print_prediction(self, similarity, indent):
            line = "{:{width}}".format('    ' * indent + ' '.join(similarity.word_pair),
                                       width=(self.LINE_LENGTH - 12 * self.COL))
            line += "{:>{width}}".format(similarity.gold, width=3 * self.COL)
            line += "{:>{width}}".format("{:.2}".format(self.evaluation.predictions[similarity.word_pair]),
                                         width=3 * self.COL)
            return line + "\n"

    def __init__(self, representation, *datasets, lower=True, metric=mangoes.utils.metrics.rowwise_cosine_similarity):
        super(Evaluation, self).__init__(Evaluator(representation),
                                         *datasets, lower=lower, evaluator_kwargs={'metric': metric})

    def _filter_list_of_questions(self, list_of_questions):
        subset_similarities = []  # word pairs that can be predicted, respecting duplicates and order from original
        unique_word_pairs = set()  # unique word pairs to resolve
        unique_word_pairs_with_gold = set()  # unique word pairs with expected similarity to detect duplicates
        nb_oov = 0  # number of ignored pairs due to OOV terms
        nb_duplicates = 0  # number of duplicates among the filtered pairs

        for similarity in list_of_questions:
            if self.lower:
                similarity = _Similarity((similarity.word_pair[0].lower(), similarity.word_pair[1].lower()),
                                         similarity.gold)

            (a, b), s = similarity

            if a in self.evaluator.representation.words and b in self.evaluator.representation.words:
                subset_similarities.append(similarity)

                if similarity in unique_word_pairs_with_gold:
                    nb_duplicates += 1
                else:
                    unique_word_pairs_with_gold.add(similarity)

                unique_word_pairs.add((a, b))

            else:
                nb_oov += 1

        result = self._FilteredSubset(subset_similarities, len(list_of_questions), nb_oov, nb_duplicates)
        return result, unique_word_pairs, unique_word_pairs_with_gold

    def _score(self, similarities):
        predictions = [self.predictions[similarity.word_pair] for similarity in similarities]
        gold = [similarity.gold for similarity in similarities]
        pearson = self._Coeff(*scipy.stats.pearsonr(predictions, gold))
        spearman = self._Coeff(*scipy.stats.spearmanr(predictions, gold))
        return pearson, spearman
