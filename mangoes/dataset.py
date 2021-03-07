# -*- coding: utf-8 -*-
"""Module to access to available datasets and create new ones.

Datasets available in this module :

* `WS353` for the `WordSim353 <http://alfonseca.org/eng/research/wordsim353.html>`_ dataset
  (Finkelstein et al., 2002) [1]_.\n
  Also partitioned by [2]_ into :
    * `WS_SIM` : WordSim Similarity
    * `WS_REL` : WordSim Relatedness
* `RG65` for Rubenstein and Goodenough (1965) dataset [3]_
* `RAREWORD` for the Luong et al.'s (2013)
  `Rare Word (RW) Similarity Dataset <https://nlp.stanford.edu/~lmthang/morphoNLM/>`_ [4]_
* `MEN` for the Bruni et al.'s (2012) MEN dataset [5]_
* `MTURK` for the `Radinsky et al.'s (2011) Mechanical Turk dataset` [6]_
* `SIMLEX` for the Hill et al.'s (2016) SimLex-999 dataset [7]_
* `GOOGLE` for the Mikolov et al.'s (2013) Google dataset [8]_ .\n
  Also partitionned into :
    * `GOOGLE_SEMANTIC` for semantic analogies
    * `GOOGLE_SYNTACTIC` for syntactic analogies
* `MSR` for the Mikolov et al.'s (2013) Microsoft Research dataset [9]_
* `OD_8_8_8` [10]_
* `WIKI_SEM_500` [11]_

Warnings
--------
The Simlex dataset is not compatible with this version of mangoes

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
.. [7] Hill, F., Reichart, R., & Korhonen, A. (2016). Simlex-999: Evaluating semantic models with (genuine) similarity
       estimation. Computational Linguistics.
.. [8] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector
       space. arXiv preprint arXiv:1301.3781.
.. [9] Mikolov, T., Yih, W. T., & Zweig, G. (2013, June). Linguistic regularities in continuous space word
       representations. In hlt-Naacl (Vol. 13, pp. 746-751).
.. [10] José Camacho-Collados and Roberto Navigli. Find the word that does not belong: A Framework for an Intrinsic
        Evaluation of Word Vector Representations. In Proceedings of the ACL Workshop on Evaluating Vector Space
        Representations for NLP, Berlin, Germany, August 12, 2016.
.. [11]
"""

import warnings
warnings.warn("The dataset module is deprecated and will be removed in next release: use mangoes.evaluation package "
              "instead", DeprecationWarning)

import os.path
import zipfile
from collections import defaultdict, namedtuple

import pkg_resources

import mangoes.utils.exceptions
from mangoes.constants import ENCODING

_Question = namedtuple("Question", ["question", "gold"])


def _default_question_parser(question):
    return _Question(*question.rsplit(maxsplit=1))


def _outlier_detection_parser(question):
    return _Question(question, question.split()[-1])


class Dataset:
    """Base class to create an evaluation dataset.


    Parameters
    ----------
    content: str or list or dict
        Content of the dataset.Can be :
            - a path to a file containing the questions
            - a path to a directory containing such files : each subdirectory and each file will be
            considered as subsets.
            - a list of questions (each question is a string)
            Ex with analogies : ['king queen man woman', 'boy girl man woman', ...]
            - a dictionary containing list of questions structured in subsets.
            Ex : {"root": {"subset1":[...], "subset2":[...]}
    name: str
        name of the dataset
    lower: boolean
        True if the questions have to be lowercased

    Attributes
    ----------
    name
    lower
    subsets_to_questions: dict
        dictionary where keys are the name of the subsets and values are list of questions or nested subsets
    questions_to_subsets: dict
        dictionary where keys are the questions and values are the list of the subsets they belong to

    Examples
    --------
    >>> dataset = mangoes.dataset.Dataset(['a b c d', 'e f g h'], name="My dataset")
    >>> dataset.subsets_to_questions
    {'My dataset': ['a b c d', 'e f g h']}
    >>> dataset.questions_to_subsets
    {'a b c d': ['/My dataset'], 'e f g h': ['/My dataset']}

    >>> dataset = mangoes.dataset.Dataset({"subset1": ['a b c d','e f g h'], "subset2": ['a b c d']}, name="My dataset")
    >>> dataset.subsets_to_questions
    {'My dataset': {'subset1': ['a b c d', 'e f g h'], 'subset2': ['a b c d']}}
    >>> dataset.questions_to_subsets
    {'a b c d': {'/My dataset', '/My dataset/subset1', '/My dataset/subset2'},
     'e f g h': {'/My dataset', '/My dataset/subset1'}}

    """

    def __init__(self, content, name="", language="en", lower=True, question_parser=_default_question_parser):
        self.name = name
        self.language = language
        self.lower = lower
        self.parse_question = question_parser

        self.__questions_to_subsets = {}
        self.__subsets_to_questions = {}
        if isinstance(content, str):
            if not name:
                self.name = _get_name(content)
            content = self._read_from(content)
        else:
            content = {self.name: content}

        if self.lower:
            content = _lower_subset(content)

        self.subsets_to_questions = content

    @property
    def subsets_to_questions(self):
        return self.__subsets_to_questions

    @property
    def questions_to_subsets(self):
        return self.__questions_to_subsets

    @subsets_to_questions.setter
    def subsets_to_questions(self, subsets_to_questions):
        self.__subsets_to_questions = subsets_to_questions
        self.__questions_to_subsets = dict(self._flatten(self.__subsets_to_questions))

    def get_questions_and_gold(self, subset=None):
        """Returns a list of tuples with questions and expected answers

        Examples
        --------
        >>> dataset = Dataset({"My dataset": {"subset1": ['a b c d', 'e f g h'],
        >>>                                   "subset2": ['a b c d']}})
        >>> dataset.get_questions_and_gold()
        [Question(question='e f g', gold='h'), Question(question='a b c', gold='d')]
        >>> dataset.get_questions_and_gold("/My dataset/subset2")
        ['a b c d']

        Parameters
        ----------
        subset: str or None
            if None, return all the questions in the dataset, else, only the questions of the given subset.

        Returns
        -------
        list of tuples (question, gold)

        """
        if not subset:
            return [self.parse_question(q) for q in self.questions_to_subsets]
        return [self.parse_question(q) for q in self.questions_to_subsets
                if subset in self.questions_to_subsets[q]]

    @classmethod
    def _flatten(cls, subsets, prefix="", lower=True):
        result = defaultdict(set)

        for subset_name, subset_data in subsets.items():
            subset_path = prefix.split("/") + [subset_name]

            if isinstance(subset_data, dict):
                for question, list_of_subsets in cls._flatten(subset_data, prefix="/".join(subset_path)).items():
                    if lower:
                        question = question.lower()
                    result[question].update(list_of_subsets)

            else:
                for question_w_gold in subset_data:
                    if lower:
                        question_w_gold = question_w_gold.lower()
                    if subset_path:
                        result[question_w_gold].update(["/".join(subset_path[:i + 1])
                                                        for i in range(1, len(subset_path))
                                                        if subset_path[:i + 1]])
        return result

    @classmethod
    def _read_from(cls, path):
        if pkg_resources.resource_exists(__name__, path):
            path = pkg_resources.resource_filename(__name__, path)

        if os.path.isdir(path):
            return cls._read_from_dir(path)
        elif zipfile.is_zipfile(path):
            return cls._read_from_zip(path)
        else:
            try:
                return cls._read_from_file(path)
            except FileNotFoundError:
                raise mangoes.utils.exceptions.ResourceNotFound(path=path)

    @classmethod
    def _read_from_file(cls, path):
        with open(path, encoding=ENCODING) as file_content:
            return {_get_name(path): [line.strip() for line in file_content]}

    @classmethod
    def _read_from_dir(cls, path):
        questions = {}
        for subset in [os.path.join(path, subpath) for subpath in os.listdir(path)]:
            subset_name = _get_name(subset)
            if not subset_name.startswith('.'):
                content = cls._read_from(subset)
                questions[subset_name] = content[subset_name]
        return {_get_name(path): questions}

    @classmethod
    def _read_from_zip(cls, path):
        questions = {}
        with zipfile.ZipFile(path, 'r') as zip_file:
            for filename in zip_file.namelist():
                content = [line.strip() for line in zip_file.read(filename).decode().split('\n') if line.strip()]
                if content:
                    questions[_get_name(filename)] = content
        return {_get_name(path): questions}


class OutlierDetectionDataset(Dataset):
    """Base class for dataset for Outlier Detection task

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.parse_question = _outlier_detection_parser

    @classmethod
    def _read_from_file(cls, path):
        with open(path, encoding=ENCODING) as file_content:
            cluster, outliers = cls._parse_cluster_and_outliers(file_content)
            return {_get_name(path): [cluster + " " + outlier for outlier in outliers]}

    @classmethod
    def _read_from_zip(cls, path):
        questions = {}
        with zipfile.ZipFile(path, 'r') as zip_file:
            for filename in zip_file.namelist():
                cluster, outliers = cls._parse_cluster_and_outliers(zip_file.read(filename).decode().split('\n'))
                if outliers:
                    questions[_get_name(filename)] = [cluster + " " + outlier for outlier in outliers]
        return {_get_name(path): questions}

    @classmethod
    def _parse_cluster_and_outliers(cls, cluster_and_outliers):
        cluster = []
        outliers = []

        current = cluster
        for line in cluster_and_outliers:
            if not line.strip():
                current = outliers
                continue
            current.append(line.strip())
        return " ".join(cluster), outliers


def _get_name(content):
    return os.path.splitext(os.path.basename(content))[0]


def nb_questions(subset):
    """Returns the total number of questions in a subset of a Dataset

    Parameters
    ----------
    subset: dict
        subset of a Dataset

    Returns
    -------
    int

    """
    if not isinstance(subset, dict):
        return len(subset)
    return sum([nb_questions(v) for v in subset.values()])


def _lower_subset(subset):
    if isinstance(subset, list):
        return [y.lower() for y in subset]
    for key in subset.keys():
        subset[key] = _lower_subset(subset[key])
    return subset


WS353 = "ws353"
WS353_RELATEDNESS = "ws353_relatedness"
WS353_SIMILARITY = "ws353_similarity"
MEN = "men"
MTURK = "mturk"
RAREWORD = "rareword"
RG65 = "rg65"
SIMLEX = "simlex"
GOOGLE = "google"
GOOGLE_SEMANTIC = "google_semantic"
GOOGLE_SYNTACTIC = "google_syntactic"
MSR = "msr"
ANALOGY = "analogy"
SIMILARITY = "similarity"
OUTLIER_DETECTION = "outlier_detection"
OD_8_8_8 = "8-8-8"
WIKI_SEM_500 = "wiki_sem_500"

AVAILABLE_DATASETS = {"en": {WS353: ("resources/en/similarity/wordsim353.txt", Dataset),
                             WS353_RELATEDNESS: ("resources/en/similarity/ws353_relatedness.txt", Dataset),
                             WS353_SIMILARITY: ("resources/en/similarity/ws353_similarity.txt", Dataset),
                             MEN: ("resources/en/similarity/men.txt", Dataset),
                             MTURK: ("resources/en/similarity/mturk.txt", Dataset),
                             RAREWORD: ("resources/en/similarity/rareword.txt", Dataset),
                             RG65: ("resources/en/similarity/rg65.txt", Dataset),
                             # SIMLEX: ("resources/en/similarity/SimLex-999.txt", Dataset),
                             GOOGLE: ("resources/en/analogy/google", Dataset),
                             GOOGLE_SEMANTIC: ("resources/en/analogy/google/semantic", Dataset),
                             GOOGLE_SYNTACTIC: ("resources/en/analogy/google/syntactic", Dataset),
                             MSR: ("resources/en/analogy/msr/syntactic", Dataset),
                             ANALOGY: ("resources/en/analogy", Dataset),
                             SIMILARITY: ("resources/en/similarity", Dataset),
                             OUTLIER_DETECTION: ("resources/en/outlier_detection", OutlierDetectionDataset),
                             OD_8_8_8: ("resources/en/outlier_detection/8-8-8", OutlierDetectionDataset),
                             WIKI_SEM_500: (
                                 "resources/en/outlier_detection/wiki-sem-500-tokenized.zip", OutlierDetectionDataset)},
                      "fr": {RG65: ("resources/fr/similarity/rg65-fr.txt", Dataset),
                             SIMILARITY: ("resources/fr/similarity", Dataset), }
                      }


def load(dataset_name, language="en", lower=True):
    """Loads a dataset from the AVAILABLE_DATASETS

    Parameters
    ----------
    dataset_name: str
        the name of the dataset, must be in AVAILABLE_DATASETS
    language: {'en', 'fr'}
        Code of the language of the dataset (default = 'en')
    lower: boolean
        whether the questions of the dataset should be lowercased

    Returns
    -------
    mangoes.Dataset
    """
    if dataset_name == SIMLEX:
        raise NotImplementedError("The Simlex dataset is not yet compatible")
        # return _load_simlex(lower=lower)
    if not language:
        language = "en"
    path, handler = AVAILABLE_DATASETS[language][dataset_name]
    return handler(path, language=language, lower=lower)

# def _load_simlex(lower=True):
#     TODO : adapt to use the simlex dataset
# def parse_question(question):
#     word1, word2, _, gold, *_ = question.split()
#     return Dataset.Question(question=' '.join([word1, word2]), gold=gold)
#
# path, handler = AVAILABLE_DATASETS[SIMLEX]
# return handler(path, lower=lower, question_parser=parse_question)
