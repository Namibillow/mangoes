# -*- coding: utf-8 -*-
"""Base classes used by evaluation tasks in `evaluation` package.

"""
import abc
import os
import zipfile
from collections import namedtuple

import pkg_resources

import mangoes
from mangoes.constants import ENCODING


# ####################################################################################################################
# DATASETS

class BaseDataset:
    """Base abstract class to define datasets

    """
    def __init__(self, name, data):
        """

        Parameters
        ----------
        name: str
            The name of the dataset
        data: list or dict or str
            If list, the list contains the "questions" of the dataset. The format depends on the task.
            If dict, each key is a subset of the dataset; the value can be a list or another subset
            If str, the string should be the path to a file, a zip or a directory containing the dataset
        """
        self.name = name
        if isinstance(data, str):
            if not self._check_resource_exists(data):
                raise mangoes.utils.exceptions.ResourceNotFound(path=data)
            self._data = data
        else:
            def parse_data(content):
                if isinstance(content, list):
                    return [self.parse_question(l) for l in content]
                else:
                    return {subset : parse_data(content[subset]) for subset in content}

            self._data = parse_data(data)

    @classmethod
    @abc.abstractmethod
    def parse_question(self, question):
        """Parse a string into a namedtuple representing a question for the given task"""
        pass

    @classmethod
    def parse_file(cls, file_content):
        return [cls.parse_question(line) for line in file_content if line.strip()]

    @property
    def data(self):
        # lazy load the data from files
        if isinstance(self._data, str):
            self._data = self._read_from(self._data)
        return self._data

    def get_subset(self, path):
        data = self.data
        for subset in path.split('/'):
            data = data[subset]
        return data

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
            return cls.parse_file(file_content)

    @classmethod
    def _read_from_dir(cls, path):
        questions = {}
        for subset in [os.path.join(path, subpath) for subpath in os.listdir(path)]:
            subset_name = _get_name(subset)
            if not subset_name.startswith('.'):
                questions[subset_name] = cls._read_from(subset)
        return questions

    @classmethod
    def _read_from_zip(cls, path):
        questions = {}
        with zipfile.ZipFile(path, 'r') as zip_file:
            for filename in zip_file.namelist():
                content = cls.parse_file(zip_file.read(filename).decode().split('\n'))
                if content:
                    subset_name = mangoes.evaluation.base._get_name(filename)
                    questions[subset_name] = content
        return questions

    @staticmethod
    def _check_resource_exists(path):
        if pkg_resources.resource_exists(__name__, path):
            return True

        return os.path.exists(path)


class BaseEvaluator:
    """Base abstract class for all Evaluator

    An Evaluator is a class that, given a representation, will predict results for the task it is implementing.
    """
    @abc.abstractmethod
    def predict(self, question):
        pass


class PrintableReport:
    """Base abstract class to print Evaluation reports"""
    LINE_LENGTH = 96
    COL = 4

    def __init__(self, evaluation, keep_duplicates=True):
        self.evaluation = evaluation
        self.keep_duplicates = keep_duplicates

    def __str__(self):
        return self.to_string()

    def to_string(self, show_subsets=False, show_questions=False):
        string = "\n"
        string += self._print_header()
        for dataset in self.evaluation.datasets:
            string += self._print_subset(dataset, None, show_subsets, show_questions)
            string += '-' * self.LINE_LENGTH + "\n"
        return string

    def _print_header(self):
        string = ""
        for line in self.HEADER:
            string += "{:>{width}}".format("", width=self.LINE_LENGTH - sum(line[1]) * self.COL)
            for h, l in zip(*line):
                string += "{:>{width}}".format(h, width=l * self.COL)
            string += "\n"
        return string + "=" * self.LINE_LENGTH + "\n"

    def _print_predictions_header(self):
        string = ""
        for line in self.PREDICTION_HEADER:
            string = "{:{width}}".format("", width=(self.LINE_LENGTH - sum(line[1]) * self.COL))
            for label, length in zip(*line):
                string += "{:>{width}}".format(label, width=length * self.COL)
            string += "\n"
        return string

    def _print_subset(self, dataset, subset_path=None, show_subsets=False, show_questions=False, indent=0):
        full_subset_path = dataset.name + '/' + subset_path if subset_path else dataset.name
        data = dataset.get_subset(subset_path) if subset_path else dataset.data

        string = self._print_score_line(full_subset_path, indent)

        if isinstance(data, list):
            if show_questions and data:
                string += self._print_predictions(full_subset_path, indent)
        elif show_subsets or show_questions:
            string += "\n"
            for subset in sorted(data):
                string += self._print_subset(dataset, (subset_path + '/' if subset_path else '') + subset,
                                             show_subsets, show_questions, indent + 1)
            if not show_questions: string += "\n"

        return string

    def _print_predictions(self, subset_path, indent=0):
        questions = self.evaluation._questions_by_subset[subset_path].questions#['questions']
        if questions:
            if not self.keep_duplicates:
                # remove duplicates preserving order
                import collections
                questions = list(collections.OrderedDict.fromkeys(questions))

            string = "\n"
            string += self._print_predictions_header()

            for similarity in questions:
                string += self._print_prediction(similarity, indent)

            return string + "\n"
        return "\n"

    def _print_score_line(self, name, indent=None):
        score = self.evaluation.get_score(dataset=name, keep_duplicates=self.keep_duplicates)
        nb_total_questions_in_original = self.evaluation._questions_by_subset[name].nb_total#["nb_total"]
        nb_duplicates = self.evaluation._questions_by_subset[name].nb_duplicates#["nb_duplicates"]

        line = "{:<{width}}".format('    ' * indent + name.split('/')[-1], width=(self.LINE_LENGTH - 3 * 3 * self.COL))
        line += "{:>{width}}".format("{}/{}".format(score.nb, nb_total_questions_in_original), width=3 * self.COL)

        line += self._print_score(score)

        if nb_duplicates:
            prefix = 'including ' if self.keep_duplicates else '-'
            plural = 's' if nb_duplicates > 1 else ''
            line += "\n{:>{width}}".format('    ' * indent + "({}{} duplicate{})".format(prefix, nb_duplicates, plural),
                                           width=(self.LINE_LENGTH - 3 * 2 * self.COL))
        return line + "\n"

    @abc.abstractmethod
    def _print_score(self, score):
        pass


class BaseEvaluation:
    _Report = PrintableReport
    _FilteredSubset = namedtuple('PreparedSubset', 'questions nb_total nb_oov nb_duplicates')
    # A Filtered Subset is the result of the filtering of a dataset, according to a representation:
    #     questions = the questions of this subset that can be answered with the representation
    #     nb_total = total number of questions in the original dataset, including oov and duplicates
    #     nb_oov = number of ignored questions because some terms are not in the represented vocabulary
    #     nb_duplicates = number of duplicates after removing oov questions

    def __init__(self, evaluator, *datasets, lower=True, evaluator_kwargs={}):
        self.evaluator = evaluator
        self.evaluator_kwargs = evaluator_kwargs
        self.datasets = datasets
        self.lower = lower

        self.predictions, self._questions_by_subset = self._prepare()

    @abc.abstractmethod
    def _filter_list_of_questions(self, list_of_questions):
        pass

    def _prepare(self):
        """
        Prepare the dataset(s) : count and remove oov questions, count duplicates,
        make predictions for all remaining questions
        Returns
        -------

        """
        questions_by_subset = {}
        unique_questions = set()

        def _prepare_data(name, data):
            questions_by_subset[name] = {}
            if isinstance(data, list):
                filtered_data, unique_questions_in_data, unique_with_gold_in_data = self._filter_list_of_questions(data)

                questions_by_subset[name] = filtered_data
                unique_questions.update(unique_questions_in_data)

                return unique_with_gold_in_data
            else:
                questions_in_data = []  # all ordered considered questions in this dataset, with duplicates
                unique_with_gold_in_data = set() # keep track of questions to count duplicates
                nb_total_in_data, nb_oov_in_data, nb_duplicates_in_data = 0, 0, 0

                for subset in data:
                    unique_with_gold_in_subset = _prepare_data(name + '/' + subset, data[subset])
                    filtered_subset = questions_by_subset[name + '/' + subset]

                    # add the considered questions of this subset
                    questions_in_data.extend(filtered_subset.questions)

                    nb_total_in_data += filtered_subset.nb_total
                    nb_oov_in_data += filtered_subset.nb_oov

                    # count duplicates : the ones within the subset ...
                    nb_duplicates_in_data += filtered_subset.nb_duplicates
                    # ... and the ones between the subsets
                    nb_duplicates_in_data += sum([q in unique_with_gold_in_data for q in set(filtered_subset.questions)])

                    unique_with_gold_in_data.update(unique_with_gold_in_subset)

                questions_by_subset[name] = self._FilteredSubset(questions_in_data,
                                                                 nb_total_in_data, nb_oov_in_data, nb_duplicates_in_data)
                return unique_with_gold_in_data

        for dataset in self.datasets:
            _prepare_data(dataset.name, dataset.data)

        return self.evaluator.predict(unique_questions, **self.evaluator_kwargs), questions_by_subset

    def get_score(self, dataset=None, keep_duplicates=True):
        """Return the score(s) of the evauation

        Parameters
        ----------
        dataset: str or None
            If several datasets were given, name of the dataset to get the score.
        keep_duplicates: bool
            Whether or not the duplicates should be kept or ignored to get the score

        Returns
        -------
        tuple or dict
            If only one dataset, returns the Score for this dataset : the Score is a namedtuple that depends on the task
            If several datasets, return a dict with datasets names as keys and Score as values

        """
        if dataset:
            if keep_duplicates:
                questions = self._questions_by_subset[dataset].questions
            else:
                questions = set(self._questions_by_subset[dataset].questions)

            return self._Score(*self._score(questions),
                               len(questions))
        else:
            if len(self.datasets) == 1:
                return self.get_score(self.datasets[0].name, keep_duplicates=keep_duplicates)
            return {dataset.name: self.get_score(dataset.name, keep_duplicates=keep_duplicates)
                    for dataset in self.datasets}

    def get_report(self, keep_duplicates=True, show_subsets=False, show_questions=False):
        """Gets a PrintableReport for this evaluation

        Parameters
        ----------
        keep_duplicates: bool
            Whether or not the duplicates should be kept or ignored to get the scores
        show_subsets
            Whether or not to see the scores for all subsets in the report
        show_questions
            Whether or not to see predictions in the report

        Returns
        -------
        str
            The formatted report to be print

        """
        return self._Report(self, keep_duplicates).to_string(show_subsets, show_questions)


def _get_name(content):
    return os.path.splitext(os.path.basename(content))[0]
