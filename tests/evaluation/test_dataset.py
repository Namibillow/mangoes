# -*- coding: utf-8 -*-

import logging
import pytest

import mangoes.evaluation.base
import mangoes.utils.exceptions

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


class GenericDataset(mangoes.evaluation.base.BaseDataset):
    @classmethod
    def parse_question(self, question):
        return question.strip()


def test_user_dataset():
    dataset = GenericDataset("My dataset", ['a b c d', 'e f g h'])
    assert ['a b c d', 'e f g h'] == dataset.data


def test_user_dataset_with_subsets():
    dataset = GenericDataset("My dataset", {"subset1": ['a b c d', 'e f g h'],
                                            "subset2": ['a b c d']})

    assert {"subset1": ['a b c d', 'e f g h'],
            "subset2": ['a b c d']} == dataset.data

    assert ['a b c d', 'e f g h'] == dataset.get_subset("subset1")
    assert ['a b c d'] == dataset.get_subset("subset2")


def test_user_dataset_file(dataset_file_path):
    dataset = GenericDataset("dataset", dataset_file_path)

    assert "dataset" == dataset.name
    assert ['a b c d', 'e f g h'] == dataset.data


def test_user_dataset_dir(dataset_dir_path):
    dataset = GenericDataset("dataset", dataset_dir_path)

    assert {"subset1": ['a b c d', 'e f g h'],
            "subset2": ['i j k l']} == dataset.data


def test_user_dataset_zip(dataset_zip_path):
    dataset = GenericDataset("dataset", dataset_zip_path)

    assert {"subset1": ['a b c d', 'e f g h'],
            "subset2": ['i j k l']} == dataset.data


def test_ws353_has_353_questions():
    import mangoes.evaluation.similarity
    ws353 = mangoes.evaluation.similarity.WS353
    assert 353 == len(ws353.data)


def test_google_semantic_has_5_subsets():
    import mangoes.evaluation.analogy
    google_semantic_dataset = mangoes.evaluation.analogy.GOOGLE_SEMANTIC

    assert {"capital-common-countries", "capital-world", "city-in-state", "currency",
            "family"} == set(google_semantic_dataset.data.keys())


def test_8_8_8_dataset_has_64_questions():
    import mangoes.evaluation.outlier
    _8_8_8_dataset = mangoes.evaluation.outlier._8_8_8
    assert 8 == len(_8_8_8_dataset.data)
    for subset in _8_8_8_dataset.data:
        assert 8 == len(_8_8_8_dataset.data[subset])


def test_wiki_sem_500_has_500_subsets():
    import mangoes.evaluation.outlier
    wikisem500_dataset = mangoes.evaluation.outlier.WIKI_SEM_500
    assert 500 == len(wikisem500_dataset.data)


def test_analogy_parse_question():
    import mangoes.evaluation.analogy
    assert ('a b c', 'd') == mangoes.evaluation.analogy.Dataset.parse_question('a b c d')


def test_similarity_parse_question():
    import mangoes.evaluation.similarity
    assert (('a', 'b'), 0.5) == mangoes.evaluation.similarity.Dataset.parse_question('a b 0.5')


def test_outlier_detection_parse_question():
    import mangoes.evaluation.outlier
    assert 'a b c' == mangoes.evaluation.outlier.Dataset.parse_question('a b c')


# exceptions
def test_exception_not_existing_dataset():
    with pytest.raises(mangoes.utils.exceptions.ResourceNotFound):
        GenericDataset("not existing", "not/existing/path")


# ###########################################################################################
# ### FIXTURES

@pytest.fixture
def dataset_file_path(tmpdir_factory):
    dataset_file = tmpdir_factory.mktemp('data').join('dataset.txt')
    dataset_file.write_text('a b c d\ne f g h', encoding="utf-8")
    return str(dataset_file)


@pytest.fixture
def dataset_dir_path(tmpdir_factory):
    dataset_dir = tmpdir_factory.mktemp('dataset')

    subset_file1 = dataset_dir.join('subset1.txt')
    subset_file1.write_text('a b c d\ne f g h', encoding="utf-8")

    subset_file2 = dataset_dir.join('subset2.txt')
    subset_file2.write_text('i j k l\n', encoding="utf-8")

    return str(dataset_dir)


@pytest.fixture
def dataset_zip_path(tmpdir_factory):
    import zipfile
    import os
    zip_filepath = os.path.join(str(tmpdir_factory.mktemp('zip')), 'dataset.zip')
    with zipfile.ZipFile(zip_filepath, "w") as zip_file:
        zip_file.writestr('subset1.txt', 'a b c d\ne f g h')
        zip_file.writestr('subset2.txt', 'i j k l\n')
    return zip_filepath
