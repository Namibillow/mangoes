# -*- coding: utf-8 -*-

import logging

import pytest
from transformers import BertTokenizerFast
import numpy as np

from mangoes.modeling import MangoesLineByLineIterableDataset, MangoesLineByLineDataset, \
    MangoesLineByLineDatasetForNSP,MangoesTextClassificationDataset, MangoesQuestionAnsweringDataset, \
    MangoesCoreferenceDataset, MangoesMultipleChoiceDataset

logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)

bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_iterable_linebyline_dataset_length(raw_source_file, extra_small_source_file):
    dataset = MangoesLineByLineIterableDataset([raw_source_file, extra_small_source_file], bert_tokenizer)
    assert len(dataset) == 7


@pytest.mark.unittest
def test_iterable_linebyline_dataset_encoding(raw_source_file):
    dataset = MangoesLineByLineIterableDataset(raw_source_file, bert_tokenizer)
    assert np.array_equal(np.array([101, 3376, 2003, 2488, 2084, 9200,  102]), next(iter(dataset))['input_ids'].numpy())


@pytest.mark.unittest
def test_linebyline_dataset_length(raw_source_file, extra_small_source_file):
    dataset = MangoesLineByLineDataset([raw_source_file, extra_small_source_file], bert_tokenizer)
    assert len(dataset) == 7


@pytest.mark.unittest
def test_linebyline_dataset_encoding(raw_source_file):
    dataset = MangoesLineByLineDataset(raw_source_file, bert_tokenizer)
    assert np.array_equal(np.array([101, 3376, 2003, 2488, 2084, 9200,  102]), dataset[0]['input_ids'].numpy())


@pytest.mark.unittest
def test_linebyline_nsp_dataset_encoding(raw_source_file, extra_small_source_file):
    dataset = MangoesLineByLineDatasetForNSP([raw_source_file, extra_small_source_file], bert_tokenizer)
    assert len(dataset.documents) == 2


@pytest.mark.unittest
def test_seq_classification_encoding_indices(raw_small_source_string):
    dataset = MangoesTextClassificationDataset(raw_small_source_string, [0, 1], bert_tokenizer, max_len=20)
    outputs = dataset[0]
    target = bert_tokenizer(raw_small_source_string[0], return_tensors="pt", padding="max_length", max_length=20)
    assert np.array_equal(outputs['input_ids'].numpy(), target["input_ids"].flatten().numpy())
    assert outputs['labels'].numpy() == 0


@pytest.mark.unittest
def test_seq_classification_encoding_rawlabels(raw_small_source_string):
    dataset = MangoesTextClassificationDataset(raw_small_source_string, ["neg", "pos"], bert_tokenizer, max_len=20,
                                               label2id={"neg": 0, "pos": 1})
    outputs = dataset[0]
    target = bert_tokenizer(raw_small_source_string[0], return_tensors="pt", padding="max_length", max_length=20)
    assert np.array_equal(outputs['input_ids'].numpy(), target["input_ids"].flatten().numpy())
    assert outputs['labels'].numpy() == 0


@pytest.mark.unittest
def test_token_classification_encoding_indices(raw_source_string):
    dataset = MangoesTextClassificationDataset([raw_source_string[0]], [[0, 1, 0, 1, 0, 1, 0]], bert_tokenizer,
                                               max_len=20)
    outputs = dataset[0]
    target = bert_tokenizer(raw_source_string[0], return_tensors="pt", padding="max_length", max_length=20)
    assert np.array_equal(outputs['input_ids'].numpy(), target["input_ids"].flatten().numpy())
    assert np.array_equal(outputs['labels'].numpy(), [0, 1, 0, 1, 0, 1, 0])


@pytest.mark.unittest
def test_token_classification_encoding_rawlabels(raw_source_string):
    dataset = MangoesTextClassificationDataset([raw_source_string[0]],
                                               [[0, 1, 0, 1, 0, 1, 0]],
                                               bert_tokenizer, max_len=20, label2id={"neg": 0, "pos": 1})
    outputs = dataset[0]
    target = bert_tokenizer(raw_source_string[0], return_tensors="pt", padding="max_length", max_length=20)
    assert np.array_equal(outputs['input_ids'].numpy(), target["input_ids"].flatten().numpy())
    assert np.array_equal(outputs['labels'].numpy(), [0, 1, 0, 1, 0, 1, 0])


@pytest.mark.unittest
def test_bert_questionanswer_dataset(question_answering_data, tmpdir_factory):
    dataset = MangoesQuestionAnsweringDataset(bert_tokenizer,
                                              question_texts=question_answering_data[0],
                                              context_texts=question_answering_data[1],
                                              answer_texts=question_answering_data[2],
                                              start_indices=question_answering_data[3])
    assert len(dataset) == 2

    outputs = dataset[0]
    assert outputs["start_positions"].item() == 10
    assert outputs["end_positions"].item() == 10


@pytest.mark.unittest
def test_bert_multiplechoice_dataset(multiple_choice_example, tmpdir_factory):
    dataset = MangoesMultipleChoiceDataset(bert_tokenizer, question_texts=[multiple_choice_example[0]] * 4,
                                           choices_texts=[multiple_choice_example[1]] * 4, labels=[0] * 4,
                                           max_seq_length=256)
    assert len(dataset) == 4
    outputs = dataset[0]
    assert len(outputs["input_ids"][0].tolist()) == 256


@pytest.mark.unittest
def test_coref_dataset_subtoken_data_aggregation(raw_future_subtokens_source_string):
    split_input = raw_future_subtokens_source_string.split()
    token_data = list(range(len(split_input)))
    offset_mapping = bert_tokenizer(split_input, add_special_tokens=False, is_split_into_words=True,
                                    return_offsets_mapping=True)["offset_mapping"]
    subtoken_data = MangoesCoreferenceDataset.get_subtoken_data(token_data, offset_mapping)
    expected = [0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7]
    assert subtoken_data == expected


@pytest.mark.unittest
def test_coref_dataset_pad_values_function():
    values = [1, 1, 1, 1]
    padded_values = MangoesCoreferenceDataset.pad_list(values, target_length=6)
    expected = [1, 1, 1, 1, 0, 0]
    assert padded_values == expected


@pytest.mark.unittest
def test_coref_dataset_init(raw_coref_data):
    coref_documents, coref_cluster_ids, _, _ = raw_coref_data
    max_segment_len = 30
    dataset = MangoesCoreferenceDataset(bert_tokenizer, False, max_segment_len=max_segment_len, max_segments=5,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids)
    assert len(dataset) == 2
    for i in range(len(dataset.examples)):
        for j in range(len(dataset.examples[i][0])):
            assert len(dataset.examples[i][0][j]) == max_segment_len
            assert len(dataset.examples[i][1][j]) == max_segment_len

    assert np.array_equal(dataset.examples[0][3].numpy(), [1,  4,  7, 10, 13, 16, 25, 28, 31])
    assert np.array_equal(dataset.examples[0][4].numpy(), [1,  4,  7, 10, 13, 16, 25, 28, 31])
    assert np.array_equal(dataset.examples[1][3].numpy(), [1,  4,  7, 10, 13, 16])
    assert np.array_equal(dataset.examples[1][4].numpy(), [1,  4,  7, 10, 13, 16])


@pytest.mark.unittest
def test_coref_dataset_init_sentence_len_gt_max_segment_len(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 11    # shorter than longest sentence
    dataset = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=5,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids,
                                        speaker_ids=coref_speakers, genres=coref_genres)
    assert len(dataset) == 2
    assert len(dataset.examples[0][0]) == 4
    for i in range(len(dataset.examples)):
        for j in range(len(dataset.examples[i][0])):
            assert len(dataset.examples[i][0][j]) == max_segment_len
            assert len(dataset.examples[i][1][j]) == max_segment_len
    assert np.array_equal(dataset.examples[0][3].numpy(), [1,  4,  7, 12, 15, 18, 29, 32, 35])
    assert np.array_equal(dataset.examples[0][4].numpy(), [1,  4,  7, 12, 15, 18, 29, 32, 35])
    assert np.array_equal(dataset.examples[1][3].numpy(), [1,  4,  7, 12, 15, 18])
    assert np.array_equal(dataset.examples[1][4].numpy(), [1,  4,  7, 12, 15, 18])


@pytest.mark.unittest
def test_coref_dataset_init_genre_mapping(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, _ = raw_coref_data
    coref_genres = ["test1", "test2"]
    genre_mapping = {"test1": 0, "test2": 1}
    max_segment_len = 30
    dataset = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=5,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids,
                                        speaker_ids=coref_speakers, genres=coref_genres, genre_to_id=genre_mapping)
    assert len(dataset) == 2
    assert dataset.examples[0][-1] == 0
    assert dataset.examples[1][-1] == 1


@pytest.mark.unittest
def test_coref_dataset_init_with_tuple_clusterids(raw_coref_data):
    coref_documents, coref_cluster_ids, _, _ = raw_coref_data
    # change cluster ids to include tuples
    coref_cluster_ids = [[[(5, 12), -1, -1, 12, -1, 12, -1, -1],
                          [(14, 6), 6, -1, 14, -1, 14, -1, -1, -1, -1, -1, -1],
                          [14, -1, -1, 12, -1, 14, -1, -1]],
                         [[13, -1, -1, 13, -1, 13, -1, -1],
                          [15, -1, -1, 15, -1, 15, -1, -1, -1, -1, -1, -1]]]
    max_segment_len = 30
    dataset = MangoesCoreferenceDataset(bert_tokenizer, False, max_segment_len=max_segment_len, max_segments=5,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids)
    assert len(dataset) == 2
    for i in range(len(dataset.examples)):
        for j in range(len(dataset.examples[i][0])):
            assert len(dataset.examples[i][0][j]) == max_segment_len
            assert len(dataset.examples[i][1][j]) == max_segment_len

    assert np.array_equal(dataset.examples[0][3].numpy(), [1, 1, 4, 7, 10, 10, 13, 16, 25, 28, 31])
    assert np.array_equal(dataset.examples[0][4].numpy(), [1, 1, 4, 7, 10, 11, 13, 16, 25, 28, 31])
    assert np.array_equal(dataset.examples[0][5].numpy(), [5, 12, 12, 12, 14, 6, 14, 14, 14, 12, 14])


@pytest.mark.unittest
def test_coref_dataset_init_with_metadata(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 30
    dataset = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=5,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids,
                                        speaker_ids=coref_speakers, genres=coref_genres)
    assert len(dataset) == 2
    for i in range(len(dataset.examples)):
        assert dataset.examples[i][7] == coref_genres[i]
        for j in range(len(dataset.examples[i][0])):
            assert len(dataset.examples[i][6][j]) == max_segment_len


@pytest.mark.unittest
def test_coref_dataset_getitem(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 15
    dataset = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=2,
                                        documents=coref_documents, cluster_ids=coref_cluster_ids,
                                        speaker_ids=coref_speakers, genres=coref_genres)
    assert len(dataset) == 2
    assert len(dataset[0]["input_ids"]) == 2


@pytest.mark.unittest
def test_coref_dataset_init_with_metadata_error(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 30
    with pytest.raises(RuntimeError) as excinfo:
        _ = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=5,
                                      documents=coref_documents, cluster_ids=coref_cluster_ids)
    assert "use_metadata" in str(excinfo.value)


@pytest.mark.unittest
def test_coref_dataset_init_with_genre_error(raw_coref_data):
    coref_documents, coref_cluster_ids, coref_speakers, _ = raw_coref_data
    coref_genres = ["test1", "test2"]
    max_segment_len = 30
    with pytest.raises(RuntimeError) as excinfo:
        _ = MangoesCoreferenceDataset(bert_tokenizer, True, max_segment_len=max_segment_len, max_segments=5,
                                      documents=coref_documents, cluster_ids=coref_cluster_ids,
                                      speaker_ids=coref_speakers, genres=coref_genres)
    assert "genre_to_id" in str(excinfo.value)


