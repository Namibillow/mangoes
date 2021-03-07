# -*- coding: utf-8 -*-
import logging

import pytest
from transformers import AutoModel, BertTokenizerFast, pipeline
import numpy as np

from mangoes.modeling import merge_subword_embeddings, PretrainedTransformerModelForFeatureExtraction


logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
extraction_pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer)
pretrained_model = PretrainedTransformerModelForFeatureExtraction.load("bert-base-uncased", "bert-base-uncased",
                                                                       device=-1)


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_nonbert_transformer(raw_small_source_string):
    distilbert_model = PretrainedTransformerModelForFeatureExtraction.\
        load("distilbert-base-uncased", "distilbert-base-uncased")
    outputs = distilbert_model.generate_outputs(raw_small_source_string, output_hidden_states=True)
    assert outputs["offset_mappings"].cpu().numpy().shape == (2, 8, 2)
    assert outputs["hidden_states"][-1].cpu().numpy().shape == (2, 8, 768)


@pytest.mark.unittest
def test_bert_feature_extraction_predict(raw_small_source_string):
    embeddings = np.squeeze(pretrained_model.predict(raw_small_source_string))
    assert np.array_equal(np.squeeze(extraction_pipeline(raw_small_source_string)), embeddings)


@pytest.mark.unittest
def test_bert_feature_extraction_raw(raw_small_source_string):
    embeddings = pretrained_model.generate_outputs(raw_small_source_string, output_hidden_states=True)["hidden_states"]\
        [-1].cpu().numpy()
    assert np.array_equal(np.squeeze(extraction_pipeline(raw_small_source_string)), embeddings)


@pytest.mark.unittest
def test_bert_feature_extraction_sentences(raw_sentences_small, raw_small_source_string):
    embeddings = pretrained_model.generate_outputs(raw_sentences_small, pre_tokenized=True,
                                                   output_hidden_states=True)["hidden_states"][-1].cpu().numpy()
    assert np.array_equal(np.squeeze(extraction_pipeline(raw_small_source_string)), embeddings)


@pytest.mark.unittest
def test_bert_feature_extraction_word_embeddings_raw(raw_small_source_string):
    embeddings = pretrained_model.generate_outputs(raw_small_source_string, output_hidden_states=True,
                                                   word_embeddings=True)["hidden_states"][-1].cpu().numpy()
    # construct target word embeddings
    raw_outputs = np.squeeze(extraction_pipeline(raw_small_source_string)).astype(embeddings.dtype)
    outputs = np.zeros((2, 4, 768))
    outputs[0][0] = np.mean(raw_outputs[0][1:4][:], axis=0)     # "I'm" is split into 3 sub-words
    outputs[0][1] = raw_outputs[0][4][:]
    outputs[0][2] = raw_outputs[0][5][:]
    outputs[0][3] = raw_outputs[0][6][:]
    outputs[1][0] = raw_outputs[1][1][:]
    outputs[1][1] = raw_outputs[1][2][:]
    outputs[1][2] = raw_outputs[1][3][:]

    assert np.array_equal(outputs, embeddings)


@pytest.mark.unittest
def test_bert_feature_extraction_word_embeddings_sentences(raw_sentences_small, raw_small_source_string):
    embeddings = pretrained_model.generate_outputs(raw_sentences_small, pre_tokenized=True, output_hidden_states=True,
                                                   word_embeddings=True)["hidden_states"][-1].cpu().numpy()
    # construct target word embeddings
    raw_outputs = np.squeeze(extraction_pipeline(raw_small_source_string)).astype(embeddings.dtype)
    outputs = np.zeros((2, 4, 768))
    outputs[0][0] = np.mean(raw_outputs[0][1:4][:], axis=0)     # "I'm" is split into 3 sub-words
    outputs[0][1] = raw_outputs[0][4][:]
    outputs[0][2] = raw_outputs[0][5][:]
    outputs[0][3] = raw_outputs[0][6][:]
    outputs[1][0] = raw_outputs[1][1][:]
    outputs[1][1] = raw_outputs[1][2][:]
    outputs[1][2] = raw_outputs[1][3][:]

    assert np.array_equal(outputs, embeddings)


@pytest.mark.unittest
def test_feature_extraction_subword_merging_raw(raw_small_source_string):
    raw_outputs = np.squeeze(extraction_pipeline(raw_small_source_string))
    offset_mapping = tokenizer(raw_small_source_string, is_split_into_words=False, return_offsets_mapping=True,
                               padding=True, return_tensors='pt').pop('offset_mapping').numpy()
    word_embeddings = merge_subword_embeddings(raw_outputs, raw_small_source_string, offset_mapping)
    # construct target word embeddings
    outputs = np.zeros((2, 4, 768))
    outputs[0][0] = np.mean(raw_outputs[0][1:4][:], 0)     # "I'm" is split into 3 sub-words
    outputs[0][1] = raw_outputs[0][4][:]
    outputs[0][2] = raw_outputs[0][5][:]
    outputs[0][3] = raw_outputs[0][6][:]
    outputs[1][0] = raw_outputs[1][1][:]
    outputs[1][1] = raw_outputs[1][2][:]
    outputs[1][2] = raw_outputs[1][3][:]

    assert np.array_equal(outputs, word_embeddings)


@pytest.mark.unittest
def test_feature_extraction_subword_merging_sentences(raw_sentences_small, raw_small_source_string):
    raw_outputs = np.squeeze(extraction_pipeline(raw_small_source_string))
    offset_mapping = tokenizer(raw_sentences_small, is_split_into_words=True, return_offsets_mapping=True,
                               padding=True, return_tensors='pt').pop('offset_mapping').numpy()
    word_embeddings = merge_subword_embeddings(raw_outputs, raw_sentences_small, offset_mapping, pretokenized=True)

    # construct target word embeddings
    outputs = np.zeros((2, 4, 768))
    outputs[0][0] = np.mean(raw_outputs[0][1:4][:], 0)     # "I'm" is split into 3 sub-words
    outputs[0][1] = raw_outputs[0][4][:]
    outputs[0][2] = raw_outputs[0][5][:]
    outputs[0][3] = raw_outputs[0][6][:]
    outputs[1][0] = raw_outputs[1][1][:]
    outputs[1][1] = raw_outputs[1][2][:]
    outputs[1][2] = raw_outputs[1][3][:]

    assert np.array_equal(outputs, word_embeddings)

