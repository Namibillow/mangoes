# -*- coding: utf-8 -*-

import logging

import pytest
from transformers import BertConfig, BertTokenizerFast
import torch
import numpy as np

from mangoes.modeling import BertForCoreferenceResolutionBase


logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
vocab_size = len(tokenizer.get_vocab())
config = BertConfig(vocab_size, hidden_size=120, num_hidden_layers=2)
model = BertForCoreferenceResolutionBase(config, max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                         use_metadata=True)


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_transformers_coref_forward_output_sizes(raw_small_source_string):
    inputs = tokenizer(raw_small_source_string, add_special_tokens=False, padding=True, return_tensors="pt")
    genre = torch.as_tensor([1])
    speaker_ids = torch.zeros_like(inputs["attention_mask"])
    sentence_map = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    output_dict = model.forward(inputs["input_ids"], inputs["attention_mask"], sentence_map, speaker_ids=speaker_ids,
                                genre=genre, return_dict=True)
    assert output_dict["candidate_starts"].size(0) == 9
    assert output_dict["candidate_starts"].size(0) == output_dict["candidate_ends"].size(0)
    assert output_dict["candidate_mention_scores"].size(0) == output_dict["candidate_starts"].size(0)
    assert output_dict["top_span_starts"].size(0) == 9
    assert output_dict["top_span_starts"].size(0) == output_dict["top_span_ends"].size(0)


@pytest.mark.unittest
def test_transformers_coref_loss_calculation(raw_small_source_string):
    inputs = tokenizer(raw_small_source_string, add_special_tokens=False, padding=True, return_tensors="pt")
    genre = torch.as_tensor([1])
    speaker_ids = torch.zeros_like(inputs["attention_mask"])
    sentence_map = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    gold_starts = torch.as_tensor([0, 3, 6])
    gold_ends = torch.as_tensor([0, 5, 6])
    cluster_ids = torch.as_tensor([0, 0, 0])
    output_dict = model.forward(inputs["input_ids"], inputs["attention_mask"], sentence_map, speaker_ids=speaker_ids,
                                genre=genre, gold_starts=gold_starts, gold_ends=gold_ends, cluster_ids=cluster_ids,
                                return_dict=True)

    assert output_dict["loss"].item() >= 0.0


@pytest.mark.unittest
def test_transformers_coref_save_load_full_model(raw_small_source_string, tmpdir_factory):
    model.eval()
    inputs = tokenizer(raw_small_source_string, add_special_tokens=False, padding=True, return_tensors="pt")
    genre = torch.as_tensor([1])
    speaker_ids = torch.zeros_like(inputs["attention_mask"])
    sentence_map = torch.as_tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])
    gold_starts = torch.as_tensor([0, 3, 6])
    gold_ends = torch.as_tensor([0, 5, 6])
    cluster_ids = torch.as_tensor([0, 0, 0])
    output_dict = model.forward(inputs["input_ids"], inputs["attention_mask"], sentence_map, speaker_ids=speaker_ids,
                                genre=genre, gold_starts=gold_starts, gold_ends=gold_ends, cluster_ids=cluster_ids,
                                return_dict=True)

    save_dir = tmpdir_factory.mktemp('model')
    model.save_pretrained(save_dir)

    new_model = BertForCoreferenceResolutionBase.from_pretrained(save_dir)
    new_model.eval()
    new_output_dict = new_model.forward(inputs["input_ids"], inputs["attention_mask"], sentence_map,
                                        speaker_ids=speaker_ids, genre=genre, gold_starts=gold_starts,
                                        gold_ends=gold_ends, cluster_ids=cluster_ids, return_dict=True)

    for key in output_dict.keys():
        assert np.array_equal(output_dict[key].detach().cpu().numpy(), new_output_dict[key].detach().cpu().numpy())



