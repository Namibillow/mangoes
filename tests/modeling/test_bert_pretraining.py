# -*- coding: utf-8 -*-
import logging

import pytest
from transformers import BertTokenizerFast
import numpy as np

from mangoes.modeling import BERTForPreTraining, BERTForMaskedLanguageModeling, BERTWordPieceTokenizer,\
    BERTForSequenceClassification


logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# ###########################################################################################
# ### Unit tests
@pytest.mark.unittest
def test_bert_pretraining_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForPreTraining(str(tok_path), )
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_pretraining_save_load(raw_small_source_string, tmpdir_factory):
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    outputs = mod.generate_outputs(raw_small_source_string)
    save_dir = tmpdir_factory.mktemp('data')
    mod.save(save_dir, save_tokenizer=False)

    loaded_model = BERTForPreTraining.load("bert-base-uncased", save_dir)
    outputs2 = loaded_model.generate_outputs(raw_small_source_string)
    assert np.array_equal(outputs["seq_relationship_logits"].numpy(), outputs2["seq_relationship_logits"].numpy())
    assert np.array_equal(outputs["prediction_logits"].numpy(), outputs2["prediction_logits"].numpy())


@pytest.mark.unittest
def test_bert_pretraining_predict_textpair_batch():
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    predictions = mod.predict([f"I {tokenizer.mask_token} getting up early", f"I was getting up early"],
                              [f"Next sentence", f"Next {tokenizer.mask_token}"], top_k=1)
    assert "token_str" in predictions[0][0]
    assert "score" in predictions[0][0]
    assert "sequence" in predictions[0][0]
    assert "token_str" in predictions[0][0]
    assert len(predictions[1]) == 2


@pytest.mark.unittest
def test_bert_pretraining_predict_textpair_single():
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    predictions = mod.predict(f"I {tokenizer.mask_token} getting up early", f"Next sentence", top_k=1)
    assert "token_str" in predictions[0][0]
    assert "score" in predictions[0][0]
    assert "sequence" in predictions[0][0]
    assert "token_str" in predictions[0][0]
    assert len(predictions[1]) == 1


@pytest.mark.unittest
def test_bert_pretraining_predict_batch():
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    predictions = mod.predict([f"I {tokenizer.mask_token} getting up early", f"Next {tokenizer.mask_token}"], top_k=1)
    assert "token_str" in predictions[0][0]
    assert "score" in predictions[0][0]
    assert "sequence" in predictions[0][0]
    assert "token_str" in predictions[0][0]
    assert len(predictions[1]) == 2


@pytest.mark.unittest
def test_bert_pretraining_predict_single():
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    predictions = mod.predict(f"I {tokenizer.mask_token} getting up early", top_k=1)
    assert "token_str" in predictions[0][0]
    assert "score" in predictions[0][0]
    assert "sequence" in predictions[0][0]
    assert "token_str" in predictions[0][0]
    assert len(predictions[1]) == 1


@pytest.mark.unittest
def test_bert_pretraining_outputs_textpair(raw_small_source_string):
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    outputs = mod.generate_outputs(raw_small_source_string[0], text_pairs=raw_small_source_string[1],
                                   pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    # max sequence length is 8 for this input
    assert tuple(outputs["prediction_logits"].shape) == (1, 12, mod.model.config.vocab_size)
    assert tuple(outputs["seq_relationship_logits"].shape) == (1, 2)
    assert len(outputs["attentions"]) == mod.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (1, mod.model.config.num_attention_heads, 12, 12)
    assert len(outputs["hidden_states"]) == mod.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (1, 12, mod.model.config.hidden_size)
    assert tuple(outputs["offset_mappings"].shape) == (1, 12, 2)


@pytest.mark.unittest
def test_bert_pretraining_outputs_single(raw_small_source_string):
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    outputs = mod.generate_outputs(raw_small_source_string, pre_tokenized=False, output_hidden_states=True,
                                   output_attentions=True)
    # max sequence length is 8 for this input
    assert tuple(outputs["prediction_logits"].shape) == (2, 8, mod.model.config.vocab_size)
    assert tuple(outputs["seq_relationship_logits"].shape) == (2, 2)
    assert len(outputs["attentions"]) == mod.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, mod.model.config.num_attention_heads, 8, 8)
    assert len(outputs["hidden_states"]) == mod.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 8, mod.model.config.hidden_size)
    assert tuple(outputs["offset_mappings"].shape) == (2, 8, 2)


@pytest.mark.unittest
def test_bert_pretraining_train(raw_source_file, extra_small_source_file, tmpdir_factory):
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    mod.train(train_text=raw_source_file, eval_text=extra_small_source_file, output_dir=output_dir,
              num_train_epochs=1, )
    assert not mod.model.training


@pytest.mark.unittest
def test_bert_maskedlm_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForMaskedLanguageModeling(str(tok_path), hidden_size=132, intermediate_size=256, num_hidden_layers=1)
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_maskedlm_save_load(raw_small_source_string, raw_source_file, tmpdir_factory):
    maskedlm_model = BERTForMaskedLanguageModeling("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                                   num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    maskedlm_model.train(train_text=raw_source_file, output_dir=output_dir, num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    maskedlm_model.save(save_dir, save_tokenizer=False)
    outputs = maskedlm_model.generate_outputs(raw_small_source_string)

    loaded_model = BERTForMaskedLanguageModeling.load("bert-base-uncased", save_dir)
    outputs2 = loaded_model.generate_outputs(raw_small_source_string)
    assert np.array_equal(outputs["logits"].numpy(), outputs2["logits"].numpy(),)


@pytest.mark.unittest
def test_bert_maskedlm_outputs(raw_small_source_string):
    pretrained_maskedlm_model = BERTForMaskedLanguageModeling("bert-base-uncased", hidden_size=132,
                                                              intermediate_size=256, num_hidden_layers=1)
    outputs = pretrained_maskedlm_model.generate_outputs(raw_small_source_string, pre_tokenized=False,
                                                         output_hidden_states=True,
                                                         output_attentions=True)
    # max sequence length is 8 for this input
    assert tuple(outputs["logits"].shape) == (2, 8, pretrained_maskedlm_model.model.config.vocab_size)
    assert len(outputs["attentions"]) == pretrained_maskedlm_model.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, pretrained_maskedlm_model.model.config.num_attention_heads, 8, 8)
    assert len(outputs["hidden_states"]) == pretrained_maskedlm_model.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 8, pretrained_maskedlm_model.model.config.hidden_size)


@pytest.mark.unittest
def test_bert_maskedlm_predict():
    mod = BERTForMaskedLanguageModeling.load("bert-base-uncased", "bert-base-uncased")
    predictions = mod.predict(f"I {tokenizer.mask_token} getting up early", top_k=1)
    assert predictions[0]["token_str"] == "was"


@pytest.mark.unittest
def test_bert_maskedlm_train_iterable_dataset(raw_source_file, extra_small_source_file, tmpdir_factory):
    mod = BERTForMaskedLanguageModeling("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                        num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    mod.train(train_text=raw_source_file, eval_text=extra_small_source_file, output_dir=output_dir,
              num_train_epochs=1, )
    assert not mod.model.training


@pytest.mark.unittest
def test_bert_maskedlm_train_linebyline_dataset(raw_source_file, extra_small_source_file, tmpdir_factory):
    mod = BERTForMaskedLanguageModeling("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                        num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    mod.train(train_text=raw_source_file, eval_text=extra_small_source_file, iterable_dataset=False,
              output_dir=output_dir, num_train_epochs=1, )
    assert not mod.model.training


# #########################################################################################ds##
# ### Integration tests
@pytest.mark.integration
def test_pretrain_fine_tune(raw_source_file, raw_small_source_string, tmpdir_factory):
    mod = BERTForPreTraining("bert-base-uncased", hidden_size=132, num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    mod.train(train_text=raw_source_file, output_dir=output_dir, num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    mod.save(str(save_dir), save_tokenizer=True)

    loaded_model = BERTForSequenceClassification.load(str(save_dir), str(save_dir), labels=["pos", "neg"])
    loaded_model.train(train_text=raw_small_source_string, train_targets=[0, 1], output_dir=output_dir,
                       num_train_epochs=1, )
    assert not loaded_model.model.training


@pytest.mark.integration
def test_lm_fine_tune(raw_source_file, raw_small_source_string, tmpdir_factory):
    mod = BERTForMaskedLanguageModeling("bert-base-uncased", hidden_size=132, num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    mod.train(train_text=raw_source_file, output_dir=output_dir, num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    mod.save(str(save_dir), save_tokenizer=True)

    loaded_model = BERTForSequenceClassification.load(str(save_dir), str(save_dir), labels=["pos", "neg"])
    loaded_model.train(train_text=raw_small_source_string, train_targets=[0, 1], output_dir=output_dir,
                       num_train_epochs=1, )
    assert not loaded_model.model.training
