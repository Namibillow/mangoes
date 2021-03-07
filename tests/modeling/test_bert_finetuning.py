# -*- coding: utf-8 -*-
import logging
import warnings

import pytest
from transformers import SquadExample, BertModel, BertConfig, TrainingArguments
import numpy as np

from mangoes.modeling import BERTForSequenceClassification, BERTForTokenClassification, BERTWordPieceTokenizer, \
    BERTForQuestionAnswering, BERTForCoreferenceResolution, MangoesCoreferenceDataset, CoreferenceFineTuneTrainer, \
    BERTForMultipleChoice, MangoesQuestionAnsweringDataset, MangoesMultipleChoiceDataset


logging_level = logging.WARNING
logging_format = '%(asctime)s :: %(name)s::%(funcName)s() :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging_level, format=logging_format)  # , filename="report.log")
logger = logging.getLogger(__name__)


@pytest.mark.unittest
def test_bert_seqclassifier_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForSequenceClassification(str(tok_path), labels=["pos", "neg"])
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_seqclassifier_save_load(raw_small_source_string, tmpdir_factory):
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    seq_classifier_model.train(train_text=raw_small_source_string, train_targets=[0, 1], output_dir=output_dir,
                               num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    seq_classifier_model.save(save_dir, save_tokenizer=False)
    outputs = seq_classifier_model.generate_outputs(raw_small_source_string)

    loaded_model = BERTForSequenceClassification.load("bert-base-uncased", save_dir, labels=["pos", "neg"])
    outputs2 = loaded_model.generate_outputs(raw_small_source_string)
    assert np.array_equal(outputs["logits"].numpy(), outputs2["logits"].numpy(),)


@pytest.mark.unittest
def test_bert_seqclassifier_outputs(raw_small_source_string):
    pretrained_seqclassifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                                   hidden_size=132, num_hidden_layers=1,
                                                                   intermediate_size=256)
    outputs = pretrained_seqclassifier_model.generate_outputs(raw_small_source_string, pre_tokenized=False,
                                                              output_hidden_states=True,
                                                              output_attentions=True)
    # max sequence length is 8 for this input
    assert tuple(outputs["logits"].shape) == (2, 2)
    assert len(outputs["attentions"]) == pretrained_seqclassifier_model.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, pretrained_seqclassifier_model.model.config.num_attention_heads, 8, 8)
    assert len(outputs["hidden_states"]) == pretrained_seqclassifier_model.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 8, pretrained_seqclassifier_model.model.config.hidden_size)


@pytest.mark.unittest
def test_bert_seqclassifier_predict():
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1)
    predictions = seq_classifier_model.predict("this is a test sentence")
    assert 0.0 <= predictions[0]["score"] <= 1.0


@pytest.mark.unittest
def test_bert_seqclassifier_train(raw_small_source_string, tmpdir_factory):
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('data')
    seq_classifier_model.train(train_text=raw_small_source_string, train_targets=["pos", "neg"],
                               output_dir=output_dir, num_train_epochs=1, )
    assert not seq_classifier_model.model.training


@pytest.mark.unittest
def test_bert_seqclassifier_train_freezebase(raw_small_source_string, tmpdir_factory):
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('data')
    seq_classifier_model.train(train_text=raw_small_source_string, train_targets=["pos", "neg"],
                               output_dir=output_dir, num_train_epochs=1, freeze_base=True)
    assert not seq_classifier_model.model.training
    for param in seq_classifier_model.model.base_model.parameters():
        assert not param.requires_grad


@pytest.mark.unittest
def test_bert_seqclassifier_train_multiplelearnrate(raw_small_source_string, tmpdir_factory):
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('data')
    seq_classifier_model.train(train_text=raw_small_source_string, train_targets=["pos", "neg"],
                               output_dir=output_dir, num_train_epochs=1, task_learn_rate=0.001)
    assert not seq_classifier_model.model.training


@pytest.mark.unittest
def test_bert_seqclassifier_train_with_eval(raw_small_source_string, tmpdir_factory):
    seq_classifier_model = BERTForSequenceClassification("bert-base-uncased", labels=["pos", "neg"],
                                                         hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('data')
    seq_classifier_model.train(train_text=raw_small_source_string, train_targets=["pos", "neg"],
                               eval_text=raw_small_source_string, eval_targets=["pos", "neg"],
                               output_dir=output_dir, num_train_epochs=1, )
    assert not seq_classifier_model.model.training


@pytest.mark.unittest
def test_bert_tokenclassifier_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForTokenClassification(str(tok_path), labels=["pos", "neg"],)
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_tokenclassifier_outputs(raw_small_source_string):
    pretrained_mod = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                num_hidden_layers=1, intermediate_size=256)
    outputs = pretrained_mod.generate_outputs(raw_small_source_string, pre_tokenized=False, output_hidden_states=True,
                                              output_attentions=True)
    # max sequence length is 8 for this input
    assert tuple(outputs["logits"].shape) == (2, 8, 2)
    assert len(outputs["attentions"]) == pretrained_mod.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, pretrained_mod.model.config.num_attention_heads, 8, 8)
    assert len(outputs["hidden_states"]) == pretrained_mod.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 8, pretrained_mod.model.config.hidden_size)


@pytest.mark.unittest
def test_bert_tokenclassifier_train(raw_source_string, tmpdir_factory):
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                        num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    token_classifier_model.train(train_text=[raw_source_string[0]], train_targets=[0, 1, 0, 1, 0, 1, 0],
                                 output_dir=output_dir, num_train_epochs=1, )
    assert not token_classifier_model.model.training


@pytest.mark.unittest
def test_bert_tokenclassifier_train_freezebase(raw_source_string, tmpdir_factory):
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                        num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    token_classifier_model.train(train_text=[raw_source_string[0]], train_targets=[0, 1, 0, 1, 0, 1, 0],
                                 output_dir=output_dir, num_train_epochs=1, freeze_base=True)
    assert not token_classifier_model.model.training
    for param in token_classifier_model.model.base_model.parameters():
        assert not param.requires_grad


@pytest.mark.unittest
def test_bert_tokenclassifier_train_multiplelearnrate(raw_source_string, tmpdir_factory):
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                        num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    token_classifier_model.train(train_text=[raw_source_string[0]], train_targets=[0, 1, 0, 1, 0, 1, 0],
                                 output_dir=output_dir, num_train_epochs=1, task_learn_rate=0.001)
    assert not token_classifier_model.model.training


@pytest.mark.unittest
def test_bert_tokenclassifier_train_with_eval(raw_source_string, tmpdir_factory):
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                        num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    token_classifier_model.train(train_text=[raw_source_string[0]], train_targets=[0, 1, 0, 1, 0, 1, 0],
                                 eval_text=[raw_source_string[0]], eval_targets=[0, 1, 0, 1, 0, 1, 0],
                                 output_dir=output_dir, num_train_epochs=1, )
    assert not token_classifier_model.model.training


@pytest.mark.unittest
def test_bert_tokenclassifier_predict():
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"], hidden_size=132,
                                                        num_hidden_layers=1, intermediate_size=256)
    predictions = token_classifier_model.predict("this is a test sentence")
    assert 0.0 <= predictions[0][0]["score"] <= 1.0
    assert predictions[0][0]["word"] == "this"
    assert predictions[0][0]["entity"] in ["pos", "neg"]


@pytest.mark.unittest
def test_bert_tokenclassifier_save_load(raw_source_string, tmpdir_factory):
    token_classifier_model = BERTForTokenClassification("bert-base-uncased", labels=["pos", "neg"],
                                                        hidden_size=132, num_hidden_layers=1, intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    token_classifier_model.train(train_text=[raw_source_string[0]], train_targets=[0, 1, 0, 1, 0, 1, 0],
                                 output_dir=output_dir, num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    token_classifier_model.save(save_dir, save_tokenizer=False)
    outputs = token_classifier_model.generate_outputs(raw_source_string)

    loaded_model = BERTForTokenClassification.load("bert-base-uncased", save_dir, labels=["pos", "neg"])
    outputs2 = loaded_model.generate_outputs(raw_source_string)
    assert np.array_equal(outputs["logits"].numpy(), outputs2["logits"].numpy(),)


@pytest.mark.unittest
def test_bert_questionanswer_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForQuestionAnswering(str(tok_path))
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_questionanswer_outputs(question_answering_data):
    pretrained_mod = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                              intermediate_size=256)
    outputs = pretrained_mod.generate_outputs(question=question_answering_data[0], context=question_answering_data[1],
                                              pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    # max sequence length is 19 for these inputs (first question tokens + first context tokens + 3 special tokens)
    assert tuple(outputs["start_logits"].shape) == (2, 19)
    assert tuple(outputs["end_logits"].shape) == (2, 19)
    assert len(outputs["attentions"]) == pretrained_mod.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, pretrained_mod.model.config.num_attention_heads, 19, 19)
    assert len(outputs["hidden_states"]) == pretrained_mod.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 19, pretrained_mod.model.config.hidden_size)
    assert tuple(outputs["offset_mappings"].shape) == (2, 19, 2)


@pytest.mark.unittest
def test_bert_questionanswer_outputs_shared_context(question_answering_data):
    pretrained_mod = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                              intermediate_size=256)
    # test using same context for both questions
    outputs = pretrained_mod.generate_outputs(question=question_answering_data[0],
                                              context=question_answering_data[1][0],
                                              pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    # max sequence length is 19 for these inputs (first question tokens + first context tokens + 3 special tokens)
    assert tuple(outputs["start_logits"].shape) == (2, 19)
    assert tuple(outputs["end_logits"].shape) == (2, 19)


@pytest.mark.unittest
def test_bert_questionanswer_outputs_single_example(question_answering_data):
    pretrained_mod = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                              intermediate_size=256)
    # test using only one question/context
    outputs = pretrained_mod.generate_outputs(question=question_answering_data[0][0],
                                              context=question_answering_data[1][0],
                                              pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    # max sequence length is 19 for these inputs (first question tokens + first context tokens + 3 special tokens)
    assert tuple(outputs["start_logits"].shape) == (1, 19)
    assert tuple(outputs["end_logits"].shape) == (1, 19)


@pytest.mark.unittest
def test_bert_questionanswer_outputs_input_error(question_answering_data):
    pretrained_mod = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                              intermediate_size=256)
    # test using only one question/context
    with pytest.raises(RuntimeError) as excinfo:
        pretrained_mod.generate_outputs(question=question_answering_data[0][:1],
                                        context=question_answering_data[1],
                                        pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    assert "don't have the same lengths" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_questionanswer_predict_strings(question_answering_data):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    predictions = question_answer_model.predict(question=question_answering_data[0][0],
                                                context=question_answering_data[1][0])
    assert 0.0 <= predictions[0]["score"] <= 1.0
    assert predictions[0]["start"] < len(question_answering_data[1][0])
    assert predictions[0]["end"] < len(question_answering_data[1][0])


@pytest.mark.unittest
def test_bert_questionanswer_predict_squadexample(question_answering_data):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    example = SquadExample(qas_id=0, question_text=question_answering_data[0][0],
                           context_text=question_answering_data[1][0], start_position_character=0,
                           title="", answer_text=question_answering_data[2][0])
    predictions = question_answer_model.predict(example)
    assert 0.0 <= predictions[0]["score"] <= 1.0
    assert predictions[0]["start"] < len(question_answering_data[1][0])
    assert predictions[0]["end"] < len(question_answering_data[1][0])


@pytest.mark.unittest
def test_bert_questionanswer_train(question_answering_data, tmpdir_factory):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    question_answer_model.train(train_question_texts=question_answering_data[0],
                                train_context_texts=question_answering_data[1],
                                train_answer_texts=question_answering_data[2],
                                train_start_indices=question_answering_data[3],
                                output_dir=output_dir, num_train_epochs=1)
    assert not question_answer_model.model.training


@pytest.mark.unittest
def test_bert_questionanswer_train(question_answering_data, tmpdir_factory):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    question_answer_model.train(train_question_texts=question_answering_data[0],
                                train_context_texts=question_answering_data[1],
                                train_answer_texts=question_answering_data[2],
                                train_start_indices=question_answering_data[3],
                                output_dir=output_dir, num_train_epochs=1, freeze_base=True)
    assert not question_answer_model.model.training
    for param in question_answer_model.model.base_model.parameters():
        assert not param.requires_grad


@pytest.mark.unittest
def test_bert_questionanswer_train_multiplelearnrate(question_answering_data, tmpdir_factory):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    question_answer_model.train(train_question_texts=question_answering_data[0],
                                train_context_texts=question_answering_data[1],
                                train_answer_texts=question_answering_data[2],
                                train_start_indices=question_answering_data[3],
                                output_dir=output_dir, num_train_epochs=1, task_learn_rate=0.001)
    assert not question_answer_model.model.training


@pytest.mark.unittest
def test_bert_questionanswer_train_with_eval(question_answering_data, tmpdir_factory):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    question_answer_model.train(train_question_texts=question_answering_data[0],
                                train_context_texts=question_answering_data[1],
                                train_answer_texts=question_answering_data[2],
                                train_start_indices=question_answering_data[3],
                                eval_question_texts=question_answering_data[0],
                                eval_context_texts=question_answering_data[1],
                                eval_answer_texts=question_answering_data[2],
                                eval_start_indices=question_answering_data[3],
                                output_dir=output_dir, num_train_epochs=1)
    assert not question_answer_model.model.training


@pytest.mark.unittest
def test_bert_questionanswer_save_load(question_answering_data, tmpdir_factory):
    question_answer_model = BERTForQuestionAnswering("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                     intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    question_answer_model.train(train_question_texts=question_answering_data[0],
                                train_context_texts=question_answering_data[1],
                                train_answer_texts=question_answering_data[2],
                                train_start_indices=question_answering_data[3],
                                output_dir=output_dir, num_train_epochs=1, )
    save_dir = tmpdir_factory.mktemp('data')
    question_answer_model.save(save_dir, save_tokenizer=False)
    outputs = question_answer_model.generate_outputs(question=question_answering_data[0],
                                                     context=question_answering_data[1],
                                                     pre_tokenized=False, output_hidden_states=False,
                                                     output_attentions=False)
    loaded_model = BERTForQuestionAnswering.load("bert-base-uncased", save_dir)
    outputs2 = loaded_model.generate_outputs(question=question_answering_data[0],
                                             context=question_answering_data[1],
                                             pre_tokenized=False, output_hidden_states=False,
                                             output_attentions=False)
    assert np.array_equal(outputs["start_logits"].numpy(), outputs2["start_logits"].numpy(),)
    assert np.array_equal(outputs["end_logits"].numpy(), outputs2["end_logits"].numpy(),)


@pytest.mark.unittest
def test_bert_multiplechoice_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForMultipleChoice(str(tok_path))
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_multiplechoice_outputs_input_error(multiple_choice_example):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                                  num_hidden_layers=1)
    # test using only one question/context
    with pytest.raises(RuntimeError) as excinfo:
        multiple_choice_model.generate_outputs(questions=[multiple_choice_example[0]] * 2,
                                               choices=[multiple_choice_example[1]],
                                               pre_tokenized=False, output_hidden_states=True, output_attentions=True)
    assert "Number of questions does not match number of sets of choices" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_multiplechoice_predict(multiple_choice_example):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    predictions = multiple_choice_model.predict(questions=multiple_choice_example[0],
                                                choices=multiple_choice_example[1])
    assert 0.0 <= predictions[0]["score"] <= 1.0


@pytest.mark.unittest
def test_bert_multiplechoice_outputs(multiple_choice_example):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    outputs = multiple_choice_model.generate_outputs(questions=multiple_choice_example[0],
                                                     choices=multiple_choice_example[1],
                                                     pre_tokenized=False, output_hidden_states=True,
                                                     output_attentions=True)
    # max sequence length is 35 for these inputs
    assert tuple(outputs["logits"].shape) == (1, 2)
    assert len(outputs["attentions"]) == multiple_choice_model.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (1, 2, multiple_choice_model.model.config.num_attention_heads, 35,
                                                     35)
    assert len(outputs["hidden_states"]) == multiple_choice_model.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (1, 2, 35, multiple_choice_model.model.config.hidden_size)
    assert tuple(outputs["offset_mappings"].shape) == (1, 2, 35, 2)


@pytest.mark.unittest
def test_bert_multiplechoice_outputs_batch(multiple_choice_example):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    outputs = multiple_choice_model.generate_outputs(questions=[multiple_choice_example[0]] * 2,
                                                     choices=[multiple_choice_example[1]] * 2,
                                                     pre_tokenized=False, output_hidden_states=True,
                                                     output_attentions=True)
    # max sequence length is 35 for these inputs
    assert tuple(outputs["logits"].shape) == (2, 2)
    assert len(outputs["attentions"]) == multiple_choice_model.model.config.num_hidden_layers
    assert tuple(outputs["attentions"][0].shape) == (2, 2, multiple_choice_model.model.config.num_attention_heads, 35,
                                                     35)
    assert len(outputs["hidden_states"]) == multiple_choice_model.model.config.num_hidden_layers + 1
    assert tuple(outputs["hidden_states"][0].shape) == (2, 2, 35, multiple_choice_model.model.config.hidden_size)
    assert tuple(outputs["offset_mappings"].shape) == (2, 2, 35, 2)


@pytest.mark.unittest
def test_bert_multiplechoice_train(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 5,
                                train_choices_texts=[multiple_choice_example[1]] * 5,
                                train_labels=[0] * 5,
                                per_device_train_batch_size=2,
                                output_dir=output_dir, num_train_epochs=1)
    assert not multiple_choice_model.model.training


@pytest.mark.unittest
def test_bert_multiplechoice_train_freezebase(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 5,
                                train_choices_texts=[multiple_choice_example[1]] * 5,
                                train_labels=[0] * 5,
                                per_device_train_batch_size=2,
                                output_dir=output_dir, num_train_epochs=1, freeze_base=True)
    assert not multiple_choice_model.model.training
    for param in multiple_choice_model.model.base_model.parameters():
        assert not param.requires_grad


@pytest.mark.unittest
def test_bert_multiplechoice_train_with_validation(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 5,
                                eval_question_texts=[multiple_choice_example[0]] * 5,
                                train_choices_texts=[multiple_choice_example[1]] * 5,
                                eval_choices_texts=[multiple_choice_example[1]] * 5,
                                train_labels=[0] * 5,
                                eval_labels=[0] * 5,
                                per_device_train_batch_size=2,
                                output_dir=output_dir, num_train_epochs=1)
    assert not multiple_choice_model.model.training


@pytest.mark.unittest
def test_bert_multiplechoice_train_multiple_learnrate(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 5,
                                train_choices_texts=[multiple_choice_example[1]] * 5,
                                train_labels=[0] * 5,
                                per_device_train_batch_size=2,
                                output_dir=output_dir, num_train_epochs=1, task_learn_rate=0.001)
    assert not multiple_choice_model.model.training


@pytest.mark.unittest
def test_bert_multiplechoice_train_dir_error(multiple_choice_example):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                                  num_hidden_layers=1)
    # test using only one question/context
    with pytest.raises(RuntimeError) as excinfo:
        multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 2,
                                    train_choices_texts=[multiple_choice_example[1]])
    assert "output directory" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_multiplechoice_train_label_error(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, intermediate_size=256,
                                                  num_hidden_layers=1)
    output_dir = tmpdir_factory.mktemp('model')
    # test using only one question/context
    with pytest.raises(RuntimeError) as excinfo:
        multiple_choice_model.train(output_dir, train_question_texts=[multiple_choice_example[0]] * 2,
                                    train_choices_texts=[multiple_choice_example[1]] * 2)
    assert "Incomplete training data" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_multiplechoice_save_load(multiple_choice_example, tmpdir_factory):
    multiple_choice_model = BERTForMultipleChoice("bert-base-uncased", hidden_size=132, num_hidden_layers=1,
                                                  intermediate_size=256)
    output_dir = tmpdir_factory.mktemp('model')
    multiple_choice_model.train(train_question_texts=[multiple_choice_example[0]] * 5,
                                train_choices_texts=[multiple_choice_example[1]] * 5,
                                train_labels=[0] * 5,
                                per_device_train_batch_size=2,
                                output_dir=output_dir, num_train_epochs=1)
    save_dir = tmpdir_factory.mktemp('data')
    multiple_choice_model.save(save_dir, save_tokenizer=False)
    outputs = multiple_choice_model.generate_outputs(questions=[multiple_choice_example[0]] * 2,
                                                     choices=[multiple_choice_example[1]] * 2,
                                                     pre_tokenized=False, output_hidden_states=False,
                                                     output_attentions=False)
    loaded_model = BERTForMultipleChoice.load("bert-base-uncased", save_dir)
    outputs2 = loaded_model.generate_outputs(questions=[multiple_choice_example[0]] * 2,
                                             choices=[multiple_choice_example[1]] * 2,
                                             pre_tokenized=False, output_hidden_states=False,
                                             output_attentions=False)
    assert np.array_equal(outputs["logits"].numpy(), outputs2["logits"].numpy(),)


@pytest.mark.unittest
def test_bert_coref_init_saved_tokenizer(raw_source_file, tmpdir_factory):
    tok = BERTWordPieceTokenizer()
    tok.train(raw_source_file)
    tok_path = tmpdir_factory.mktemp('data')
    tok.save_model(str(tok_path))
    mod = BERTForCoreferenceResolution(str(tok_path), max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    assert mod.model.config.vocab_size == len(tok.get_vocab())


@pytest.mark.unittest
def test_bert_coref_outputs_metadata_error(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, _, _ = raw_coref_data
    with pytest.raises(RuntimeError) as excinfo:
        mod.generate_outputs(coref_documents[0], pre_tokenized=True)
    assert "use metadata" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_coref_outputs_metadata_genre_mapping_error(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    with pytest.raises(RuntimeError) as excinfo:
        mod.generate_outputs(coref_documents[0], pre_tokenized=True, speaker_ids=coref_speakers[0], genre="unknown")
    assert "genre id mapping" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_coref_outputs_metadata_genre_type_error(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    with pytest.raises(RuntimeError) as excinfo:
        mod.generate_outputs(coref_documents[0], pre_tokenized=True, speaker_ids=coref_speakers[0], genre=[1])
    assert "string or int" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_coref_outputs_metadata_speakerid_error(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    with pytest.raises(RuntimeError) as excinfo:
        mod.generate_outputs(coref_documents[0], pre_tokenized=True, speaker_ids=coref_speakers[0][:2], genre=1)
    assert "Lengths of speaker ids and text arguments are different" in str(excinfo.value)


@pytest.mark.unittest
def test_bert_coref_outputs_max_num_segments(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    with pytest.warns(RuntimeWarning):
        mod.generate_outputs(coref_documents[0], pre_tokenized=True, speaker_ids=coref_speakers[0], genre=1,
                             max_segment_len=12, max_segments=1)


@pytest.mark.unittest
def test_bert_coref_outputs_pretokenized(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=2, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    results = mod.generate_outputs(coref_documents[0], pre_tokenized=True, speaker_ids=coref_speakers[0], genre=1)
    assert set(list(results.keys())) == {"candidate_starts", "candidate_ends", "candidate_mention_scores",
                                         "top_span_starts", "top_span_ends", "top_antecedents", "top_antecedent_scores",
                                         "flattened_ids", "flattened_text", "loss"}


@pytest.mark.unittest
def test_bert_coref_outputs_raw(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=2, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    input_docs = [' '.join(sent) for sent in coref_documents[0]]
    input_speakers = [sent[0] for sent in coref_speakers[0]]
    results = mod.generate_outputs(input_docs, pre_tokenized=False, speaker_ids=input_speakers, genre=1)
    assert set(list(results.keys())) == {"candidate_starts", "candidate_ends", "candidate_mention_scores",
                                         "top_span_starts", "top_span_ends", "top_antecedents", "top_antecedent_scores",
                                         "flattened_ids", "flattened_text", "loss"}


@pytest.mark.unittest
def test_bert_coref_outputs_single_sent(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=2, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    results = mod.generate_outputs(coref_documents[0][0], pre_tokenized=True, speaker_ids=coref_speakers[0][0], genre=1)
    assert set(list(results.keys())) == {"candidate_starts", "candidate_ends", "candidate_mention_scores",
                                         "top_span_starts", "top_span_ends", "top_antecedents", "top_antecedent_scores",
                                         "flattened_ids", "flattened_text", "loss"}


@pytest.mark.unittest
def test_bert_coref_predict(raw_coref_data):
    mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=2, top_span_ratio=1.0, ffnn_hidden_size=100,
                                       use_metadata=True, hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    coref_documents, _, coref_speakers, _ = raw_coref_data
    results = mod.predict(coref_documents[0][0], pre_tokenized=True, speaker_ids=coref_speakers[0][0], genre=1)
    for cluster_dict in results:
        assert len(cluster_dict["cluster_ids"]) == len(cluster_dict["cluster_tokens"])


@pytest.mark.unittest
def test_bert_coref_train_single_learnrate(raw_coref_data, tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    config = BertConfig(hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    base_model = BertModel(config)
    base_model.save_pretrained(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path, max_span_width=1,
                                            top_span_ratio=1.0, ffnn_hidden_size=100, use_metadata=True)

    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 15
    mod.train(mod_path, max_segment_len=max_segment_len, max_segments=5, task_learn_rate=None,
              train_documents=coref_documents, train_cluster_ids=coref_cluster_ids, train_speaker_ids=coref_speakers,
              train_genres=coref_genres, num_train_epochs=1)
    assert not mod.model.training


@pytest.mark.unittest
def test_bert_coref_train_single_learnrate_freezebase(raw_coref_data, tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    config = BertConfig(hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    base_model = BertModel(config)
    base_model.save_pretrained(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path, max_span_width=1,
                                            top_span_ratio=1.0, ffnn_hidden_size=100, use_metadata=True)

    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 15
    mod.train(mod_path, max_segment_len=max_segment_len, max_segments=5, task_learn_rate=None,
              train_documents=coref_documents, train_cluster_ids=coref_cluster_ids, train_speaker_ids=coref_speakers,
              train_genres=coref_genres, num_train_epochs=1, freeze_base=True)
    assert not mod.model.training
    for param in mod.model.base_model.parameters():
        assert not param.requires_grad


@pytest.mark.unittest
def test_bert_coref_train_single_learnrate_eval(raw_coref_data, tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    config = BertConfig(hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    base_model = BertModel(config)
    base_model.save_pretrained(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path, max_span_width=1,
                                            top_span_ratio=1.0, ffnn_hidden_size=100, use_metadata=True)

    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 15
    mod.train(mod_path, max_segment_len=max_segment_len, max_segments=5, task_learn_rate=None,
              train_documents=coref_documents, train_cluster_ids=coref_cluster_ids, train_speaker_ids=coref_speakers,
              train_genres=coref_genres, eval_documents=coref_documents, eval_cluster_ids=coref_cluster_ids,
              eval_speaker_ids=coref_speakers, eval_genres=coref_genres, num_train_epochs=1)
    assert not mod.model.training


@pytest.mark.unittest
def test_bert_coref_train_multiple_learnrate_custom_trainer(raw_coref_data, tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    config = BertConfig(hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    base_model = BertModel(config)
    base_model.save_pretrained(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path, max_span_width=1,
                                            top_span_ratio=1.0, ffnn_hidden_size=100, use_metadata=True)
    coref_documents, coref_cluster_ids, coref_speakers, coref_genres = raw_coref_data
    max_segment_len = 15
    base_learn_rate = 0.001
    task_learn_rate = 0.01
    max_segments = 5
    train_dataset = MangoesCoreferenceDataset(mod.tokenizer, mod.model.config.task_specific_params["use_metadata"],
                                              max_segment_len, max_segments, coref_documents, coref_cluster_ids,
                                              coref_speakers, coref_genres)
    train_args = TrainingArguments(output_dir=mod_path, per_device_eval_batch_size=1,
                                   per_device_train_batch_size=1, num_train_epochs=1, learning_rate=base_learn_rate)
    trainer = CoreferenceFineTuneTrainer(task_learn_rate, "bert", mod.model, args=train_args,
                                         train_dataset=train_dataset, tokenizer=mod.tokenizer)

    mod.train(mod_path, trainer=trainer)
    assert trainer.optimizer.param_groups[0]["initial_lr"] == base_learn_rate
    assert trainer.optimizer.param_groups[2]["initial_lr"] == task_learn_rate


@pytest.mark.unittest
def test_bert_coref_load_base(tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    config = BertConfig(hidden_size=120, num_hidden_layers=2, intermediate_size=256)
    base_model = BertModel(config)
    base_model.save_pretrained(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path, max_span_width=1,
                                            top_span_ratio=1.0, ffnn_hidden_size=100, use_metadata=True)
    assert mod.model.config.task_specific_params["max_span_width"] == 1
    assert mod.model.config.task_specific_params["ffnn_hidden_size"] == 100
    assert mod.model.config.task_specific_params["top_span_ratio"] == 1.0
    assert mod.model.config.task_specific_params["use_metadata"]
    assert mod.model.config.task_specific_params["max_top_antecendents"] == 50
    assert mod.model.config.task_specific_params["metadata_feature_size"] == 20
    assert mod.model.config.num_hidden_layers == 2
    assert mod.model.config.hidden_size == 120
    assert mod.model.config.intermediate_size == 256


@pytest.mark.unittest
def test_bert_coref_load_full(tmpdir_factory):
    mod_path = tmpdir_factory.mktemp('mod_path')
    saved_mod = BERTForCoreferenceResolution("bert-base-uncased", max_span_width=1, top_span_ratio=1.0,
                                             ffnn_hidden_size=100, use_metadata=True, hidden_size=120,
                                             num_hidden_layers=2, intermediate_size=256)

    saved_mod.save(mod_path)
    mod = BERTForCoreferenceResolution.load("bert-base-uncased", pretrained_model=mod_path)
    assert mod.model.config.task_specific_params["max_span_width"] == 1
    assert mod.model.config.task_specific_params["ffnn_hidden_size"] == 100
    assert mod.model.config.task_specific_params["top_span_ratio"] == 1.0
    assert mod.model.config.task_specific_params["use_metadata"]
    assert mod.model.config.task_specific_params["max_top_antecendents"] == 50
    assert mod.model.config.task_specific_params["metadata_feature_size"] == 20
    assert mod.model.config.num_hidden_layers == 2
    assert mod.model.config.hidden_size == 120
    assert mod.model.config.intermediate_size == 256
