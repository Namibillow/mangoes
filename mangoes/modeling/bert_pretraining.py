# -*- coding: utf-8 -*-
"""
This module provides an interface into the transformers BERT models for pretraining using either masked language
modeling and next sentence prediction, or only masked language modeling.
"""
import warnings

import transformers
import torch

from mangoes.modeling.training_utils import MangoesLineByLineDatasetForNSP, MangoesLineByLineIterableDataset, \
    IterableCompatibleTrainer, MangoesLineByLineDataset
from mangoes.modeling.bert_base import PipelineMixin, TransformerModel


class BERTForPreTraining(PipelineMixin, TransformerModel):
    """Class for pretraining a Bert model (masked language modeling and next sentence prediction)
    This is a base bert architecture with the masked language modeling head and next sentence prediction head.

    :var self.tokenizer: transformers.BertTokenizerFast object,
        see https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast for more documentation
    :var self.model: transformers.BertForPreTraining object,
        see https://huggingface.co/transformers/model_doc/bert.html#bertforpretraining

    Parameters
    ----------
    pretrained_tokenizer: str
        Either:
            - A string with the `shortcut name` of a pretrained tokenizer to load from cache or download, e.g.,
              ``bert-base-uncased``.
            - A string with the `identifier name` of a pretrained tokenizer that was user-uploaded to our S3, e.g.,
              ``dbmdz/bert-base-german-cased``.
            - A path to a `directory` containing tokenizer vocab or file saved using
              :func:`tokenizer.save_model` or :func:`tokenizer.save`, e.g., ``./my_model_directory/``.
    device: int, optional, defaults to None
        if -1, use cpu, if >= 0, use CUDA device number. If None, will use GPU if available
    **keyword_args include arguments passed to transformers.BertConfig (see mangoes.modeling.BERTBase docstring)
    """

    def __init__(self, pretrained_tokenizer, device=None, **keyword_args):
        TransformerModel.__init__(self, device)
        PipelineMixin.__init__(self)
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(pretrained_tokenizer)
        vocab_size = len(self.tokenizer.get_vocab())
        config = transformers.BertConfig(vocab_size, **keyword_args)
        self.model = transformers.BertForPreTraining(config)

    def _construct_pipeline(self):
        """
        Implementation for creating inference pipeline.
        """
        self.model.eval()
        mlm_model = transformers.BertForMaskedLM(config=self.model.config)
        bert_weights = self.model.bert.state_dict()
        bert_weights.pop('pooler.dense.weight', None)
        bert_weights.pop('pooler.dense.bias', None)
        mlm_model.bert.load_state_dict(bert_weights)
        mlm_weights = self.model.cls.predictions.state_dict()
        mlm_weights_new = {}
        for key in mlm_weights.keys():
            mlm_weights_new["predictions." + key] = mlm_weights[key]
        mlm_model.cls.load_state_dict(mlm_weights_new)
        self.pipeline = transformers.FillMaskPipeline(mlm_model, self.tokenizer,
                                                      device=-1 if self.model.device.type == "cpu" else
                                                      self.model.device.index)

    def predict(self, inputs, text_pairs=None, top_k=5):
        """
        Predicts masked tokens and next sentence prediction scores.

        Parameters
        ----------
        inputs: str or list of strs
            inputs to predict on
        text_pairs: str or list of strs
            The next sentence for each sentence in the text argument, to predict using the next sentence prediction head
            of the model. Optional, if left blank, next sentence prediction outputs don't mean anything.
            If not None, should be same type as inputs, and same length if list.
        top_k: number of predictions to return per masked token

        Returns
        -------
        masked language predictions: list of dict
            for each masked token, a dict containing:
                sequence (str) – The corresponding input with the mask token prediction.
                score (float) – The corresponding probability.
                token_id (int) – The predicted token id (to replace the masked one).
                token_str (str) – The predicted token (to replace the masked one).
        next_sentence_predictions: List of ints, size (batch_size)
            list of next sentence predictions. If input is 1 string, list will have 1 element. 0 indicates sequence B is
            a continuation of sequence A, 1 indicates sequence B is a random sequence.
        """
        if text_pairs:
            assert type(inputs) == type(text_pairs)
            sequences = []
            if isinstance(inputs, list):
                for i in range(len(inputs)):
                    sequences.append(self.tokenizer.cls_token + " " + inputs[i] + " " + self.tokenizer.sep_token + " "
                                     + text_pairs[i] + " " + self.tokenizer.sep_token)
            else:
                sequences.append(self.tokenizer.cls_token + " " + inputs + " " + self.tokenizer.sep_token + " " +
                                 text_pairs + " " + self.tokenizer.sep_token)
            masked_predictions = self._predict(sequences, top_k=top_k, add_special_tokens=False)
        else:
            masked_predictions = self._predict(inputs, top_k=top_k)
        if isinstance(masked_predictions[0], list):
            masked_predictions = [m[0] for m in masked_predictions]
        outputs = self.generate_outputs(inputs, text_pairs=text_pairs)
        return masked_predictions, torch.argmax(outputs["seq_relationship_logits"], dim=1).tolist()

    def generate_outputs(self, text, text_pairs=None, pre_tokenized=False, output_attentions=False,
                         output_hidden_states=False, word_embeddings=False):
        """
        Tokenize input text and pass it through the BERT model, optionally outputting hidden states or attention
        matrices.

        Parameters
        ----------
        text: str or List[str] if pre_tokenized=False, else List[str] or List[List[str]]
            the text to compute features for.
        text_pairs: str or List[str] if pre_tokenized=False, else List[str] or List[List[str]], optional
            The next sentence for each sentence in the text argument, to predict using the next sentence prediction head
            of the model. Optional, if left blank, next sentence prediction outputs don't mean anything.
        pre_tokenized: Boolean
            whether or not the input text is pretokenized (ie, split on spaces)
        output_attentions: Boolean, optional, defaults to False
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states: Boolean, optional, defaults to False
            Whether or not to return the hidden states of all layers.
        word_embeddings: Boolean
            Whether or not to filter special token embeddings and average subword embeddings (hidden states) into word
            embeddings. This functionality is not available for this task class. Use the feature extraction class
            instead.

        Returns
        -------
        Dict containing (note that if single text sequence is passed as input, batch size will be 1):
        hidden_states: (Tuple (one for each layer) of torch.FloatTensor (batch_size, sequence_length, hidden_size)).
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            Only returned if output_hidden_states is True
        attentions: Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length,
            sequence_length).
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            Only returned if output_attentions is True
        offset_mappings: Tensor of shape (batch_size, sequence_length, 2)
            Tensor containing (char_start, char_end) for each token, giving index into input strings of start and end
            character for each token. If input is pre-tokenized, start and end index maps into associated word. Note
            that special tokens are included with 0 for start and end indices, as these don't map into input text
            because they are added inside the function. If input includes text pairs, offset mappings for both sentences
            in pair are merged to one row, but indices still align to sentences separately.
        prediction_logits: (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size))
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits: (torch.FloatTensor of shape (batch_size, 2))
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax). Only meaningful if text_pairs input argument is non None.
        """
        self.model.eval()
        inputs = self.tokenizer(text, text_pairs, is_split_into_words=pre_tokenized, padding=True, return_tensors='pt',
                                return_offsets_mapping=True)
        offset_mappings = inputs.pop('offset_mapping')
        with torch.no_grad():
            if self.model_device.type == "cuda":
                inputs = {name: tensor.to(self.model_device) for name, tensor in inputs.items()}
                if not next(self.model.parameters()).is_cuda:
                    self.model.to(self.model_device)
            results = self.model.forward(**inputs, return_dict=True, output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states)
        if word_embeddings:
            warnings.warn("Word embedding consolidation not available for pretraining model. Consider using the"
                          "mangoes.modeling.PretrainedTransformerModelForFeatureExtraction class for feature "
                          "extractions.")
        results["offset_mappings"] = offset_mappings
        return dict(results)

    @classmethod
    def load(cls, pretrained_tokenizer, pretrained_model, device=None):
        """
        load a mangoes BERTForPreTraining object from saved tokenizer and model files.

        Parameters
        ----------
        pretrained_tokenizer: str
            Either:
                - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                  ``bert-base-uncased``.
                - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                  ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~save_pretrained`, e.g., ``./my_model_directory/``.
        pretrained_model: str
            Either:
                - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                  ``bert-base-uncased``.
                - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                  ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~save_pretrained`, e.g., ``./my_model_directory/``.
        device: int, optional, defaults to None
            if -1, use cpu, if >= 0, use CUDA device number. If None, will use GPU if available
        """
        model = transformers.BertForPreTraining.from_pretrained(pretrained_model)
        model_object = cls(pretrained_tokenizer, device)
        model_object.model = model
        model_object.model.eval()
        return model_object

    def train(self, output_dir=None, train_text=None, eval_text=None, mlm_probability=0.15,
              short_seq_probability=0.1, nsp_probability=0.5, max_len=None, collator=None, train_dataset=None,
              eval_dataset=None, compute_metrics=None, trainer=None, **training_args):
        """
        Pretrain BERT model on text files

        Input file format:

        (1) One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of
        text.
        (2) Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task
        doesn't span between documents. Opening a new file will assume a new document as well.
        (3) Assumes different input files contain different documents. (ie, a document cannot span multiple files)


        Example:
        I am very happy. (new line) Here is the second sentence. (new line) (new line) A new document.

        Parameters
        ----------
        output_dir: str
            Path to the output directory where the model predictions and checkpoints will be written. Used to
            instantiate Trainer if trainer argument is None.
        train_text: str or List[str]
            path to training file(s). See above note on formatting for Next Sentence Prediction
        eval_text: (Optional) str or List[str]
            path to evaluation file(s). See above note on formatting for Next Sentence Prediction
        mlm_probability: float
            probability of tokens to be masked while pretraining
        short_seq_probability: float
            probability of short sequences when generating next sentence pairs
        nsp_probability: float
            probability of random next sentence when generating next sentence pairs
        max_len: int
            max length of input sequence. Will default to self.tokenizer.max_length() if None
        collator: Transformers.DataCollator
            custom collator to use
        train_dataset, eval_dataset: torch.Dataset
            instantiated custom dataset object
        compute_metrics: function
            The function that will be used to compute metrics at evaluation. Must return a dictionary string to metric
            values. Used by the trainer, see https://huggingface.co/transformers/training.html#trainer for more info.
        trainer: Transformers.Trainer
            custom instantiated trainer to use
        training_args:
            keyword arguments for training. For complete list, see
            https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
        """
        if not trainer:
            if not output_dir:
                raise RuntimeError("Must provide output directory argument to train() method if trainer argument is "
                                   "None")
            if not collator:
                collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=mlm_probability)
            if not train_dataset:
                if not train_text:
                    raise RuntimeError("No train text or dataset provided to train method")
                train_dataset = MangoesLineByLineDatasetForNSP(train_text, self.tokenizer,
                                                               short_seq_probability=short_seq_probability,
                                                               nsp_probability=nsp_probability, max_len=max_len)
            if eval_text and not eval_dataset:
                eval_dataset = MangoesLineByLineDatasetForNSP(eval_text, self.tokenizer,
                                                              short_seq_probability=short_seq_probability,
                                                              nsp_probability=nsp_probability, max_len=max_len)
            if eval_dataset is not None and "evaluation_strategy" not in training_args:
                training_args["evaluation_strategy"] = "epoch"
            train_args = transformers.TrainingArguments(output_dir=output_dir, **training_args)
            trainer = transformers.Trainer(self.model, args=train_args, train_dataset=train_dataset,
                                           eval_dataset=eval_dataset, data_collator=collator, tokenizer=self.tokenizer,
                                           compute_metrics=compute_metrics)
        trainer.train()
        self.model.eval()


class BERTForMaskedLanguageModeling(PipelineMixin, TransformerModel):
    """Class for pretraining a Bert model (masked language modeling and next sentence prediction)
    This is a base bert architecture with the masked language modeling head.

    :var self.tokenizer: transformers.BertTokenizerFast object,
        see https://huggingface.co/transformers/model_doc/bert.html#berttokenizerfast for more documentation
    :var self.model: transformers.BertForMaskedLM object,
        see https://huggingface.co/transformers/model_doc/bert.html#bertformaskedlm

    Parameters
    ----------
    pretrained_tokenizer: str
        Either:
            - A string with the `shortcut name` of a pretrained tokenizer to load from cache or download, e.g.,
              ``bert-base-uncased``.
            - A string with the `identifier name` of a pretrained tokenizer that was user-uploaded to our S3, e.g.,
              ``dbmdz/bert-base-german-cased``.
            - A path to a `directory` containing tokenizer vocab or file saved using
              :func:`tokenizer.save_model` or :func:`tokenizer.save`, e.g., ``./my_model_directory/``.
    device: int, optional, defaults to None
        if -1, use cpu, if >= 0, use CUDA device number. If None, will use GPU if available
    **keyword_args include arguments passed to transformers.BertConfig (see mangoes.modeling.BERTBase docstring)
    """

    def __init__(self, pretrained_tokenizer, device=None, **keyword_args):
        TransformerModel.__init__(self, device)
        PipelineMixin.__init__(self)
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(pretrained_tokenizer)
        vocab_size = len(self.tokenizer.get_vocab())
        config = transformers.BertConfig(vocab_size, **keyword_args)
        self.model = transformers.BertForMaskedLM(config)

    def _construct_pipeline(self):
        """
        Implementation for creating inference pipeline.
        """
        self.model.eval()
        self.pipeline = transformers.FillMaskPipeline(self.model, self.tokenizer,
                                                      device=-1 if self.model.device.type == "cpu" else
                                                      self.model.device.index)

    def predict(self, inputs, top_k=5):
        """
        Predicts masked tokens

        Parameters
        ----------
        inputs: str or list of strs
            inputs to fill masked tokens
        top_k: number of predictions to return per masked token

        Returns
        -------
        list of dict, or list of list of dict
        for each masked token, a dict containing:
            sequence (str) – The corresponding input with the mask token prediction.
            score (float) – The corresponding probability.
            token (int) – The predicted token id (to replace the masked one).
            token (str) – The predicted token (to replace the masked one).
        """
        return self._predict(inputs, top_k=top_k)

    @classmethod
    def load(cls, pretrained_tokenizer, pretrained_model, device=None):
        """
        Load a mangoes BERTForMaskedLanguageModeling object from saved tokenizer and model files.

        Parameters
        ----------
        pretrained_tokenizer: str
            Either:
                - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                  ``bert-base-uncased``.
                - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                  ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~save_pretrained`, e.g., ``./my_model_directory/``.
        pretrained_model: str
            Either:
                - A string with the `shortcut name` of a pretrained model to load from cache or download, e.g.,
                  ``bert-base-uncased``.
                - A string with the `identifier name` of a pretrained model that was user-uploaded to our S3, e.g.,
                  ``dbmdz/bert-base-german-cased``.
                - A path to a `directory` containing model weights saved using
                  :func:`~save_pretrained`, e.g., ``./my_model_directory/``.
        device: int, optional, defaults to None
            if -1, use cpu, if >= 0, use CUDA device number. If None, will use GPU if available
        """
        model = transformers.BertForMaskedLM.from_pretrained(pretrained_model)
        model_object = cls(pretrained_tokenizer, device)
        model_object.model = model
        model_object.model.eval()
        return model_object

    def train(self, output_dir=None, train_text=None, eval_text=None, mlm_probability=0.15, iterable_dataset=True,
              max_len=None, collator=None, train_dataset=None, eval_dataset=None, compute_metrics=None, trainer=None,
              **training_args):
        """
        Pretrain BERT model on text files using only masked lm.

        Parameters
        ----------
        output_dir: str, defaults to None
            Path to the output directory where the model predictions and checkpoints will be written. Used to
            instantiate Trainer if trainer argument is None.
        train_text: str or List[str]
            path to training file(s)
        eval_text: (Optional) str or List[str]
            path to evaluation file(s)
        mlm_probability: float
            probability of tokens to be masked while pretraining
        iterable_dataset: Boolean
            when loading dataset during training, whether to process on the fly with no shuffling (True) or load all
            into memory with shuffling (False)
        max_len: int
            max length of input sequence. Will default to self.tokenizer.max_length() if None
        collator: Transformers.DataCollator
            custom collator to use
        train_dataset, eval_dataset: torch.Dataset
            instantiated custom dataset object
        compute_metrics: function
            The function that will be used to compute metrics at evaluation. Must return a dictionary string to metric
            values. Used by the trainer, see https://huggingface.co/transformers/training.html#trainer for more info.
        trainer: Transformers.Trainer
            custom instantiated trainer to use
        training_args:
            keyword arguments for training. For complete list, see
            https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments
        """
        if not trainer:
            if not output_dir:
                raise RuntimeError("Must provide output directory argument to train() method if trainer argument is "
                                   "None")
            if not collator:
                collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm_probability=mlm_probability)
            if not train_dataset and iterable_dataset:
                if not train_text:
                    raise RuntimeError("No train text or dataset provided to train method")
                train_dataset = MangoesLineByLineIterableDataset(train_text, self.tokenizer, max_len=max_len)
                if eval_text:
                    eval_dataset = MangoesLineByLineIterableDataset(eval_text, self.tokenizer, max_len=max_len)
                if eval_dataset is not None and "evaluation_strategy" not in training_args:
                    training_args["evaluation_strategy"] = "epoch"
                train_args = transformers.TrainingArguments(output_dir=output_dir, **training_args)
                trainer = IterableCompatibleTrainer(self.model, args=train_args, train_dataset=train_dataset,
                                                    eval_dataset=eval_dataset if eval_text else None,
                                                    data_collator=collator, tokenizer=self.tokenizer)
            else:
                if not train_dataset:
                    if not train_text:
                        raise RuntimeError("No train text or dataset provided to train method")
                    train_dataset = MangoesLineByLineDataset(train_text, self.tokenizer, max_len=max_len)
                if eval_text and not eval_dataset:
                    eval_dataset = MangoesLineByLineDataset(eval_text, self.tokenizer, max_len=max_len)
                if eval_dataset is not None and "evaluation_strategy" not in training_args:
                    training_args["evaluation_strategy"] = "epoch"
                train_args = transformers.TrainingArguments(output_dir=output_dir, **training_args)
                trainer = transformers.Trainer(self.model, args=train_args, train_dataset=train_dataset,
                                               eval_dataset=eval_dataset, data_collator=collator,
                                               tokenizer=self.tokenizer, compute_metrics=compute_metrics)
        trainer.train()
        self.model.eval()
