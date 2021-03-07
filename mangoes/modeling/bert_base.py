# -*- coding: utf-8 -*-
"""
This module provides an interface into the transformers BERT models, including pretrained models, pretraining from
scratch, and fine tuning.
The main BERT classes inherit from transformers BERT models and include tokenizers, allowing for easy inference and
feature extraction.
"""
import warnings
from abc import ABC, abstractmethod

import transformers
import numpy as np
import torch


def merge_subword_embeddings(embeddings, text, token_offsets, pretokenized=False):
    """
    Function to merge possible subword embeddings for a text to word embeddings, and filter out special token
    embeddings.
    If a word in the text is represented by two or more sub-words, this function will average the sub-word
    embeddings to create a single embedding vector for this word.
    Accepts torch tensors or numpy arrays.

    Parameters
    ----------
    embeddings: numpy array or torch tensor of shape (num_tokens, embedding_size) if one sentence, or (num_sentences,
        num_tokens, embedding_size) if multiple sentences
        the subword embeddings to be merged
    text: str or List[str] if pre_tokenized=False, else List[str] or List[List[str]]
        the text corresponding to the embedding output
    token_offsets: (char_start, char_end) for each token. Indices correspond to either text (in not pretokenized)
        or word (if pretokenized)
    pretokenized: if input into tokenizer was split into words

    Returns
    -------
    numpy array or torch tensor with sub-word embeddings averaged to form word embeddings, padded to number of words in
        longest sentence. Padded words will contain vectors of zeros.
    """
    if not pretokenized:
        words = [text.split()] if isinstance(text, str) else [sent.split() for sent in text]
    else:
        words = [text] if isinstance(text[0], str) else text
    embedding_size = embeddings.shape[-1]
    num_sentences = len(words)
    max_words = max([len(sent) for sent in words])
    if isinstance(embeddings, torch.Tensor):
        to_torch = True
        embeddings = embeddings.cpu().numpy()
    else:
        to_torch = False
    output_array = np.zeros((num_sentences, max_words, embedding_size), dtype=embeddings.dtype)
    for i in range(num_sentences):
        current_token_index = 0
        while token_offsets[i][current_token_index][1] == 0:
            # loop past any special tokens at the start
            current_token_index += 1
        for j in range(len(words[i])):
            current_start = current_token_index
            current_end = current_token_index
            while current_end < len(token_offsets[i]) - 1 and \
                    token_offsets[i][current_end][1] == token_offsets[i][current_end + 1][0]:
                current_end += 1
            if current_start == current_end:
                output_array[i, j, :] = embeddings[i, current_start, :]
            else:
                output_array[i, j, :] = np.mean(embeddings[i][current_start:current_end + 1][:], 0)
            current_token_index = current_end + 1
    if to_torch:
        output_array = torch.from_numpy(output_array)
    return output_array


class PipelineMixin(ABC):
    """
    Mixin class for tasks covered by transformers Pipelines.
    """
    def __init__(self):
        self.pipeline = None

    @abstractmethod
    def _construct_pipeline(self):
        """
        abstract method for creating inference pipeline
        """

    def _predict(self, inputs, **keyword_args):
        if not self.pipeline:
            self._construct_pipeline()
        if inputs:
            return self.pipeline(inputs, **keyword_args)
        return self.pipeline(**keyword_args)


class TransformerModel(ABC):
    """Base class for mangoes BERT models.
    Includes functionality for extracting embeddings from trained base Bert model.

    Parameters
    ----------
    device: int, or None
        if -1, use cpu, if >= 0, use CUDA device number. If None, will use GPU if available
    """
    def __init__(self, device=None):
        self.model = None
        self.tokenizer = None
        if not device:
            # will use GPU for feature extraction, if available and no device input argument
            if torch.cuda.is_available():
                self.model_device = torch.device("cuda:0")
            else:
                self.model_device = torch.device("cpu")
        else:
            self.model_device = torch.device("cpu" if device < 0 else "cuda:{}".format(device))
            if self.model_device.type == "cuda":
                if torch.cuda.device_count() <= device:
                    warnings.warn(f"CUDA device {device} is not available on this machine, using CPU for BERT model",
                                  RuntimeWarning)
                    self.model_device = torch.device("cpu")

    @abstractmethod
    def train(self, output_dir=None, train_text=None, eval_text=None, collator=None, train_dataset=None,
              eval_dataset=None, trainer=None, **training_args):
        """
        abstract method for BERT training
        """

    @classmethod
    @abstractmethod
    def load(cls, pretrained_tokenizer, pretrained_model):
        """
        abstract method for loading a pretrained tokenizer and model
        """

    def save(self, output_directory, save_tokenizer=False):
        """
        Method to save BERT model and optionally save tokenizer. The tokenizer is already saved (the input to this class
        includes a pretrained tokenizer), but this method will save the tokenizer as well if needed.
        Both the tokenizer files and model files will be saved to the output directory. The output directory can be
        inputted as an argument to the "load()" method of the inheriting classes (for the model and tokenizer
        arguments)

        Parameters
        ----------
        output_directory: str
            path to directory to save model
        save_tokenizer: Boolean
            whether to save tokenizer in directory or not, defaults to False
        """
        self.model.save_pretrained(output_directory)
        if save_tokenizer:
            self.tokenizer.save_pretrained(output_directory)

    def generate_outputs(self, text, pre_tokenized=False, output_attentions=False, output_hidden_states=False,
                         word_embeddings=False):
        """
        Tokenize input text and pass it through the BERT model, optionally outputting hidden states or attention
        matrices.

        Parameters
        ----------
        text: str or List[str] if pre_tokenized=False, else List[str] or List[List[str]]
            the text to compute features for.
        pre_tokenized: Boolean
            whether or not the input text is pre-tokenized (ie, split on spaces)
        output_attentions: Boolean, optional, defaults to False
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states: Boolean, optional, defaults to False
            Whether or not to return the hidden states of all layers.
        word_embeddings: Boolean
            whether or not to filter special token embeddings and average sub-word embeddings (hidden states) into word
            embeddings. If pre-tokenized inputs, the sub-word embeddings will be averaged into the tokens pass as i
            nputs.
            If pre-tokenized=False, the text will be split on whitespace and the sub-word embeddings will be averaged
            back into these words produced by splitting the text on whitespace.
            Only used if output_hidden_states = True.
            If False, number of output embeddings could be greater than (number of words + special tokens).
            If True, number of output embeddings == number of words, sub-words are averaged together to create word
            level embeddings and special token embeddings are excluded.

        Returns
        -------
        Dict containing (note that if single text sequence is passed as input, batch size will be 1):
        hidden_states: (Tuple (one for each layer) of torch.FloatTensor (batch_size, sequence_length, hidden_size)).
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            Only returned if output_hidden_states is True. If word_embeddings, the sequence length will be the number of
            words in the longest sentence, ie the maximum number of words. Shorter sequences will be padded with zeros.
        attentions: Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length,
            sequence_length).
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            Only returned if output_attentions is True
        offset_mappings: Tensor of shape (batch_size, sequence_length, 2)
            Tensor containing (char_start, char_end) for each token, giving index into input strings of start and end
            character for each token. If input is pre-tokenized, start and end index maps into associated word. Note
            that special tokens are included with 0 for start and end indices, as these don't map into input text
            because they are added inside the function.
            This output is only available to tokenizers that inherit from transformers.PreTrainedTokenizerFast . This
            includes the BERT tokenizer and most other common tokenizers, but not all possible tokenizers in the
            library. If the tokenizer did not inherit from this class, this output value will be None.
        if PretrainedTransformerModelForFeatureExtraction:
            last_hidden_state: (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output: (torch.FloatTensor of shape (batch_size, hidden_size))
                Last layer hidden-state of the first token of the sequence (classification token) further processed by a
                Linear layer and a Tanh activation function.
        if BERTForPreTraining:
            prediction_logits: (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            seq_relationship_logits: (torch.FloatTensor of shape (batch_size, 2))
                Prediction scores of the next sequence prediction (classification) head (scores of True/False
                continuation before SoftMax).
        if BERTForMaskedLanguageModeling:
            logits: (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size))
                Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        if BERTForSequenceClassification:
            logits: (torch.FloatTensor of shape (batch_size, config.num_labels))
                classification scores, before softmax
        if BERTForTokenClassification:
            logits: (torch.FloatTensor of shape (batch_size, sequence_length, config.num_labels))
                classification scores, before softmax
        """
        self.model.eval()
        inputs = self.tokenizer(text, is_split_into_words=pre_tokenized,
                                return_offsets_mapping=isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast),
                                padding=True, return_tensors='pt')
        if isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast):
            offset_mappings = inputs.pop('offset_mapping')
        else:
            offset_mappings = None
        with torch.no_grad():
            if self.model_device.type == "cuda":
                inputs = {name: tensor.to(self.model_device) for name, tensor in inputs.items()}
                if not next(self.model.parameters()).is_cuda:
                    self.model.to(self.model_device)
            results = self.model.forward(**inputs, return_dict=True,  output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states)
        if output_hidden_states and word_embeddings:
            if offset_mappings is not None:
                hidden_states = []  # tuples are immutable so need to create new and replace in results dict
                for i in range(len(results["hidden_states"])):
                    hidden_states.append(merge_subword_embeddings(results["hidden_states"][i], text,
                                                                  token_offsets=offset_mappings.numpy(),
                                                                  pretokenized=pre_tokenized))
                results["hidden_states"] = hidden_states
            else:
                warnings.warn("Tokenizer type does not support offset mappings, so word embedding consolidation is not "
                              "possible", RuntimeWarning)
        results["offset_mappings"] = offset_mappings
        return dict(results)


class PretrainedTransformerModelForFeatureExtraction(TransformerModel, PipelineMixin):
    """Class for using a pretrained transformer model and tokenizer. This class is meant to be used if you want to use a
    model that is not covered by the BERT classes in this module. It utilizes the AutoModel and AutoTokenizer classes
    in transformers. As it uses AutoModel (and not AutoModelForQuestionAnswering, for example), it can only be used
    for base pretrained models, and not fine tuning tasks.

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
    def __init__(self, pretrained_tokenizer, pretrained_model, device=None):
        TransformerModel.__init__(self, device)
        PipelineMixin.__init__(self)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_tokenizer)
        self.model = transformers.AutoModel.from_pretrained(pretrained_model)
        self.model.eval()

    def _construct_pipeline(self):
        """
        Implementation for creating inference pipeline.
        """
        self.model.eval()
        self.pipeline = transformers.FeatureExtractionPipeline(self.model, self.tokenizer,
                                                               device=-1 if self.model.device.type == "cpu" else
                                                               self.model.device.index)

    def predict(self, inputs, **kwargs):
        """
        Run input text through the feature extraction pipeline, extracting the hidden states of each layer.

        Parameters
        ----------
        inputs: str or list of strs
            inputs to extract features

        Returns
        -------
        nested list of float, hidden states.
        """
        return self._predict(inputs, **kwargs)

    @classmethod
    def load(cls, pretrained_tokenizer, pretrained_model, device=None):
        """
        Load a model and tokenizer. For this class, this method is essentially the same as the init function, and the
        two can be used interchangeably.

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
        model_object = cls(pretrained_tokenizer, pretrained_model, device)
        return model_object

    def train(self, output_dir=None, train_dataset=None, eval_dataset=None, collator=None, trainer=None,
              **training_args):
        """
        This function does nothing, use a task specific BERT class to pre-train or fine-tune
        """
        warnings.warn("This base model class does not implement training, it is meant for feature extraction")
