# -*- coding: utf-8 -*-
"""
This module provides helper classes for training huggingface models using the interface in mangoes.modeling.bert.
These classes are mainly called internally from mangoes.modeling.bert, however these classes can be instantiated on
their own (or subclassed) and passed to training methods in mangoes.modeling.bert for more customization/control.
"""
import fileinput
import collections
import random

import transformers
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset


def freeze_base_layers(model):
    """
    Function to freeze the base layer in a fine tuning/pretraining model
    """
    for param in model.base_model.parameters():
        param.requires_grad = False


class MangoesLineByLineIterableDataset(IterableDataset):
    """
    Subclass of torch.utils.data.IterableDataset.
    Used for large text datasets that are not to be loaded into memory at the same time.

    Parameters
    ----------
    filenames: str or List[str]
        paths to file containing text
    tokenizer: transformers.Tokenizer
    max_len: int
        size of input tensors. If None, will default to tokenizer.model_max_length
    length: int
        number of lines in the dataset. If None, will lazily calculate it
    encoding: str
        encoding of text files
    """
    def __init__(self, filenames, tokenizer, max_len=None, length=None, encoding=None):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.tokenizer = tokenizer
        self.block_size = max_len if max_len else tokenizer.model_max_length
        self.encoding = encoding
        self.length = length

    def preprocess(self, text):
        """
        Preprocess a line of text using the tokenizer

        Parameters
        ----------
        text: str

        Returns
        -------
        dictionary containing input_id encodings
        """
        batch_encoding = self.tokenizer(text.strip("\n"), truncation=True, max_length=self.block_size)
        examples = batch_encoding["input_ids"]
        return {"input_ids": torch.tensor(examples, dtype=torch.long)}

    def line_mapper(self, line):
        return self.preprocess(line)

    def __iter__(self):
        reader = fileinput.FileInput(self.filenames, mode="r",
                                     openhook=None if not self.encoding else fileinput.hook_encoded(self.encoding))
        mapped_itr = map(self.line_mapper, reader)
        return mapped_itr

    def __len__(self):
        if not self.length:
            self.length = self._get_length()
        return self.length

    def _get_length(self):
        """
        Helper function to get number of line in texts
        """
        return sum(1 for _ in fileinput.FileInput(self.filenames, mode="r",
                                                  openhook=None if not self.encoding else
                                                  fileinput.hook_encoded(self.encoding)))


class MangoesTextClassificationDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset class for sequence or token classification.
    To be used with transformers.Trainer.

    Parameters
    ----------
    texts: List[str]
    labels: List[str or int] if seq classification, else List[List[str or int]] if token classification
        Labels can be raw strings, which will us label2id to convert to ids, or the ids themselves.
    tokenizer: transformers.Tokenizer
    max_len: int
        max length of input sequences, if None, will default to tokenizer.model_max_length
    label2id: dict of str -> int
        if labels are not already converted to output ids, dictionary with mapping to use.
    """
    def __init__(self, texts, labels, tokenizer, max_len=None, label2id=None):
        self.texts = texts
        token_classes = True if isinstance(labels[0], list) else False
        if token_classes:
            # if token classification
            raw_labels = True if isinstance(labels[0][0], str) else False
        else:
            # if sequence classification
            raw_labels = True if isinstance(labels[0], str) else False
        if raw_labels and not label2id:
            raise TypeError('Labels passed to dataset are not converted into output ids and no id mapping was passed.')
        if raw_labels:
            if token_classes:
                labels = [[label2id[label] for label in sublist] for sublist in labels]
            else:
                labels = [label2id[label] for label in labels]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len if max_len else tokenizer.model_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = self.labels[item]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class MangoesLineByLineDataset(Dataset):
    """
    Subclass of torch.utils.data.Dataset class for pretraining language models.
    Similar to transformers.LineByLineTextDataset, but supports multiple input files

    Parameters
    ----------
    filenames: str or List[str]
        paths to files to include in dataset.
    tokenizer: transformers.Tokenizer
    max_len: int
        max length of input sequences, if None, will default to tokenizer.model_max_length
    encoding: str
        encoding of text files
    """
    def __init__(self, filenames, tokenizer, max_len=None, encoding=None):
        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.encoding = encoding
        with fileinput.input(self.filenames, mode="r", openhook=None if not self.encoding else
                             fileinput.hook_encoded(self.encoding)) as f:
            lines = [line for line in f if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=max_len)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(example, dtype=torch.long)} for example in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class MangoesQuestionAnsweringDataset(Dataset):
    """
    Subclass of Torch Dataset for question answering datasets. Currently meant to work with BERT models.

    Parameters
    ----------
    tokenizer: transformers.Tokenizer
    question_texts: List of str
        The texts corresponding to the questions
    context_texts: List of str
        The texts corresponding to the contexts
    answer_texts: List of str
        The texts corresponding to the answers
    start_indices: List of int
        The character positions of the start of the answers
    max_seq_length:int
        The maximum total input sequence length after tokenization.
    doc_stride: int
        When splitting up a long document into chunks, how much stride to take between chunks.
    max_query_length: int
        The maximum number of tokens for the question.
    """
    def __init__(self, tokenizer, question_texts, context_texts, answer_texts, start_indices, max_seq_length=384,
                 doc_stride=128, max_query_length=64):

        if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            import os
            # TODO: (02/12/2020) currently the FastTokenizers don't work with QAPiplines, so we have to typecast
            # https://github.com/huggingface/transformers/issues/7735
            # once this is fixed in transformers, we should remove this hacky fix
            tokenizer.save_vocabulary("./")
            tokenizer = transformers.BertTokenizer("./vocab.txt", **tokenizer.init_kwargs)
            os.remove("./vocab.txt")
        # convert to squad examples
        if not len(question_texts) == len(answer_texts) or not len(question_texts) == len(start_indices) or \
                not len(question_texts) == len(context_texts):
            raise ValueError("Question Answering dataset needs answers, contexts, and start indices for every example")
        examples = []
        for i in range(len(question_texts)):
            examples.append(transformers.SquadExample(qas_id=len(examples),
                                                      question_text=question_texts[i],
                                                      context_text=context_texts[i],
                                                      answer_text=answer_texts[i],
                                                      start_position_character=start_indices[i],
                                                      title=""))
        self.features = transformers.squad_convert_examples_to_features(examples, tokenizer,
                                                                        max_seq_length=max_seq_length,
                                                                        doc_stride=doc_stride,
                                                                        max_query_length=max_query_length,
                                                                        is_training=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        feature = self.features[i]

        inputs = {
            "input_ids": torch.tensor(feature.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feature.attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(feature.token_type_ids, dtype=torch.long),
        }
        start_positions = torch.tensor(feature.start_position, dtype=torch.long)
        end_positions = torch.tensor(feature.end_position, dtype=torch.long)
        inputs.update({"start_positions": start_positions, "end_positions": end_positions})
        return inputs


class MangoesCoreferenceDataset(Dataset):
    """
    Subclass of Torch Dataset for co-reference datasets such as Ontonotes. Currently meant to work with BERT models.

    Each example is one document. Documents are parsed by first tokenizing each sentence then aggregating sentences into
    segments, keeping track of label, metadata, and sentence indices.

    Parameters
    ----------
    tokenizer: transformers.BertTokenizerFast
        tokenizer to use
    use_metadata: Boolean
        Whether or not to use speaker ids and genres
    max_segment_len: int
        maximum number of sub-tokens for one segment
    max_segments: int
        Maximum number of segments to return per __getitem__ (ie per document)
    documents: List of Lists of Lists of strings
        Text for each document. As cluster ids are labeled by word, a document is a list of sentences. One
        sentence is a list of words (ie already split on whitespace/punctuation)
    cluster_ids: List of Lists of Lists of (ints or Tuple(int, int))
        Cluster ids for each word in documents argument. Assumes words that aren't mentions have either None or -1 as
        id. In the case where a word belongs to two different spans (with different cluster ids), the cluster id for
        word should be a tuple of ints corresponding to the different cluster ids.
    speaker_ids: List of Lists of Lists of ints
        Speaker id for each word in documents. Assumes positive ids (special tokens (such as [CLS] and [SEP] that are
        added at beginning and end of segments) will be assigned speaker ids of -1)
    genres: List of ints or strings
        Genre (id) for each document. If strings, genre_to_id parameter needs to not be None
    genre_to_id: dict of string->int
        Mapping of genres to their id number.
    """
    def __init__(self, tokenizer, use_metadata, max_segment_len, max_segments, documents, cluster_ids, speaker_ids=None,
                 genres=None, genre_to_id=None):
        self.use_metadata = use_metadata
        if (use_metadata and not speaker_ids) or (use_metadata and not genres):
            raise RuntimeError("use_metadata argument is set to True in MangoesCoreferenceDataset init function, but "
                               "missing speaker and/or genre input data")
        if use_metadata and isinstance(genres[0], str) and not genre_to_id:
            raise RuntimeError("Input genre data has not been converted to ids yet, and genre_to_id parameter is "
                               "unfilled")
        self.examples = []
        self.max_segments = max_segments
        for i in range(len(documents)):
            # for each sentence, tokenize into word pieces then aggregate input ids, clusters, speakers
            subtoken_ids = []
            subtoken_cluster_ids = []
            subtoken_speakers = []
            subtoken_offset_mappings = []
            for j in range(len(documents[i])):
                encoding = tokenizer(documents[i][j], add_special_tokens=False, is_split_into_words=True,
                                     return_offsets_mapping=True)
                subtoken_ids.append(encoding["input_ids"])
                subtoken_cluster_ids.append(self.get_subtoken_data(cluster_ids[i][j], encoding["offset_mapping"]))
                subtoken_offset_mappings.append(encoding["offset_mapping"])
                if use_metadata and speaker_ids:
                    subtoken_speakers.append(self.get_subtoken_data(speaker_ids[i][j], encoding["offset_mapping"]))

            # aggregate into segments
            assert len(subtoken_ids) == len(subtoken_cluster_ids)
            current_segment_ids = []
            current_segment_cluster_ids = []
            current_segment_speaker_ids = []
            current_sentence_map = []
            segments_ids = []
            segments_clusters = []
            segments_speakers = []
            segments_attention_mask = []
            sentence_map = []
            for j in range(len(subtoken_ids)):
                if len(current_segment_ids) + len(subtoken_ids[j]) <= max_segment_len - 2:
                    current_segment_ids += subtoken_ids[j]
                    current_segment_cluster_ids += subtoken_cluster_ids[j]
                    current_sentence_map += [j] * len(subtoken_ids[j])
                    if use_metadata and speaker_ids:
                        current_segment_speaker_ids += subtoken_speakers[j]
                else:
                    if len(current_segment_ids) > 0:
                        # segments contain cls and sep special tokens at beginning and end for BERT processing
                        segments_ids.append(self.pad_list([tokenizer.cls_token_id] + current_segment_ids +
                                                          [tokenizer.sep_token_id], max_segment_len,
                                                          tokenizer.convert_tokens_to_ids(tokenizer.pad_token)))
                        segments_clusters.append(self.pad_list([None] + current_segment_cluster_ids + [None],
                                                               max_segment_len, None))
                        segments_attention_mask.append(self.pad_list([1] * (len(current_segment_ids) + 2),
                                                                     max_segment_len))
                        sentence_map += [current_sentence_map[0]] + current_sentence_map + [current_sentence_map[-1]]
                        if use_metadata and speaker_ids:
                            segments_speakers.append(self.pad_list([-1] + current_segment_speaker_ids + [-1],
                                                                   max_segment_len))
                    if len(subtoken_ids[j]) > max_segment_len - 2:
                        # if sentence j is longer than max_seq_len, create segment out of as much as possible,
                        # then remove these from sentence j and continue
                        segment_stop_index = max_segment_len - 2
                        while subtoken_offset_mappings[j][segment_stop_index-1][0] > 0 or \
                                subtoken_offset_mappings[j][segment_stop_index][0] > 0:
                            # if breaking sentence in the middle of a token, truncate so whole token is in next segment
                            segment_stop_index -= 1
                        segments_ids.append(self.pad_list([tokenizer.cls_token_id] +
                                                          subtoken_ids[j][:segment_stop_index] +
                                                          [tokenizer.sep_token_id], max_segment_len,
                                                          tokenizer.convert_tokens_to_ids(tokenizer.pad_token)))
                        segments_clusters.append(self.pad_list([None] + subtoken_cluster_ids[j][:segment_stop_index]
                                                               + [None], max_segment_len, None))
                        segments_attention_mask.append(
                            self.pad_list([1] * (segment_stop_index + 2), max_segment_len))
                        sentence_map += [j] * (segment_stop_index + 2)
                        if use_metadata and speaker_ids:
                            segments_speakers.append(self.pad_list([-1] + subtoken_speakers[j][:segment_stop_index] +
                                                                   [-1], max_segment_len))
                        # remove already added data
                        subtoken_ids[j] = subtoken_ids[j][segment_stop_index:]
                        subtoken_cluster_ids[j] = subtoken_cluster_ids[j][segment_stop_index:]
                        if use_metadata and speaker_ids:
                            subtoken_speakers[j] = subtoken_speakers[j][segment_stop_index:]
                    current_segment_ids = subtoken_ids[j]
                    current_segment_cluster_ids = subtoken_cluster_ids[j]
                    current_sentence_map = [j] * len(subtoken_ids[j])
                    if use_metadata and speaker_ids:
                        current_segment_speaker_ids = subtoken_speakers[j]
            # get last segment
            segments_ids.append(self.pad_list([tokenizer.cls_token_id] + current_segment_ids +
                                              [tokenizer.sep_token_id], max_segment_len,
                                              tokenizer.convert_tokens_to_ids(tokenizer.pad_token)))
            segments_clusters.append(self.pad_list([None] + current_segment_cluster_ids + [None],
                                                   max_segment_len, None))
            segments_attention_mask.append(self.pad_list([1] * (len(current_segment_ids) + 2), max_segment_len))
            sentence_map += [current_sentence_map[0]] + current_sentence_map + [current_sentence_map[-1]]

            if use_metadata:
                segments_speakers.append(self.pad_list([-1] + current_segment_speaker_ids + [-1],
                                                       max_segment_len))
            # create document level info (cluster indices, cluster ids, sentence map, genre)
            gold_starts = []
            gold_ends = []
            gold_cluster_ids = []
            current_offset = 0
            for j in range(len(segments_clusters)):
                # loop over segments and create gold start/ends, ids
                valid_sub_tokens = sum(segments_attention_mask[j])
                cluster_sightings = {}  # keys: clusterids, values: all indices where that clusterid is observed
                for k in range(valid_sub_tokens):
                    if segments_clusters[j][k]:
                        if isinstance(segments_clusters[j][k], tuple) or isinstance(segments_clusters[j][k], list):
                            for clus_id in segments_clusters[j][k]:
                                if clus_id in cluster_sightings:
                                    cluster_sightings[clus_id].append(k)
                                else:
                                    cluster_sightings[clus_id] = [k]
                        elif segments_clusters[j][k] >= 0:
                            if segments_clusters[j][k] in cluster_sightings:
                                cluster_sightings[segments_clusters[j][k]].append(k)
                            else:
                                cluster_sightings[segments_clusters[j][k]] = [k]
                for clus_id, indices in cluster_sightings.items():
                    indices_pointer = 0
                    while indices_pointer < len(indices):
                        gold_starts.append(indices[indices_pointer] + current_offset)
                        gold_cluster_ids.append(clus_id)
                        while indices_pointer < len(indices) - 1 and indices[indices_pointer] == \
                                indices[indices_pointer + 1] - 1:
                            indices_pointer += 1
                        gold_ends.append(indices[indices_pointer] + current_offset)
                        indices_pointer += 1
                current_offset += valid_sub_tokens
            # sort cluster data by cluster start
            cluster_data = sorted(zip(gold_starts, gold_ends, gold_cluster_ids), key=lambda x: x[0])
            cluster_data = [list(t) for t in zip(*cluster_data)]
            if len(cluster_data) == 0:
                cluster_data = [[], [], []]
            self.examples.append([torch.as_tensor(segments_ids), torch.as_tensor(segments_attention_mask),
                                  torch.as_tensor(sentence_map), torch.as_tensor(cluster_data[0]),
                                  torch.as_tensor(cluster_data[1]), torch.as_tensor(cluster_data[2])])
            if use_metadata:
                self.examples[-1].append(torch.as_tensor(segments_speakers))
                self.examples[-1].append(torch.as_tensor(genres[i] if isinstance(genres[i], int) else
                                                         genre_to_id[genres[i]]))

        # self.examples: list of tensors in following order:
        #   ids, attentionmask, sentencemap, goldstarts, goldends, clusterids, speaker ids, genre

    @staticmethod
    def pad_list(values, target_length, pad_value=0):
        """
        Function to pad a list of values to a specific length, appending the pad_value to the end of the list.

        Parameters
        ----------
        values: List
        target_length: int
        pad_value: value pad the list with

        Returns
        -------
        list of values, padded to target length
        """
        while len(values) < target_length:
            values.append(pad_value)
        return values

    @staticmethod
    def get_subtoken_data(token_data, offset_mapping):
        """
        Function to map token data to sub tokens. For example, if a token is split into two sub-tokens,
        the cluster id (or speaker id) for the token needs to be associated with both sub-tokens.

        Parameters
        ----------
        token_data: cluster ids of tokens
        offset_mapping: for each sub-token, a (start index, end index) tuple of indices into it's original token
            As returned by a transformers.tokenizer if return_offsets_mapping=True.

        Returns
        -------
        List containing cluster ids for each token
        """
        token_index = -1
        sub_token_data = []
        for (start, _) in offset_mapping:
            if start == 0:
                token_index += 1
            sub_token_data.append(token_data[token_index])
        return sub_token_data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """
        Each example is 1 document that consists of the following:
        input_ids: tensor of size (num_segments, sequence_length)
            input token ids
        attention_mask: tensor of size (num_segments, sequence_length)
            attention mask of input segments
        sentence_map: tensor of size (num_tokens)
            sentence id for each input token in input document
        speaker_ids: tensor of size (num_segments, sequence_length)
            speaker ids for each token (only used if self.use_metadata is True)
        genre: tensor of size (1)
            genre id for document
        gold_starts: tensor of size (labeled)
            start token indices (in flattened document) of labeled spans
        gold_ends: tensor of size (labeled)
            end token indices (in flattened document) of labeled spans
        cluster_ids: tensor of size (labeled)
            cluster ids of each labeled span
        """
        if self.use_metadata:
            ids, attention_mask, sentence_map, gold_starts, gold_ends, cluster_ids, speaker_ids, genre = \
                self.examples[i]
        else:
            ids, attention_mask, sentence_map, gold_starts, gold_ends, cluster_ids = self.examples[i]
        if len(ids) > self.max_segments:
            sentence_offset = random.randint(0, len(ids) - self.max_segments)
            token_offset = attention_mask[:sentence_offset].sum()
            ids = ids[sentence_offset:sentence_offset + self.max_segments]
            attention_mask = attention_mask[sentence_offset:sentence_offset + self.max_segments]
            num_tokens = attention_mask.sum()
            sentence_map = sentence_map[token_offset:token_offset + num_tokens]
            gold_spans = torch.logical_and(gold_ends >= token_offset, gold_starts < token_offset + num_tokens)
            gold_starts = gold_starts[gold_spans] - token_offset
            gold_ends = gold_ends[gold_spans] - token_offset
            cluster_ids = cluster_ids[gold_spans]
            if self.use_metadata:
                speaker_ids = speaker_ids[sentence_offset:sentence_offset + self.max_segments]
        inputs = {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "sentence_map": sentence_map,
            "gold_starts": gold_starts,
            "gold_ends": gold_ends,
            "cluster_ids": cluster_ids
        }
        if self.use_metadata:
            inputs.update({"speaker_ids": speaker_ids, "genre": genre})
        return inputs


class MangoesLineByLineDatasetForNSP(transformers.TextDatasetForNextSentencePrediction):
    """
    Subclass of Huggingface TextDatasetForNextSentencePrediction that supports multiple input files.
    Used for next sentence prediction task.

    Input file format:

    (1) One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of
    text.
    (2) Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task
    doesn't span between documents. Opening a new file will assume a new document as well.
    (3) Assumes different input files contain different documents. (ie, a document cannot span multiple files)


    Example:
    I am very happy. (new line) Here is the second sentence. (new line) (new line) A new document.

    Parameters
    ---------
    filenames: str or List[str]
        paths to files to include in dataset.
    tokenizer: transformers.Tokenizer
    short_seq_probability: float
        probability to sample shorter sequences. Used in the original BERT paper.
    nsp_probability: float
        probability to sample random next sentence for next sentence prediction task.
    max_len: int
        max length of input sequences, if None, will default to tokenizer.model_max_length
    encoding: str
        encoding of text files
    """
    def __init__(self, filenames, tokenizer, short_seq_probability=0.1, nsp_probability=0.5, max_len=None,
                 encoding=None):
        if isinstance(filenames, str):
            filenames = [filenames]
        max_len = max_len if max_len else tokenizer.model_max_length
        self.block_size = max_len - tokenizer.num_special_tokens_to_add(pair=True)
        self.short_seq_probability = short_seq_probability
        self.nsp_probability = nsp_probability
        self.encoding = encoding
        self.tokenizer = tokenizer
        self.documents = [[]]
        with fileinput.input(filenames, mode="r", openhook=None if not self.encoding else
                             fileinput.hook_encoded(self.encoding)) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                # Empty lines are used as document delimiters, or opening new file
                if (fileinput.isfirstline() or not line) and len(self.documents[-1]) != 0:
                    self.documents.append([])
                tokens = tokenizer.tokenize(line)
                tokens = tokenizer.convert_tokens_to_ids(tokens)
                if tokens:
                    self.documents[-1].append(tokens)
        self.examples = []
        for doc_index, document in enumerate(self.documents):
            self.create_examples_from_document(document, doc_index)


class MangoesMultipleChoiceDataset(Dataset):
    """
    Subclass of Torch Dataset for multiple choice datasets such as SWAG. Currently meant to work with BERT models.

    For information on how multiple choice datasets are formatted using this class, see
    https://github.com/google-research/bert/issues/38

    And this link for explanation of Huggingface's multiple choice models:
    https://github.com/huggingface/transformers/issues/7701#issuecomment-707149546

    Parameters
    ----------
    tokenizer: transformers.BertTokenizerFast
        tokenizer to use
    question_texts: List of str
        The texts corresponding to the questions/contexts.
    choices_texts: List of str
        The texts corresponding to the answer choices
    labels: List of int
        The indices of the correct answers
    max_seq_length:int
        The maximum total input sequence length after tokenization. if None, will default to tokenizer.model_max_length.
    """
    def __init__(self, tokenizer, question_texts, choices_texts, labels, max_seq_length=None):
        self.tokenizer = tokenizer
        self.question_texts = question_texts
        self.choices_texts = choices_texts
        self.labels = labels
        self.max_seq_len = max_seq_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        num_choices = len(self.choices_texts[i])
        questions = [self.question_texts[i]] * num_choices

        inputs = self.tokenizer(questions, self.choices_texts[i], padding="max_length", max_length=self.max_seq_len,
                                truncation=True, return_tensors="pt")
        inputs["labels"] = torch.as_tensor(int(self.labels[i]))
        return inputs


class MultipleLearnRateFineTuneTrainer(transformers.Trainer):
    """
    Subclass of Huggingface Trainer to accept different learning rates for base model parameters and task specific
    parameters, in the context of a fine-tuning task.

    Parameters
    ---------
    task_learn_rate: float
        Learning rate to be used for task specific parameters, (base parameters will use the normal, ie already defined
        in args, learn rate)
    base_keyword: str
        String to be used to differentiate base model and task specific parameters. All named parameters that have
        "base_keyword" somewhere in the name will be considered part of the base model, while all parameters that don't
        will be considered part of the task specific parameters.
    For documentation of the rest of the init parameters, see
        https://huggingface.co/transformers/main_classes/trainer.html#id1
    """
    def __init__(
        self,
        task_learn_rate,
        base_keyword="bert",
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None)
    ):
        self.task_learn_rate = task_learn_rate
        self.base_keyword = base_keyword
        super(MultipleLearnRateFineTuneTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset,
                                                               tokenizer, model_init, compute_metrics, callbacks,
                                                               optimizers)

    def create_optimizer_and_scheduler(self, num_training_steps):
        """
        Setup the optimizer and the learning rate scheduler.

        This will use AdamW. If you want to use something else (ie, a different optimizer and multiple learn rates), you
        can subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and
                               self.base_keyword in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and
                               self.base_keyword in n],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and
                               self.base_keyword not in n],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.task_learn_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and
                               self.base_keyword not in n],
                    "weight_decay": 0.0,
                    "lr": self.task_learn_rate,
                },
            ]
            if self.args.adafactor:
                optimizer_cls = transformers.Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = transformers.AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = transformers.get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )


class CoreferenceFineTuneTrainer(MultipleLearnRateFineTuneTrainer):
    """
    Subclass of the Mangoes MultipleLearnRateFineTuneTrainer that does not collate examples into batches, for use in the
    default Coreference Fine Tuning Trainer.

    This method introduces a dummy batch collation method, because the batches in the implemented fine tuning method
    (see paper below) are exactly 1 document each, and are pre-collated in the dataset class.
    This is based on the independent variant of the coreference resolution method described in
    https://arxiv.org/pdf/1908.09091.pdf.

    For documentation of the init parameters, see the documentation for MangoesMultipleLearnRateFineTuneTrainer
    """
    def __init__(
            self,
            task_learn_rate,
            base_keyword="bert",
            model=None,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None)
    ):
        def collate_fn(batch):
            return batch[0]
        super(CoreferenceFineTuneTrainer, self).__init__(task_learn_rate, base_keyword, model, args,
                                                         collate_fn, train_dataset, eval_dataset, tokenizer,
                                                         model_init, compute_metrics, callbacks, optimizers)


class IterableCompatibleTrainer(transformers.Trainer):
    """
    Subclass of Huggingface Trainer to accept Iterable datasets
    """

    def get_train_dataloader(self):
        """
        Returns
        -------
        torch DataLoader
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not isinstance(self.train_dataset, IterableDataset):
            train_sampler = self._get_train_sampler()
        else:
            train_sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """
        Parameters
        ----------
        eval_dataset: torch Dataset or IterableDataset

        Returns
        -------
        torch DataLoader
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if not isinstance(eval_dataset, IterableDataset):
            eval_sampler = self._get_eval_sampler(eval_dataset)
        else:
            eval_sampler = None
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )
