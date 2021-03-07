# -*- coding: utf-8 -*-
"""
This module provides a torch/transformers implementation of the fine-tuning procedure described in "BERT for Coreference
Resolution: Baselines and Analysis" (https://arxiv.org/pdf/1908.09091.pdf)
"""

import math

import transformers
import torch
import torch.nn.functional as F
import torch.nn as nn


class BertForCoreferenceResolutionBase(transformers.BertPreTrainedModel):
    """
    Class for fine-tuning a BERT model for the coreference resolution task.
    This is an implementation of https://arxiv.org/pdf/1908.09091.pdf, which uses the fine tuning procedure described in
    https://arxiv.org/pdf/1804.05392.pdf
    """
    def __init__(self, base_bert_config, max_span_width=30, ffnn_hidden_size=1000, top_span_ratio=0.4,
                 max_top_antecendents=50, use_metadata=False, metadata_feature_size=20,
                 genres=("bc", "bn", "mz", "nw", "pt", "tc", "wb"), max_training_segments=5, coref_depth=2,
                 coref_dropout=0.3):
        if base_bert_config.task_specific_params is None:
            # used for saving/loading model
            base_bert_config.task_specific_params = {"max_span_width": max_span_width,
                                                     "ffnn_hidden_size": ffnn_hidden_size,
                                                     "top_span_ratio": top_span_ratio,
                                                     "max_top_antecendents": max_top_antecendents,
                                                     "use_metadata": use_metadata,
                                                     "metadata_feature_size": metadata_feature_size, "genres": genres,
                                                     "max_training_segments": max_training_segments,
                                                     "coref_depth": coref_depth,
                                                     "coref_dropout": coref_dropout}

        else:
            # use config params if available
            max_span_width = base_bert_config.task_specific_params["max_span_width"] \
                if "max_span_width" in base_bert_config.task_specific_params else max_span_width
            ffnn_hidden_size = base_bert_config.task_specific_params["ffnn_hidden_size"] \
                if "ffnn_hidden_size" in base_bert_config.task_specific_params else ffnn_hidden_size
            top_span_ratio = base_bert_config.task_specific_params["top_span_ratio"] \
                if "top_span_ratio" in base_bert_config.task_specific_params else top_span_ratio
            max_top_antecendents = base_bert_config.task_specific_params["max_top_antecendents"] \
                if "max_top_antecendents" in base_bert_config.task_specific_params else max_top_antecendents
            use_metadata = base_bert_config.task_specific_params["use_metadata"] \
                if "use_metadata" in base_bert_config.task_specific_params else use_metadata
            metadata_feature_size = base_bert_config.task_specific_params["metadata_feature_size"] \
                if "metadata_feature_size" in base_bert_config.task_specific_params else metadata_feature_size
            genres = base_bert_config.task_specific_params["genres"] \
                if "genres" in base_bert_config.task_specific_params else genres
            max_training_segments = base_bert_config.task_specific_params["max_training_segments"] \
                if "max_training_segments" in base_bert_config.task_specific_params else max_training_segments
            coref_depth = base_bert_config.task_specific_params["coref_depth"] \
                if "coref_depth" in base_bert_config.task_specific_params else coref_depth
            coref_dropout = base_bert_config.task_specific_params["coref_dropout"] \
                if "coref_dropout" in base_bert_config.task_specific_params else coref_dropout
        super().__init__(base_bert_config)
        self.bert = transformers.BertModel(base_bert_config, add_pooling_layer=False)
        bert_emb_size = base_bert_config.hidden_size
        self.span_attend_projection = torch.nn.Linear(bert_emb_size, 1)
        span_embedding_dim = (bert_emb_size*3) + 20
        self.coref_dropout = coref_dropout
        self.mention_scorer = nn.Sequential(
            nn.Linear(span_embedding_dim, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.width_scores = nn.Sequential(
            nn.Linear(metadata_feature_size, ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.fast_antecedent_projection = torch.nn.Linear(span_embedding_dim, span_embedding_dim)
        self.slow_antecedent_scorer = nn.Sequential(
            nn.Linear((span_embedding_dim * 3) + (metadata_feature_size * (4 if use_metadata else 2)),
                      ffnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(coref_dropout),
            nn.Linear(ffnn_hidden_size, 1),
        )
        self.slow_antecedent_projection = torch.nn.Linear(span_embedding_dim * 2, span_embedding_dim)
        # metadata embeddings
        self.use_metadata = use_metadata
        self.genres = {g: i for i, g in enumerate(genres)}
        self.genre_embeddings = nn.Embedding(num_embeddings=len(self.genres), embedding_dim=metadata_feature_size)

        self.distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=metadata_feature_size)
        self.slow_distance_embeddings = nn.Embedding(num_embeddings=10, embedding_dim=metadata_feature_size)
        self.distance_projection = nn.Linear(metadata_feature_size, 1)
        self.same_speaker_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=metadata_feature_size)
        self.span_width_embeddings = nn.Embedding(num_embeddings=max_span_width, embedding_dim=metadata_feature_size)
        self.span_width_prior_embeddings = nn.Embedding(num_embeddings=max_span_width,
                                                        embedding_dim=metadata_feature_size)
        self.segment_dist_embeddings = nn.Embedding(num_embeddings=max_training_segments,
                                                    embedding_dim=metadata_feature_size)

        self.max_span_width = max_span_width
        self.top_span_ratio = top_span_ratio
        self.max_top_antecendents = max_top_antecendents
        self.max_training_segments = max_training_segments
        self.coref_depth = coref_depth

        self.init_weights()

    def forward(self, input_ids, attention_mask, sentence_map, speaker_ids=None, genre=None, gold_starts=None,
                gold_ends=None, cluster_ids=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Pass input document through model, calculating loss if labels are present.

        Parameters
        ----------
        input_ids: tensor of size (num_segments, sequence_length)
            input token ids
        attention_mask: tensor of size (num_segments, sequence_length)
            attention mask of input segments
        sentence_map: tensor of size (num_tokens)
            sentence id for each input token in (flattened) input document
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
        output_attentions: Boolean
            Whether or not to return the attentions tensors of all attention layers.
        output_hidden_states: Boolean
            Whether or not to return the hidden states of all layers.
        return_dict: Boolean
            Whether or not to return a ModelOutput (dictionary) instead of a plain tuple.

        Returns
        -------
        tuple containing the following tensors if return_dict is False, else dict with following keys:
            loss:
                loss value if label input arguments (gold_starts, gold_ends, cluster_ids) are not None, else not
                returned.
            candidate_starts: tensor of size (num_spans)
                start token indices in flattened document of candidate spans
            candidate_ends: tensor of size (num_spans)
                end token indices in flattened document of candidate spans
            candidate_mention_scores: tensor of size (num_spans)
                mention scores for each candidate span
            top_span_starts: tensor of size (num_top_spans)
                start token indices in flattened document of candidate spans with top mention scores
            top_span_ends: tensor of size (num_top_spans)
                end token indices in flattened document of candidate spans with top mention scores
            top_antecedents: tensor of shape (num_top_spans, antecedent_candidates)
                indices in top span candidates of top antecedents for each mention
            top_antecedent_scores: tensor of shape (num_top_spans, antecedent_candidates)
                final antecedent scores of top antecedents for each mention
            flattened_ids: tensor of shape (num_words)
                flattened ids of input sentences. The start and end candidate indices map into this tensor.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bert_outputs = self.bert(input_ids, attention_mask, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=return_dict)
        mention_doc = bert_outputs[0]   # [num_seg, max_seg_len, emb_len]
        num_seg, max_seg_len, emb_len = mention_doc.shape
        mention_doc = torch.masked_select(mention_doc.view(num_seg * max_seg_len, emb_len),
                                          attention_mask.bool().view(-1, 1)).view(-1, emb_len)  # [num_words, emb_len]
        flattened_ids = torch.masked_select(input_ids, attention_mask.bool()).view(-1)  # [num_words]
        num_words = mention_doc.shape[0]

        # calculate all possible spans
        candidate_starts = torch.arange(start=0,
                                        end=num_words,
                                        device=mention_doc.device).view(-1, 1) \
            .repeat(1, self.max_span_width)  # [num_words, max_span_width]
        candidate_ends = candidate_starts + torch.arange(start=0, end=self.max_span_width,
                                                         device=mention_doc.device).unsqueeze(
            0)  # [num_words, max_span_width]
        candidate_start_sentence_indices = sentence_map[candidate_starts]  # [num_words, max_span_width]
        candidate_end_sentence_indices = sentence_map[
            torch.clamp(candidate_ends, max=num_words - 1)]  # [num_words, max_span_width]
        # find spans that are in the same segment and don't run past the end of the text
        candidate_mask = torch.logical_and(candidate_ends < num_words,
                                           torch.eq(candidate_start_sentence_indices,
                                                    candidate_end_sentence_indices)).view(
            -1).bool()  # [num_words *max_span_width]
        candidate_starts = torch.masked_select(candidate_starts.view(-1), candidate_mask)  # [candidates]
        candidate_ends = torch.masked_select(candidate_ends.view(-1), candidate_mask)  # [candidates]

        if gold_ends is not None and gold_starts is not None and cluster_ids is not None:
            candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                              cluster_ids)  # [candidates]

        # get span embeddings and mention scores
        span_emb = self.get_span_embeddings(mention_doc, candidate_starts, candidate_ends)  # [candidates, span_emb]
        candidate_mention_scores = self.mention_scorer(span_emb).squeeze(1)  # [candidates]
        # get span width scores and add to candidate mention scores
        span_width_index = candidate_ends - candidate_starts    # [candidates]
        span_width_emb = self.span_width_prior_embeddings(span_width_index)     # [candidates, emb]
        candidate_mention_scores += self.width_scores(span_width_emb).squeeze(1)  # [candidates]

        # get beam size
        num_top_mentions = int(float(num_words * self.top_span_ratio))
        num_top_antecedents = min(self.max_top_antecendents, num_top_mentions)

        # get top mention scores and sort by sort by span order
        top_span_indices = torch.argsort(candidate_mention_scores, descending=True)[:num_top_mentions]
        top_span_indices, _ = torch.sort(top_span_indices, descending=False)
        top_span_starts = candidate_starts[top_span_indices]  # [top_cand]
        top_span_ends = candidate_ends[top_span_indices]  # [top_cand]
        top_span_emb = span_emb[top_span_indices]  # [top_cand, span_emb]
        top_span_mention_scores = candidate_mention_scores[top_span_indices]  # [top_cand]
        if gold_ends is not None and gold_starts is not None and cluster_ids is not None:
            top_span_cluster_ids = candidate_cluster_ids[top_span_indices]  # [top_cand]
        if self.use_metadata:
            genre_emb = self.genre_embeddings(genre)  # [meta_emb]
            speaker_ids = torch.masked_select(speaker_ids.view(num_seg * max_seg_len),
                                              attention_mask.bool().view(-1))  # [top_cand]
            top_span_speaker_ids = speaker_ids[top_span_starts]  # [top_cand]
        else:
            genre_emb = None
            top_span_speaker_ids = None

        # course to fine pruning
        dummy_scores = torch.zeros([num_top_mentions, 1], device=top_span_indices.device)  # [top_cand, 1]
        top_antecedents, \
            top_antecedents_mask, \
            top_antecedents_fast_scores, \
            top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb,
                                                                 top_span_mention_scores,
                                                                 num_top_antecedents)

        num_segments = input_ids.shape[0]
        segment_length = input_ids.shape[1]
        word_segments = torch.arange(start=0, end=num_segments, device=input_ids.device).view(-1, 1).repeat(
            [1, segment_length])  # [segments, segment_len]
        flat_word_segments = torch.masked_select(word_segments.view(-1), attention_mask.bool().view(-1))
        mention_segments = flat_word_segments[top_span_starts].view(-1, 1)  # [top_cand, 1]
        antecedent_segments = flat_word_segments[top_span_starts[top_antecedents]]  # [top_cand, top_ant]
        segment_distance = torch.clamp(mention_segments - antecedent_segments, 0,
                                       self.max_training_segments - 1)  # [top_cand, top_ant]

        # calculate final slow scores
        for i in range(self.coref_depth):
            top_antecedent_emb = top_span_emb[top_antecedents]  # [top_cand, top_ant, emb]
            top_antecedent_scores = top_antecedents_fast_scores + \
                self.get_slow_antecedent_scores(top_span_emb,
                                                top_antecedents,
                                                top_antecedent_emb,
                                                top_antecedent_offsets,
                                                top_span_speaker_ids,
                                                genre_emb,
                                                segment_distance)  # [top_cand, top_ant]
            top_antecedent_weights = F.softmax(
                torch.cat([dummy_scores, top_antecedent_scores], 1))  # [top_cand, top_ant + 1]
            top_antecedent_emb = torch.cat([top_span_emb.unsqueeze(1), top_antecedent_emb],
                                           1)  # [top_cand, top_ant + 1, emb]
            attended_span_emb = torch.sum(top_antecedent_weights.unsqueeze(2) * top_antecedent_emb,
                                          1)  # [top_cand, emb]
            gate_vectors = torch.sigmoid(
                self.slow_antecedent_projection(torch.cat([top_span_emb, attended_span_emb], 1)))  # [top_cand, emb]
            top_span_emb = gate_vectors * attended_span_emb + (1 - gate_vectors) * top_span_emb  # [top_cand, emb]

        top_antecedent_scores = torch.cat([dummy_scores, top_antecedent_scores], 1)  # [top_cand, top_ant + 1]

        # calculate loss if labels
        if gold_ends is not None and gold_starts is not None and cluster_ids is not None:
            top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]  # [top_cand, top_ant]
            top_antecedent_cluster_ids += torch.log(top_antecedents_mask.float()).int()  # [top_cand, top_ant]
            same_cluster_indicator = torch.eq(top_antecedent_cluster_ids,
                                              top_span_cluster_ids.unsqueeze(1))  # [top_cand, top_ant]
            non_dummy_indicator = (top_span_cluster_ids > 0).unsqueeze(1)  # [top_cand, 1]
            pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator)  # [top_cand, top_ant]
            dummy_labels = torch.logical_not(pairwise_labels.any(1, keepdims=True))  # [top_cand, 1]
            top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1)  # [top_cand, top_ant + 1]
            loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [top_cand]
            loss = torch.sum(loss)
        else:
            loss = None

        if not return_dict:
            output = (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                      top_antecedents, top_antecedent_scores)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "candidate_starts": candidate_starts,
            "candidate_ends": candidate_ends,
            "candidate_mention_scores": candidate_mention_scores,
            "top_span_starts": top_span_starts,
            "top_span_ends": top_span_ends,
            "top_antecedents": top_antecedents,
            "top_antecedent_scores": top_antecedent_scores,
            "flattened_ids": flattened_ids
        }

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb, segment_distance):
        """
        Compute slow antecedent scores

        Parameters
        ----------
        top_span_emb: tensor of size (candidates, emb_size)
            span representations
        top_antecedents: tensor of size (candidates, antecedents)
            indices of antecedents for each candidate
        top_antecedent_emb: tensor of size (candidates, antecedents, emb)
            embeddings of top antecedents for each candidate
        top_antecedent_offsets: tensor of size (candidates, antecedents)
            offsets for each mention/antecedent pair
        top_span_speaker_ids: tensor of size (candidates)
            speaker ids for each span
        genre_emb: tensor of size (feature_size)
            genre embedding for document
        segment_distance: tensor of size (candidates, antecedents)
            segment distances for each candidate antecedent pair

        Returns
        -------
        tensor of shape (candidates, antecedents)
            antecedent scores
        """
        num_cand, num_ant = top_antecedents.shape
        feature_emb_list = []

        if self.use_metadata:
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]  # [top_cand, top_ant]
            same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)  # [top_cand, top_ant]
            speaker_pair_emb = self.same_speaker_embeddings(
                torch.arange(start=0, end=2, device=top_span_emb.device))  # [2, feature_size]
            feature_emb_list.append(speaker_pair_emb[same_speaker.long()])
            genre_embs = genre_emb.view(1, 1, -1).repeat(num_cand, num_ant, 1)  # [top_cand, top_ant, feature_size]
            feature_emb_list.append(genre_embs)

        # span distance
        antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets).to(
            top_span_emb.device)  # [cand, cand]
        bucket_embeddings = self.slow_distance_embeddings(
            torch.arange(start=0, end=10, device=top_span_emb.device))  # [10, feature_size]
        feature_emb_list.append(bucket_embeddings[antecedent_distance_buckets])  # [cand, ant, feature_size]

        # segment distance
        segment_distance_emb = self.segment_dist_embeddings(
            torch.arange(start=0, end=self.max_training_segments, device=top_span_emb.device))
        feature_emb_list.append(segment_distance_emb[segment_distance])  # [cand, ant, feature_size]

        feature_emb = torch.cat(feature_emb_list, 2)  # [cand, ant, emb]
        feature_emb = F.dropout(feature_emb, p=self.coref_dropout,
                                training=self.training)  # [cand, ant, emb]
        target_emb = top_span_emb.unsqueeze(1)  # [cand, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [cand, ant, emb]
        target_emb = target_emb.repeat(1, num_ant, 1)  # [cand, ant, emb]

        pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [cand, ant, emb]

        return self.slow_antecedent_scorer(pair_emb).squeeze(-1)

    def coarse_to_fine_pruning(self, span_emb, mention_scores, num_top_antecedents):
        """
        Compute fast estimate antecedent scores and prune based on these scores.

        Parameters
        ----------
        span_emb: tensor of size (candidates, emb_size)
            span representations
        mention_scores: tensor of size (candidates)
            mention scores of spans
        num_top_antecedents: int
            number of antecedents

        Returns
        -------
        top_antecedents: tensor of shape (mentions, antecedent_candidates)
            indices of top antecedents for each mention
        top_antecedents_mask: tensor of shape (mentions, antecedent_candidates)
            boolean mask for antecedent candidates
        top_antecedents_fast_scores: tensor of shape (mentions, antecedent_candidates)
            fast scores for each antecedent candidate
        top_antecedent_offsets: tensor of shape (mentions, antecedent_candidates)
            offsets for each mention/antecedent pair
        """
        num_candidates = span_emb.shape[0]
        top_span_range = torch.arange(start=0, end=num_candidates, device=span_emb.device)
        antecedent_offsets = top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0)  # [cand, cand]
        antecedents_mask = antecedent_offsets >= 1  # [cand, cand]
        fast_antecedent_scores = mention_scores.unsqueeze(1) + mention_scores.unsqueeze(0)  # [cand, cand]
        fast_antecedent_scores += torch.log(antecedents_mask.float())  # [cand, cand]
        fast_antecedent_scores += self.get_fast_antecedent_scores(span_emb)  # [cand, cand]
        # add distance scores
        antecedent_distance_buckets = self.bucket_distance(antecedent_offsets).to(span_emb.device)  # [cand, cand]
        bucket_embeddings = F.dropout(self.distance_embeddings(torch.arange(start=0, end=10, device=span_emb.device)),
                                      p=self.coref_dropout, training=self.training)  # [10, feature_size]
        bucket_scores = self.distance_projection(bucket_embeddings)  # [10, 1]
        fast_antecedent_scores += bucket_scores[antecedent_distance_buckets].squeeze(-1)  # [cand, cand]
        # get top antecedent scores/features
        _, top_antecedents = torch.topk(fast_antecedent_scores, num_top_antecedents, sorted=False,
                                        dim=1)  # [cand, num_ant]
        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)  # [cand, num_ant]
        top_antecedents_fast_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)  # [cand, num_ant]
        top_antecedents_offsets = self.batch_gather(antecedent_offsets, top_antecedents)  # [cand, num_ant]
        return top_antecedents, top_antecedents_mask, top_antecedents_fast_scores, top_antecedents_offsets

    @staticmethod
    def batch_gather(emb, indices):
        batch_size, seq_len = emb.shape
        flattened_emb = emb.view(-1, 1)
        offset = (torch.arange(start=0, end=batch_size, device=indices.device) * seq_len).unsqueeze(1)
        return flattened_emb[indices + offset].squeeze(2)

    @staticmethod
    def get_candidate_labels(candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        """
        get labels of candidates from gold ground truth

        Parameters
        ----------
        candidate_starts, candidate_ends: tensor of size (candidates)
            start and end token indices (in flattened document) of candidate spans
        labeled_starts, labeled_ends: tensor of size (labeled)
            start and end token indices (in flattened document) of labeled spans
        labels: tensor of size (labeled)
            cluster ids

        Returns
        -------
        candidate_labels: tensor of size (candidates)
        """
        same_start = torch.eq(labeled_starts.unsqueeze(1),
                              candidate_starts.unsqueeze(0))  # [num_labeled, num_candidates]
        same_end = torch.eq(labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0))  # [num_labeled, num_candidates]
        same_span = torch.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        # type casting in next line is due to torch not supporting matrix multiplication for Long tensors
        candidate_labels = torch.mm(labels.unsqueeze(0).float(), same_span.float()).long()  # [1, num_candidates]
        return candidate_labels.squeeze(0)  # [num_candidates]

    @staticmethod
    def bucket_distance(distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].

        Parameters
        ----------
        distances: tensor of size (candidates, candidates)
            token distances between pairs

        Returns
        -------
        distance buckets
            tensor of size (candidates, candidates)
        """
        logspace_idx = torch.floor(torch.log(distances.float()) / math.log(2)).int() + 3
        use_identity = (distances <= 4).int()
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return torch.clamp(combined_idx, 0, 9)

    def get_fast_antecedent_scores(self, span_emb):
        """
        Obtains representations of the spans

        Parameters
        ----------
        span_emb: tensor of size (candidates, emb_size)
            span representations

        Returns
        -------
        fast antecedent scores
            tensor of size (candidates, span_embedding_size)
        """
        source_emb = F.dropout(self.fast_antecedent_projection(span_emb),
                               p=self.coref_dropout, training=self.training)  # [cand, emb]
        target_emb = F.dropout(span_emb, p=self.coref_dropout, training=self.training)  # [cand, emb]
        return torch.mm(source_emb, target_emb.t())  # [cand, cand]

    def get_span_embeddings(self, hidden_states, span_starts, span_ends):
        """
        Obtains representations of the spans

        Parameters
        ----------
        hidden_states: tensor of size (num_tokens, emb_size)
            outputs of BERT model, reshaped
        span_starts, span_ends: tensor of size (num_candidates)
            indices of starts and ends of spans

        Returns
        -------
        tensor of size (num_candidates, span_embedding_size)
        """
        emb = [hidden_states[span_starts], hidden_states[span_ends]]

        span_width = 1 + span_ends - span_starts  # [num_cand]
        span_width_index = span_width - 1  # [num_cand]
        span_width_emb = self.span_width_embeddings(span_width_index)  # [num_cand, emb]
        span_width_emb = F.dropout(span_width_emb, p=self.coref_dropout, training=self.training)
        emb.append(span_width_emb)

        token_attention_scores = self.get_span_word_attention_scores(hidden_states, span_starts,
                                                                     span_ends)  # [num_cand, num_words]
        attended_word_representations = torch.mm(token_attention_scores, hidden_states)  # [num_cand, emb_size]
        emb.append(attended_word_representations)
        return torch.cat(emb, dim=1)

    def get_span_word_attention_scores(self, hidden_states, span_starts, span_ends):
        """

        Parameters
        ----------
        hidden_states: tensor of size (num_tokens, emb_size)
            outputs of BERT model, reshaped
        span_starts, span_ends: tensor of size (num_candidates)
            indices of starts and ends of spans

        Returns
        -------
        tensor of size (num_candidates, span_embedding_size)
        """
        document_range = torch.arange(start=0, end=hidden_states.shape[0], device=hidden_states.device).unsqueeze(
            0).repeat(span_starts.shape[0], 1)  # [num_cand, num_words]
        token_mask = torch.logical_and(document_range >= span_starts.unsqueeze(1),
                                       document_range <= span_ends.unsqueeze(1))  # [num_cand, num_words]
        token_atten = self.span_attend_projection(hidden_states).squeeze(1).unsqueeze(0)  # [1, num_words]
        token_attn = F.softmax(torch.log(token_mask.float()) + token_atten, 1)  # [num_cand, num_words]span
        return token_attn

    @staticmethod
    def softmax_loss(top_antecedent_scores, top_antecedent_labels):
        """
        Calculate softmax loss

        Parameters
        ----------
        top_antecedent_scores: tensor of size [top_cand, top_ant + 1]
            scores of each antecedent for each mention candidate
        top_antecedent_labels: tensor of size [top_cand, top_ant + 1]
            labels for each antecedent

        Returns
        -------
        tensor of size (num_candidates)
            loss for each mention
        """
        gold_scores = top_antecedent_scores + torch.log(top_antecedent_labels.float())  # [top_cand, top_ant+1]
        marginalized_gold_scores = torch.logsumexp(gold_scores, 1)  # [top_cand]
        log_norm = torch.logsumexp(top_antecedent_scores, 1)  # [top_cand]
        return log_norm - marginalized_gold_scores  # [top_cand]
