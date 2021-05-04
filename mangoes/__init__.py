# -*- coding: utf-8 -*-
"""
Mangoes is a toolbox to produce and evaluate word embeddings.

Mangoes can be used to :

* Construct a words co-occurrence matrix from a Corpus
* Create Word Embeddings from this matrix
* Analyze a corpus (counting words, sentences, ...) and create vocabularies from it
* Evaluate word embeddings
* Generate context dependent embeddings from a pretrained language model.
* Train a BERT language model
* Fine-tune a BERT language model

"""

from mangoes.corpus import Corpus
from mangoes.dataset import Dataset
from mangoes.vocabulary import Vocabulary
from mangoes.base import Embeddings, CountBasedRepresentation, create_representation
import mangoes.modeling
import mangoes.context
import mangoes.counting
import mangoes.dataset
import mangoes.evaluate
import mangoes.reduction
import mangoes.weighting

__version__ = '2.0.1'
