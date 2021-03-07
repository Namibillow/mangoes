Mangoes documentation
=======================

Mangoes 2.0 is a toolbox for constructing and evaluating static and contextual token vector representations (aka embeddings). The main functionalities are:

Contextual embeddings:

* Access a large collection of pretrained transformer-based language models.
* Pre-train a BERT language model on a corpus.
* Fine-tune a BERT language model for a number of extrinsic tasks.
* Extract features/predictions from pretrained language models.

Static embeddings:

* Process textual data and compute vocabularies and co-occurrence matrices. Input data should be raw text or annotated text.
* Compute static word embeddings with different state-of-the art unsupervised methods.
* Propose statistical and intrinsic evaluation methods, as well as some visualization tools.
* Generate context dependent embeddings from a pretrained language model.


Future releases will include methods for injecting lexical and semantic knowledge into token and multi-model embeddings, and interfaces into common external knowledge resources.

MANGOES is developed as part of the IMPRESS (Improving Embeddings with Semantic Knowledge) project: https://project.inria.fr/impress/


Quickstart
==========

Contextual Language Models
--------------------------

Accessing pretrained BERT models, in this case the bert model from the original paper with the masked language modeling head:

   >>> from mangoes.modeling import BERTForMaskedLanguageModeling
   >>> model = BERTForMaskedLanguageModeling.load("bert-base-uncased", "bert-base-uncased", device=None)
   >>> input_text = "This is a test sentence" # could also be a list of sentences
   >>> outputs = model.generate_outputs(input_text, pre_tokenized=False, output_hidden_states=True, output_attentions=False, word_embeddings=False)
   >>> # outputs is dict containing "hidden_states": hidden states of all layers for each token, as well as MLM logits.

See more code examples in the contextual language model use cases.


Static Word Embeddings
----------------------

From corpus to word embeddings :

   >>> import mangoes
   >>> import string
   >>>
   >>> path_to_corpus = "notebooks/data/wiki_article_en"
   >>>
   >>> corpus = mangoes.Corpus(path_to_corpus, lower=True)
   >>> vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_elements(string.punctuation)])
   >>> cooccurrences = mangoes.counting.count_cooccurrence(corpus, vocabulary, vocabulary)
   >>> embeddings = mangoes.create_representation(cooccurrences,
   ...                                            weighting=mangoes.weighting.PPMI(),
   ...                                            reduction=mangoes.reduction.SVD(dimensions=200))
   >>> print(embeddings.get_closest_words("september", 3))
   [('august', 5.803186132007723e-15), ('attracting', 2.7974552300044038), ('july', 2.7974552300044038)]

Evaluation :

   >>> import mangoes.evaluation.similarity
   >>> ws_evaluation = mangoes.evaluation.similarity.Evaluation(embeddings, *mangoes.evaluation.similarity.ALL_DATASETS)
   >>> print(ws_evaluation.get_report())
                                                                              pearson       spearman
                                                          Nb questions        (p-val)        (p-val)
    ================================================================================================
    WS353                                                       32/353  -0.252(2e-01)  -0.158(4e-01)
    ------------------------------------------------------------------------------------------------
    WS353 relatedness                                           26/252  -0.317(1e-01) -0.0486(8e-01)
    ------------------------------------------------------------------------------------------------
    WS353 similarity                                            21/203  -0.137(6e-01)  -0.254(3e-01)
    ------------------------------------------------------------------------------------------------
    MEN                                                        32/3000   0.262(1e-01) -0.0312(9e-01)
    ------------------------------------------------------------------------------------------------
    M. Turk                                                     15/287 -0.0791(8e-01)    0.25(4e-01)
    ------------------------------------------------------------------------------------------------
    Rareword                                                   24/2034   0.452(3e-02)   0.407(5e-02)
    ------------------------------------------------------------------------------------------------
    RG65                                                          0/65       nan(nan)       nan(nan)
    ------------------------------------------------------------------------------------------------


See more code examples in the static word embeddings use cases.

Resources
=========
You can download some `resources <https://gitlab.inria.fr/magnet/mangoes/wikis/resources>`_ created with Mangoes


Documentation
=================
.. toctree::
   :maxdepth: 1

   use_cases_contextual/index
   use_cases_static/index
   parameters
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

