[![coverage report](https://gitlab.inria.fr/magnet/mangoes/badges/master/coverage.svg)](https://gitlab.inria.fr/magnet/mangoes/commits/master)

# Mangoes

Mangoes is a toolbox for constructing and evaluating word vector representations (aka word embeddings). The main functionalities are:

* Process textual data and compute vocabularies and co-occurrence matrices. Input data should be raw text or annotated text. A companion preprocessing module is available.
* Compute word embeddings with different state-of-the art unsupervised methods.
* Propose statistical and intrinsic evaluation methods, as well as some visualization tools.
* Generate context dependent embeddings from a pretrained language model.
* Train a BERT language model
* Fine-tune a BERT language model

Over the next few releases, mangoes will be consolidated into a toolbox for injecting lexical  and  semantic  knowledge  into  word  and multimodal embeddings. 
Interfaces to external knowledge graphs such as wordnet will developed as well.


## Requirements
python >= 3.6

Mangoes depends on torch, transformers, numpy, and scipy. 
If they are not installed, they should automatically be during the installation of mangoes but, depending on your system, 
it could be easier to have them installed prior to installing mangoes.

To use the `mangoes.visualize` module, you will also need matplotlib that is **not installed** automatically. 

### Tips

Install `tqdm` to make loops show a progress meter 

## Install
```
pip install mangoes
```

**Think about using a virtual environment**

In order to avoid conflicting software versions, we advise you to use `virtualenv` to create an installation that is local to the project. 

## Usage
### Quick start

From corpus to word embeddings:
```python
import mangoes
import string

path_to_corpus = "notebooks/data/wiki_article_en"

corpus = mangoes.Corpus(path_to_corpus, lower=True, digit=True)
vocabulary = corpus.create_vocabulary(filters=[mangoes.corpus.remove_elements(string.punctuation)])
cooccurrences = mangoes.counting.count_cooccurrence(corpus, vocabulary)
embeddings = mangoes.create_representation(cooccurrences, 
                                           weighting=mangoes.weighting.PPMI(),
                                           reduction=mangoes.reduction.SVD(dimensions=200))
                                                       
print(embeddings.get_closest_words("september", 3))

# [('august', 0.0), ('july', 2.8723154258366064), ('january', 2.8723154258366064)]
```


Evaluation:
```python
import mangoes.evaluation.similarity
ws_result = mangoes.evaluation.similarity.Evaluation(embeddings, *mangoes.evaluation.similarity.ALL_DATASETS)
print(ws_result.get_report())

#                                                                          pearson        spearman
#                                                     Nb questions       (p-value)       (p-value)
# ================================================================================================
# similarity                                              379/6194  0.33(7.19e-09)  0.29(3.70e-07)
# ------------------------------------------------------------------------------------------------
#     rareword                                              9/2034  0.67(4.89e-02)  0.65(5.81e-02)
# ------------------------------------------------------------------------------------------------
#     mturk                                                 17/287  0.57(1.67e-02)  0.62(8.31e-03)
# ------------------------------------------------------------------------------------------------
#     men                                                 198/3000  0.62(3.78e-22)  0.64(7.24e-24)
# ------------------------------------------------------------------------------------------------
#     rg65                                                    2/65   1.0(0.00e+00)        1.0(nan)
# ------------------------------------------------------------------------------------------------
#     wordsim353                                            64/353  0.49(3.21e-05)  0.47(9.71e-05)
# ------------------------------------------------------------------------------------------------
#     ws353_similarity                                      37/203   0.6(8.22e-05)   0.5(1.73e-03)
# ------------------------------------------------------------------------------------------------
#     ws353_relatedness                                     52/252  0.47(4.67e-04)  0.43(1.68e-03)


```

If matplotlib is installed:
```python
import matplotlib.pyplot as plt
import mangoes.visualize

fig = plt.figure()
ax = plt.subplot(111, projection='polar')
mangoes.visualize.plot_distances(embeddings, ax)
plt.show()
```

Accessing pretrained BERT models, in this case the bert model from the original paper with the masked language modeling head:
```python
from mangoes.modeling import BERTForMaskedLanguageModeling
model = BERTForMaskedLanguageModeling.load("bert-base-uncased", "bert-base-uncased", device=None)
input_text = "This is a test sentence" # could also be a list of sentences
outputs = model.generate_outputs(input_text, pre_tokenized=False, output_hidden_states=True, output_attentions=False, word_embeddings=False)
# outputs is dict containing "hidden_states": hidden states of all layers for each token, as well as MLM logits. 
```


![](figure_1.png)


### Demos and tutorials
You can find more examples in the demo scripts: 

```
cd demo
python3 demo_embeddings_en.py
```

or in the notebooks:

```
jupyter notebook notebooks
```

or in the "Use cases" section of the documentation.

## Resources
You can download some [resources](https://gitlab.inria.fr/magnet/mangoes/wikis/resources) created with Mangoes

 
## Documentation
You can generate the documentation by running:
```
sphinx-build -b html docs/source docs/build
```

 
## Logging
If you want to activate logging in Mangoes, just define a logging configuration before you run the module.

For example, using `basicConfig`:

```python
import logging
import mangoes
logging.basicConfig(level=logging.INFO)

mangoes.Corpus("notebooks/data/wiki_article_en")
```
