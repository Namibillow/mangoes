Hyperparameters
===============

Both corpus preprocessing and word representation construction provide
different parameters that can be tuned with mangoes.

References :
LEVY, Omer, GOLDBERG, Yoav, et DAGAN, Ido. Improving distributional similarity with lessons learned from word
embeddings. Transactions of the Association for Computational Linguistics, 2015, vol. 3, p. 211-225.


.. |create_voc| replace:: :func:`.create_vocabulary`
.. |create_emb| replace:: :func:`.embeddings.create`
.. |count_cc| replace:: :func:`.count_cooccurrence`
.. |mred| replace:: :mod:`.reduction`
.. |mwght| replace:: :mod:`.weighting`


+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Description               | Params                | Values                    | Effect                            |
+===========================+=======================+===========================+===================================+
| .. _corpus-params:                                                                                                |
|                                                                                                                   |
| **CORPUS**                                                                                                        |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Text normalisation        | `lower`               | boolean (default = False) | Convert input corpus to lower case|
|                           +-----------------------+---------------------------+-----------------------------------+
|                           | `digit`               | boolean (default = False) | Replace all numeric values with 0 |
|                           +-----------------------+---------------------------+-----------------------------------+
|                           | `ignore_punctuation`  | boolean (default = False) | Ignore punctuation                |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| .. _counting-params:                                                                                              |
|                                                                                                                   |
| **COUNTING**                                                                                                      |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Vocabulary and features   | `words`               | a :class:`.Vocabulary`    | words to represent                |
| selection                 +-----------------------+---------------------------+-----------------------------------+
|                           | `context` or          | a :class:`.Vocabulary`    | words to use as features          |
|                           | `vocabulary` param of |                           |                                   |
|                           | the `context` param   |                           |                                   |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| If vocabulary is extracted from the corpus : :func:`.Corpus.create_vocabulary`                                    |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Vocabulary filters      | `filters`             | function (default = None) | Filter most or least frequent     |
|                           |                       |                           | words, remove punctuation, ...    |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Context definition        | `context`             | callable class (default = | from a sentence return the words  |
|                           |                       | :class:`.Window`)         | to be considered as co-occurring  |
|                           |                       |                           | for each word in the sentence     |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| If using window-like contexts : :class:`.context.Window`                                                          |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Size of the window      | `window_half_size`    | int (default = 1)         | size of the window                |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Fixed size or dynamic   | `dynamic`             | boolean (default = False) | Fixed size of window or random    |
| |                         |                       |                           | between 1 and `window_half_size`  |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Symmetric or asymmetric | `symmetric`           | boolean (default = True)  | The window can be centered        |
|                           |                       |                           | around a word or asymmetrical     |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Clean or dirty          | `dirty`               | boolean (default = False) | If dirty, remove ignored word     |
|                           |                       |                           | *before* creating the window      |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Subsampling               | `subsampling`         | boolean or float defining | Downsample the words more         |
|                           |                       | the threshold             | frequent than the threshold       |
|                           |                       | (default = False)         |                                   |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| .. _embedding-params:                                                                                             |
|                                                                                                                   |
| **EMBEDDING**                                                                                                     |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Transformations applied   | `transformations`     | list of functions         | Apply weighting and dimensionality|
| to the co-occurrence      |                       | (default = None)          | reduction to counts               |
| matrix                    |                       |                           |                                   |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| Dimension of the          | `dimensions`          | int                       | Size of the vectors               |
| vectors                   |                       |                           |                                   |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| If using PMI or variant                                                                                           |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Context Distribution    | `alpha`               | float (default = 1 for    | Raise context counts to the       |
| | Smoothing               |                       | not smoothed)             | power of alpha to "smooth" the    |
|                           |                       |                           | contexts’ distribution            |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Shift                   | `shift`               | int >= 1                  | Shift the matrix of log(shift)    |
|                           |                       | (default = 1 for no shift)|                                   |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| If using SVD (:func:`.svd`)                                                                                       |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Eigenvalue weighting    | `weight`              | int (default = 1)         | Weighting exponent to apply       |
|                           |                       |                           | to the eigenvalues                |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Add context vectors     | `add_context_vectors` | boolean (default = False) | Use the context vectors in        |
|                           |                       |                           | addition to the words vectors     |
+---------------------------+-----------------------+---------------------------+-----------------------------------+
| | Symmetric weighting     | `symmetric`           | boolean (default = False) | Way to compute the context vectors|
+---------------------------+-----------------------+---------------------------+-----------------------------------+
