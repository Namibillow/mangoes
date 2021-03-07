# -*- coding: utf-8 -*-
"""
This demo script creates a representation from a tokenized corpus using methods based on a matrix built by counting
co-occurrences of words in a corpus, applying PPMI and reducing the matrix to 50 dimensions with SVD

We use as a corpus a sample of 750K words from wikipedia articles, from which we extract a vocabulary of 1500 words.
This vocabulary is used both as target words (represented as vectors) and as contexts.
The cooccurrence matrix is generated from the corpus considering a symmetric windows of 2 words around the target word.

"""
import logging
import os.path

import matplotlib.pyplot as plt
import pandas as pd

import mangoes
import mangoes.visualize
import mangoes.evaluation.analogy
import mangoes.evaluation.similarity
import mangoes.evaluation.outlier

#########################
# HYPERPARAMETERS

VOCABULARY_SIZE = 1500 # only 1500 unique words are considered
WINDOW_SIZE = 2  # symmetric window of 2 words before and 2 words after the target word
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output")


def main():
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Corpus
    corpus = get_demo_corpus()

    # Vocabulary
    vocabulary = get_vocabulary(corpus)

    ############################
    # COUNTING THE COOCCURRENCES
    logging.info("Counting co-occurrences ...")
    cooccurrence_path = os.path.join(OUTPUT_PATH,
                                     "cooccurrence_{}words_win{}".format(VOCABULARY_SIZE, WINDOW_SIZE))

    # Using the previously created vocabulary as the list of words to represent and as the contexts,
    # we count the co-occurrences and create a sparse matrix with these counts
    # Then we save the resulting object (vocabulary + matrix) : it could be loaded the next time

    window = mangoes.context.Window(size=WINDOW_SIZE, vocabulary=vocabulary)
    cooccurrence_matrix = mangoes.counting.count_cooccurrence(corpus, vocabulary, context=window)
    cooccurrence_matrix.save(cooccurrence_path)
    logging.info("Done. Cooccurrence counts and vocabulary saved in {}\n".format(cooccurrence_path))

    # You can also load a previously saved matrix by commenting the lines above and uncommenting the following
    # cooccurrence_matrix = mangoes.CountBasedRepresentation.load(cooccurrence_path)

    ############################
    # FROM COUNTING TO EMBEDDINGS
    logging.info("Creating the representation ...")
    embedding_path = os.path.join(OUTPUT_PATH,
                                  "embeddings/ppmi_svd_{}words_win{}".format(VOCABULARY_SIZE, WINDOW_SIZE))

    # From the cooccurrence counts, create vectors representing words in a mangoes.Embeddings objects,
    # applying ppmi and svd
    embedding = mangoes.create_representation(cooccurrence_matrix,
                                              weighting=mangoes.weighting.PPMI(),
                                              reduction=mangoes.reduction.SVD(dimensions=50, weight=0.5))
    embedding.save(embedding_path)
    logging.info("Done. Matrix and vocabulary saved in {}\n".format(embedding_path))

    # You can also load a previously saved embedding by commenting the lines above and uncommenting the following
    # embedding = mangoes.Embeddings.load(embedding_path)

    ##########################
    # EXPLORE THE EMBEDDINGS
    print("\n> EXPLORE")

    # 1. Closest words
    # you can have a list of the n closest words of a given word according to an embedding
    # To do so, use the get_closest_words() method
    # Here we will display the 3 closest words of 10 words of our vocabulary
    print("\n>> Closest words :")

    result = {word: pd.Series([w for w, _ in embedding.get_closest_words(word, nb=3)], index=[1, 2, 3])
              for word in embedding.words[100:200:10]}
    print(pd.DataFrame(result).transpose())

    # 2. Analogies
    print("\n>> Analogies :")
    # You can resolve analogy according to a representation using the analogy() method
    # Here, we will display the results of some examples :
    for analogy in ['great greater good', 'india indian japan', 'king queen man', 'london england paris']:
    # for analogy in ["ahuied udiwmed good"]:
        print(analogy, '->', embedding.analogy(analogy).using_cosadd[0]) # using_cossmul[0]


    ##########################
    # EVALUATE THE EMBEDDINGS

    print("\n> EVALUATION")

    # The package mangoes.evaluation provides different ways to evaluate a representation :

    # 1. Analogy task
    print("\n>> Analogy :")
    # the class mangoes.evaluation.analogy.Evaluation can run evaluation according to the available analogy datasets
    # (Google and MSR) and/or to your own and print a report
    analogy_evaluation = mangoes.evaluation.analogy.Evaluation(embedding, mangoes.evaluation.analogy.GOOGLE)
    print(analogy_evaluation.get_report(show_subsets=True))

    # 3. Similarity task
    print("\n>> Similarity :")
    # the class mangoes.evaluation.similarity.Evaluation can run evaluation according to the available word similarity
    # datasets (WS353, Rarreword, MEN, ...) and/or to your own and print a report
    similarity_evaluation = mangoes.evaluation.similarity.Evaluation(embedding, *mangoes.evaluation.similarity.ALL_DATASETS)
    print(similarity_evaluation.get_report())

    # 4. Outlier detection
    print("\n>> Outlier detection :")
    # the class mangoes.evaluation.outlier.Evaluation can run evaluation according to the available analogy datasets
    # (8-8-8 and WikiSem500) and/or to your own and print a report
    outlier_detection_evaluation = mangoes.evaluation.outlier.Evaluation(embedding,
                                                                         *mangoes.evaluation.outlier.ALL_DATASETS,
                                                                         lower=True)
    print(outlier_detection_evaluation.get_report())

    ##########################
    # VISUALIZE

    plt.figure()

    # 1. distances between the words
    ax = plt.subplot(221, projection='polar')
    mangoes.visualize.plot_distances(embedding, ax)

    # 2. isotropy
    ax = plt.subplot(222)
    mangoes.visualize.plot_isotropy(embedding, ax)

    # 3. t-sne
    plt.subplot(212)
    mangoes.visualize.plot_tsne(embedding)

    plt.show()


def get_demo_corpus():
    """
    Creates a mangoes.Corpus object from an example file

    Returns
    -------
    mangoes.Corpus
    """
    corpus_metadata = os.path.join(os.path.dirname(__file__), "output/.corpus")
    try:
        logging.info("Loading corpus metadata ...")
        corpus = mangoes.Corpus.load_from_metadata(corpus_metadata)
    except FileNotFoundError:
        logging.info("Counting corpus words and sentences ...")
        corpus_path = os.path.join(os.path.dirname(__file__), "data/sample_en_tokenized_750K.txt")
        corpus = mangoes.Corpus(corpus_path, nb_sentences=750000, lower=True)
        corpus.save_metadata(corpus_metadata)
    logging.info("Done. Corpus has {} sentences, {} different words, {} tokens".format(corpus.nb_sentences,
                                                                                       len(corpus.words_count),
                                                                                       corpus.size))
    return corpus


def get_vocabulary(corpus):
    """
    From a corpus, we extract a vocabulary (i.e. a set of words) with the most frequent words.
    Or load a previously created and saved vocabulary

    Parameters
    ----------
    corpus : mangoes.Corpus

    Returns
    -------
    mangoes.Vocabulary
    """

    vocabulary_file_name = "vocabulary_{}_words".format(VOCABULARY_SIZE)
    try:
        logging.info("Loading vocabulary ...")
        vocabulary = mangoes.Vocabulary.load("output", vocabulary_file_name)
    except FileNotFoundError:
        logging.info("Extracting vocabulary from corpus ...")
        import string
        remove_punctuation = mangoes.corpus.remove_elements(string.punctuation)
        vocabulary = corpus.create_vocabulary(filters=[remove_punctuation, mangoes.corpus.truncate(VOCABULARY_SIZE)])
        vocabulary.save(os.path.join(os.path.dirname(__file__), "output"), name=vocabulary_file_name)

    logging.info("Done.\n")
    return vocabulary


if __name__ == "__main__":
    exit(main())
