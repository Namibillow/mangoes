import mangoes 
import nltk 
import logging 
import matplotlib.pyplot as plt
import mangoes.visualize

# UD_English_EWT = "../../UD_English-EWT-master/corpus/en_ewt-ud-test.conllu"
UD_English_EWT = "../../UD_English-EWT-master/corpus/"

def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("Counting corpus words and sentences ...")
    dependency_corpus = mangoes.Corpus(UD_English_EWT, 
                                        reader=mangoes.corpus.CONLLU, 
                                        language="English", 
                                        lower=True, 
                                        ignore_punctuation=False)

    logging.info("Done. Corpus has {} sentences, {} different words, {} tokens".format(dependency_corpus.nb_sentences,
                                                                                       len(dependency_corpus.words_count),
                                                                                       dependency_corpus.size))
                                                                 
    logging.info("Extracting vocabulary from corpus ...")
    stopwords_filter_lemma = mangoes.corpus.remove_elements(nltk.corpus.stopwords.words('english'), attribute="lemma")
    stopwords_filter = mangoes.corpus.remove_elements(nltk.corpus.stopwords.words('english'))

    target_vocabulary = dependency_corpus.create_vocabulary(attributes="lemma", 
                                                filters = [ stopwords_filter, 
                                                            mangoes.corpus.remove_most_frequent(100),
                                                            mangoes.corpus.remove_least_frequent(2)])

    context_vocabulary = dependency_corpus.create_vocabulary(attributes=("lemma","POS"), 
                                              filters = [ stopwords_filter_lemma])

    logging.info("Getting dependency based context")
    dependency_context = mangoes.context.DependencyBasedContext(entity=("lemma","POS"), 
                                                            labels=True,
                                                            collapse=True, 
                                                            vocabulary=context_vocabulary,
                                                            deprel_inverse=False)

 
    logging.info("Creating co-occurrence matirx")
    coocc_count = mangoes.counting.count_cooccurrence(dependency_corpus,  
                                                    target_vocabulary, 
                                                    context=dependency_context,
                                                    nb_workers=1)

    logging.info("Creating Embedding")
    ppmi = mangoes.weighting.PPMI()
    svd = mangoes.reduction.SVD(dimensions=300)
    embeddings = mangoes.create_representation(coocc_count, weighting=ppmi, reduction=svd)
    
    logging.info("Analogy Test ")
    for analogy in ["king queen male"]: # ans: female
        print(analogy, '->', embeddings.analogy(analogy, 3).using_cosadd)
        print(analogy, '->', embeddings.analogy(analogy).using_cosmul[0])

    logging.info("Visualizing ")
    plt.figure()

    # 1. distances between the words
    ax = plt.subplot(221, projection='polar')
    mangoes.visualize.plot_distances(embeddings, ax)

    # 2. isotropy
    ax = plt.subplot(222)
    mangoes.visualize.plot_isotropy(embeddings, ax)

    # 3. t-sne
    plt.subplot(212)
    mangoes.visualize.plot_tsne(embeddings)

    plt.show()

if __name__ == "__main__":

    main()
