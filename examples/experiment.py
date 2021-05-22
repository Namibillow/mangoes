"""
Perform embedding creation using parameters specified in json file and save it.
"""

import mangoes
import nltk
import os 
import datetime 
import logging 
import argparse 
import json 
from pathlib import Path
import sys 
from collections import Counter 
import pprint 
from timeit import default_timer as timer
from datetime import timedelta

READER = {
    "TEXT" : mangoes.utils.reader.TextSentenceGenerator, 
    "BROWN" : mangoes.utils.reader.BrownSentenceGenerator,
    "XML" :  mangoes.utils.reader.XmlSentenceGenerator, 
    "CONLL" : mangoes.utils.reader.ConllSentenceGenerator,
    "CONLLU" : mangoes.utils.reader.ConllUSentenceGenerator
    }

class PrettyLog():
    def __init__(self, obj):
        self.obj = obj
    def __repr__(self):
        return pprint.pformat(self.obj)

def get_filters(config_filters):
    """
        Extract filters specified to build vocabulary.
    """

    filters = []

    for k, v in config_filters.items():
        if k == "stop_words":
            attribute = v if v else None 
            filters.append(mangoes.corpus.remove_elements(nltk.corpus.stopwords.words('english'), attribute=attribute))
        elif k == "truncate" : 
            filters.append(mangoes.corpus.truncate(v))
        elif k == "remove_least_frequent":
            filters.append(mangoes.corpus.remove_least_frequent(v))
        elif k == "remove_most_frequent":
            filters.append(mangoes.corpus.remove_most_frequent(v))
        elif k == "remove_elements":
            filters.append(mangoes.corpus.remove_elements()) 
        elif k ==  "filter_by_POS":
            filters.append(mangoes.corpus.filter_by_attribute(v["attribute"], set(v["keep"])))
    return filters 

def read_json(fname):
    """
        Load json file as dictionary.
    """

    with fname.open(mode='r') as handle:
        return json.load(handle)

def get_corpus(config):
    """
        Load savd .corpus file if any else read corpus to Corpus class and save it.
        Please specify correct path to saved .corpus with "saved_corpus_path" or path to corpus with "corpus_path" in json.
    """

    try:
        logging.info(f"Loading corpus metadata from '{config['saved_corpus_path']}'...")
        corpus = mangoes.Corpus.load_from_metadata(config["saved_corpus_path"])

    except (FileNotFoundError, KeyError):
        corpus_path = config["corpus_path"]
        logging.info(f"Corpus path: {corpus_path}")
        logging.info("Counting corpus words and sentences ...")

        corpus_config = config["parameters"]["corpus"]
        name = corpus_config.get('name', None)
        language = corpus_config.get("language", None) 
        reader = READER[corpus_config.get("reader", "TEXT")] 
        lower = corpus_config.get("lower", False)
        digit = corpus_config.get("digit", False) 
        ignore_punctuation = corpus_config.get("ignore_punctuation", False)
        nb_sentences = corpus_config.get("nb_sentences", None) 
        lazy= corpus_config.get("lazy", False) 
        max_len=config["parameters"].get("max_len", 100)

        corpus = mangoes.Corpus(content=corpus_path, 
                                name=name,
                                language=language,
                                reader=reader, 
                                lower=lower, 
                                digit=digit,
                                ignore_punctuation=ignore_punctuation,
                                nb_sentences=nb_sentences, 
                                lazy=lazy,
                                max_len=max_len)

        corpus_metadata = os.path.join(config["output_path"], ".corpus")
        corpus.save_metadata(corpus_metadata)
        logging.info(f"Saving the corpus to {config['output_path']}")

    logging.info("Done. Corpus has {} sentences, {} different words, {} tokens".format(corpus.nb_sentences,
                                                                                    len(corpus.words_count),
                                                                                    corpus.size))
    logging.info("Describe corpus: ")
    corpus.describe()

    return corpus

def get_vocabulary_util(corpus, vocab_config, name):
    """
        Helper function to build vocabullary from given corpus.
    """

    logging.info(f"Building {name} words...") 

    attributes = tuple(vocab_config["attributes"]) if "attributes" in vocab_config else None 
    filters = get_filters(vocab_config["filters"]) if "filters" in vocab_config else None 

    vocabulary = corpus.create_vocabulary(attributes=attributes, filters =filters)

    logging.info(f"Total {name} words: {len(vocabulary)}")
    if "POS" in vocabulary.entity:
        POS_count = Counter(getattr(token, "POS") for token in vocabulary._index_word)
        logging.info(f"Unique count of POS for {name} Words:")
        logging.info(PrettyLog(POS_count))
    
    return vocabulary

def get_vocabulary(corpus, config):
    """
        Build and save target and context vocabularies.
        Already vocabulary is saved, then provide its path to "saved_target_vocab_path"/"saved_context_vocab_path" in json.
    """

    try:
        target_vocabulary = mangoes.Vocabulary.load(*(config["saved_target_vocab_path"].rsplit('/', 1)))
        logging.info("Loaded target vocabulary.")

    except (FileNotFoundError, KeyError):
        target_vocabulary = get_vocabulary_util(corpus, config["parameters"]["target_vocabulary"], "target")
        target_vocabulary_file_name = "vocabulary_{}_target_words".format(len(target_vocabulary))
        target_vocabulary.save(config["output_path"], name=target_vocabulary_file_name)
        logging.info("Load and saved target vocabulary")

    try:
        context_vocabulary = mangoes.Vocabulary.load(*(config["saved_context_vocab_path"].rsplit('/', 1)))
        logging.info("Loaded context vocabulary.")

    except (FileNotFoundError, KeyError):
        context_vocabulary = get_vocabulary_util(corpus, config["parameters"]["context_vocabulary"], "context")
        context_vocabulary_file_name = "vocabulary_{}_context_words".format(len(context_vocabulary))
        context_vocabulary.save(config["output_path"], name=context_vocabulary_file_name)
        logging.info("Load and saved context vocabulary")

    
    return target_vocabulary, context_vocabulary

def get_dep_context(config, context_vocabulary):
    """
        Build dependency-based-context given a context vocabulary.
    """
    logging.info("Building dependency-based-context....")

    context_config = config["parameters"]["context_vocabulary"]
    entities = context_config.get("attributes", "form")
    
    dep_config = config["parameters"]["dependency_context"]
    deprel_keep= tuple(dep_config["deprel_keep"]) if "deprel_keep" in dep_config else None 

    return mangoes.context.DependencyBasedContext(
                                                vocabulary=context_vocabulary,
                                                entity=entities,
                                                dependencies=dep_config.get("dependencies", "universal-dependencies"),
                                                collapse = dep_config.get("collapse", False),
                                                labels = dep_config.get("labels", False),
                                                depth = dep_config.get("depth", 1),
                                                directed = dep_config.get("directed", False), 
                                                deprel_keep =deprel_keep,
                                                weight = dep_config.get("weight", False),
                                                weight_scheme = dep_config.get("weight_scheme", None)
                                                )

def get_embedding_params(config):
    """
        Get embedding specified weighting and/orreduction method from config if any. 
    """

    weighting = None 
    reduction = None 
    if "embedding" in config["parameters"]:
        weighting_dict = config["parameters"]["embedding"].get("weighting", None)
        if weighting_dict:
            if weighting_dict["type"] == "PPMI": 
                alpha = weighting_dict["parameters"].get("alpha", 1)
                weighting = mangoes.weighting.PPMI(alpha=alpha)
            elif weighting_dict["type"] == "PMI":
                alpha = weighting_dict["parameters"].get("alpha", 1)
                weighting = mangoes.weightign.PMI(alpha=alpha)
            elif weighting_dict["type"] == "ShiftedPPMI":
                alpha = weighting_dict["parameters"].get("alpha", 1)
                shift =  weighting_dict["parameters"].get("shift", 1)
                weighting = mangoes.weighting.ShiftedPPMI(alpha=alpha, shift=shift)
            elif weighting_dict["type"] == "TFIDF":
                weighting = mangoes.weighting.TFIDF()
            elif weighting_dict["type"] == "ProbabilitiesRatio":
                alpha = weighting_dict["parameters"].get("alpha", 1)
                weighting = mangoes.weighting.ProbabilitiesRatio(alpha=alpha)
            elif weighting_dict["type"] == "ConditionalProbabilities": 
                weighting = mangoes.weighting.ConditionalProbabilities()
            elif weighting_dict["type"] == "JointProbabilities":
                weighting = mangoes.weighting.JointProbabilities()

        reduction_dict = config["parameters"]["embedding"].get("reduction", None)
        if reduction_dict:
            if reduction_dict["type"] == "SVD":
                dimensions = reduction_dict["parameters"].get("dimension", 300)
                weight= reduction_dict["parameters"].get("weight", 1)
                add_context_vectors=reduction_dict["parameters"].get("add_context_vectors", False)
                symmetric= reduction_dict["parameters"].get("symmetric", False)
                reduction = mangoes.reduction.SVD(dimensions=dimensions, weight=weight, add_context_vectors=add_context_vectors, symmetric=symmetric)
            elif reduction_dict["type"] == "PCA":
                dimensions = reduction_dict["parameters"].get("dimension", 300)
                reduction = mangoes.reduction.SVD(dimensions=dimensions)

    return weighting, reduction 

def main(args):
    start = timer()

    config = read_json(Path(args.config))
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    output_path = os.path.join(os.path.abspath(''), "output/{}".format(date))
    if not os.path.exists(output_path):
        print("made a dir: ", output_path)
        os.makedirs(output_path)   

    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG, 
                        format="%(asctime)s;%(levelname)s;%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.FileHandler(f"{output_path}/log", "a+"),
                            logging.StreamHandler(sys.stdout)
                        ]
                    )

    config["output_path"] = output_path

    corpus = get_corpus(config) 
    target_vocabulary, context_vocabulary = get_vocabulary(corpus, config)

    dependency_context = get_dep_context(config, context_vocabulary)

    coocc_count = mangoes.counting.count_cooccurrence(corpus,  
                                                target_vocabulary, 
                                                context=dependency_context,
                                                max_len=config["parameters"].get("max_len", 100)
                                                )

    weighting, reduction = get_embedding_params(config)

    if weighting or reduction:
        logging.info("Creating embeddings...")
        embeddings = mangoes.create_representation(coocc_count, weighting=weighting, reduction=reduction)
    else:
        embeddings = coocc_count

    embedding_path = os.path.join(config["output_path"],
                              "embeddings/ppmi_svd_{}target_words_deprel".format(len(target_vocabulary)))

    embeddings.save(embedding_path)

    logging.info("Saved embedding. ")

    end = timer()
    logging.info(f"This experiment took : {timedelta(seconds=(end - start))}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train for Dependency based context")

    args.add_argument(
        "-c", 
        "--config", 
        type=str,
        default="parameters.json",
        help="config file path (default:None)"
        )

    args = args.parse_args()

    main(args)