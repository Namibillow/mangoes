# -*-coding:utf-8 -*
"""Definitions a the context of a word in a sentence

This module defines classes to be used as `context` parameter in the
:func:`mangoes.counting.count_cooccurrence` function.

Examples
--------

    >>> window_3 = mangoes.context.Window(window_half_size=3)
    >>> contexts = window_3(['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
    >>> contexts[0]
    ['quick', 'brown', 'fox']
    >>> contexts[4]
    ['quick', 'brown', 'fox', 'over', 'the', 'lazy']
    >>> contexts[5]
    ['brown', 'fox', 'jumps', 'the', 'lazy', 'dog']
    >>> contexts[7]
    ['jumps', 'over', 'the', 'dog']

"""
import abc
import logging
import random
import warnings

import mangoes
import mangoes.utils.exceptions
from mangoes.utils import traversal

_logger = logging.getLogger(__name__)


class Context:
    """Base callable class to define the context of a word in a sentence

    Parameters
    ----------
    vocabulary: mangoes.Vocabulary or list of string
        vocabulary of the words to consider in the context. Other words are ignored.

    See Also
    --------
    :func:`mangoes.counting.count_cooccurrence`
    """

    def __init__(self, vocabulary=None):
        self._params = {}
        self.vocabulary = vocabulary
        try:
            self.filter_sentence = mangoes.vocabulary.create_tokens_filter(vocabulary.entity)
        except AttributeError:
            # no vocabulary or vocabulary has no attribute entity
            def no_filter(sentence):
                return sentence

            self.filter_sentence = no_filter

    @abc.abstractmethod
    def __call__(self, sentence, mask=False):
        """Returns the elements in the defined context for each word in sentence

        This function extracts the list of elements found in the context of the elements of a sentence

        Parameters
        ----------
        sentence: list
            The sentence as a list of words or tokens
        mask: list of bool
            List of booleans that indicates the positions in the sentence that can be ignored (value=False) :
            empty contexts will be returned for these words

        Returns
        -------
        list of the same size than sentence
        """
        pass

    @property
    def params(self):
        """Parameters of the context"""
        return self._params


class Sentence(Context):
    """Implements sentence context

     This context extracts the list of tokens in a sentence, around each word, barring the
     word itself.

    """

    def __init__(self, vocabulary=None):
        super().__init__(vocabulary)

    def __call__(self, sentence, mask=False):
        filtered_sentence = self.filter_sentence(sentence)
        return [[w for w in filtered_sentence[0:position] + filtered_sentence[position + 1:len(filtered_sentence)]
                 if not self.vocabulary or w in self.vocabulary]
                if not mask or mask[position] else [] for position in range(len(filtered_sentence))]


class Window(Context):
    """Implements window-type context

    This context extracts the list of elements found in a window defined around each word, barring the
    word itself.

    Examples
    --------

    >>> import mangoes.context
    >>> sentence = 'Beautiful is better than ugly'.split()
    >>> Window()(sentence)
    [['is'], ['Beautiful', 'better'], ['is', 'than'], ['better', 'ugly'], ['than']]
    >>> Window(size=2)(sentence) # doctest: +NORMALIZE_WHITESPACE
    [['is', 'better'],
     ['Beautiful', 'better', 'than'],
     ['Beautiful', 'is', 'than', 'ugly'],
     ['is', 'better', 'ugly'],
     ['better', 'than']]
    >>> Window(size=3, n_grams=2)(sentence) # doctest: +NORMALIZE_WHITESPACE
    [[('is', 'better'), ('better', 'than')],
     [('better', 'than'), ('than', 'ugly')],
     [('Beautiful', 'is'), ('than', 'ugly')],
     [('Beautiful', 'is'), ('is', 'better')],
     [('is', 'better'), ('better', 'than')]]
    >>> Window(size=2, distance=True)(sentence) # doctest: +NORMALIZE_WHITESPACE
     [[('is', 1), ('better', 2)],
      [('Beautiful', -1), ('better', 1), ('than', 2)],
      [('Beautiful', -2), ('is', -1), ('than', 1), ('ugly', 2)],
      [('is', -2), ('better', -1), ('ugly', 1)],
      [('better', -2), ('than', -1)]]

    Parameters
    ----------
    size: int for symmetric window, tuple of 2 int if not
        size of the search space to the left and to the right of each word (default = 1).
        If an integer is given, the window will be centered around the word; asymmetric if a couple of int

    dirty: boolean (default: False)
        if True and some of the words in the window aren't correct ids (ex : -1), they will not be fetched, but the
        window will be extended further so as to still be able to meet the quota of 2*'window_half_size' (if
        symmetric) words to fetch.

    dynamic: boolean
        if True, the size of the actual window is sampled between 1 and `window_half_size`

    n_grams: int (default: 1)
        if n_grams > 1, an element of the list will be a n-grams of words instead of a single word.

    distance: boolean (default: False)
        if True, the distance from the target word will be added

    symmetric: depricated. 

    window_half_size: depricated. 

    """

    def __init__(self, vocabulary=None, size=1, dirty=False, dynamic=False, n_grams=1, distance=False,
                 symmetric=None, window_half_size=None):
        super().__init__(vocabulary)
        if window_half_size or symmetric is not None:
            warnings.warn("'window_half_size' and 'symmetric' were replaced by the single parameter 'size' and"
                          "will be removed soon.", DeprecationWarning)
            # TODO: remove params in next release
            if size == 1:  # default
                size = window_half_size

        self._params.update({"size": size, "dirty": dirty,
                             "dynamic": dynamic, "n_grams": n_grams, "distance": distance})

        if isinstance(size, tuple):
            self.size_before, self.size_after = size
        else:
            self.size_before = self.size_after = size

        self._window = _create_window_method(self.size_before, self.size_after,
                                             dirty, dynamic, n_grams, distance)

    def __call__(self, sentence, mask=False):
        sentence = [w if not self.vocabulary or w in self.vocabulary else None for w in self.filter_sentence(sentence)]
        if not mask:
            mask = [True for _ in sentence]
        return self._window(sentence, mask)

    def call_on_encoded(self, encoded_sentence, mask=False):
        # TODO: find a better method name
        if not mask:
            mask = [True for _ in encoded_sentence]
        return self._window(encoded_sentence, mask)


def _create_window_method(size_before, size_after, dirty, dynamic, n, distance):
    """This function creates a "window" function according to the given parameters

    Returns
    -------
    _window : callable

    """
    return mangoes.utils.counting.get_window_function(size_before, size_after, dirty, dynamic, n, distance)


def _get_random_size(max_size_before, max_size_after):
    size_before = random.randint(1, max_size_before) if max_size_before > 0 else 0
    if max_size_after == max_size_before:
        size_after = size_before
    else:
        size_after = random.randint(1, max_size_after) if max_size_after > 0 else 0
    return size_before, size_after


class DependencyBasedContext(Context):
    """Implements Dependency-Based context

    Returns the modifiers and the head of each element of a sentence.

    Examples
    --------
    >>> source = mangoes.corpus.CONLLU(["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
    >>>                                 "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
    >>>                                 "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
    >>>                                 "4	star	star	NOUN	NN	_	3	dobj	_	_",
    >>>                                 "5	with	with	ADP	IN	_	6	case	_	_",
    >>>                                 "6	telescope	telescope	NOUN	NN	_	3	nmod	_	_"])
    >>> sentence = source.sentences().__next__()
    >>> context = mangoes.context.DependencyBasedContext(labels=True)
    >>> context(sentence)[1] # scientist
    {"australian/amod", "discovers/nsubj-"}

    References
    ----------
    .. [1] Levy, O., & Goldberg, Y. (2014, June). Dependency-Based Word Embeddings. In ACL (2) (pp. 302-308).

    Parameters
    ----------
    dependencies: {'universal-dependencies', 'stanford-dependencies'} or callable
        Representation used for dependencies annotation. Default is 'universal-dependencies'.
        You can also provide your own parser

    collapse: bool
        Whether or not the preposition relations should be collapsed. Default is False.

    labels: bool
        Whether or not the labels should be added to the output contexts. Default is False

    directed: bool 
        Whether or not the dependency relation should be considered as directed or undirected. Default is False.

    deprel_keep: tuple or set
        Contains valid dependency relation types existing in specified dependencies 
        Specify which dependency relation to count for creating a co-occurrence matrix. Default is None which means to include all 
        dependency relations.

    depth: int 
        Specify the length of relations to be considered. 1 means only cosinder direct dependency. > 2 consideres indirect dependency as well of that length. 
        Only positive integer is allowed. (<= 0 automatically resets to default value) Defaut is 1.   
        e.g. depth=3 includes the path legth of at most 3 distnce away from the target word. 

    weight: bool
        Whether of not to use the weight for building co-occurrng matrix. If True and no weight_scheme is not provided, then a fraction of size depth is taken. 
        e.g. if depth=1 then it assigns a value of 1, else, it assigns value of 1/|depth|.

    weight_scheme: dict 
        Where keys are existing dependency relations and values are the weight. Only applied if weight parameter is set to True. Default is None.


    """
    def __init__(self, vocabulary=None, entity="form", dependencies="universal-dependencies",
                 collapse=False, labels=False, depth=1, directed=False, deprel_keep=None, weight=False, weight_scheme=None):
        super().__init__(vocabulary)

        if labels :
            self.filter_vocabulary = self.vocabulary
            self.vocabulary = mangoes.vocabulary.DynamicVocabulary(dependency=True)
            self._format_context = self._format_context_with_label
        else:
            self.filter_vocabulary = self.vocabulary
            self._format_context = self._format_context_without_label

        self.entity = entity
        self.filter_token = mangoes.vocabulary.create_token_filter(entity)

        if weight_scheme and not weight:
            weight = True 
            
        if weight and weight_scheme and (not isinstance(weight_scheme, dict)):
            raise mangoes.utils.exceptions.UnsupportedType("dict is the only allowed type for weight_scheme")

        self._params.update({"parser": dependencies, "collapse": collapse, "labels": labels, 
                            "depth": depth, "directed": directed, "deprel_keep" : deprel_keep,
                            "weight":weight, "weight_scheme": weight_scheme
                            })

        if dependencies == "universal-dependencies":
            self.sentence_parser = self.ud_sentence_parser
        elif dependencies == "stanford-dependencies":
            self.sentence_parser = self.stanford_dependencies_sentence_parser
        elif callable(dependencies):
            self.sentence_parser = dependencies
        else:
            raise mangoes.utils.exceptions.NotAllowedValue(dependencies, ["universal-dependencies", "stanford-dependencies"])

    @property
    def collapse(self):
        return self._params["collapse"]

    @property
    def labels(self):
        return self._params["labels"]

    @property
    def depth(self):
        return self._params["depth"]

    @property
    def directed(self):
        return self._params["directed"]

    @property
    def deprel_keep(self):
        return self._params["deprel_keep"]

    @property 
    def weight(self):
        return self._params["weight"]

    @property 
    def weight_scheme(self):
        return self._params["weight_scheme"]

    def __call__(self, sentence, mask=False):
        dependency_tree = self.sentence_parser(sentence, self.collapse)

        if self.directed:
            for i in range(self.depth - 1):
                dependency_tree = self.add_children(dependency_tree, i+2)
        else:
            dependency_tree = self.add_length_path(dependency_tree, self.depth)

        contexts = [set() for _ in sentence]
        for i, token_dependencies in enumerate(dependency_tree):
            if token_dependencies:
                for target_id, label in token_dependencies:
                    labels = label.split("+")
                    weight = 1 
                    if self.weight_scheme:
                        weight = max([self.weight_scheme.get(l, 1) for l in labels]) 
                    elif self.weight:
                        weight = 1 / len(labels)

                    weight = str(weight)
                    # Only consider those relation only consisting specified dependency relation or connected by preposition
                    if not self.deprel_keep or all([(sub_label in self.deprel_keep or sub_label[:4] in ["case","prep"]) for sub_label in label.split("+")]):
                        target = self.filter_token(sentence[target_id])
                        if not self.filter_vocabulary or target in self.filter_vocabulary:
                            contexts[i].add(self._format_context(target, label, weight))

        return contexts

    def _format_context_with_label(self, token, label, weight):
        if isinstance(token, str):
            return token + "/" + label  + "/" + weight
        elif isinstance(token, tuple):
            return str(token._asdict()) + "/" + label + "/" + weight

    def _format_context_without_label(self, token, label, weight):
        if isinstance(token, str):
            return token + "/" + weight
        elif isinstance(token, tuple):
            return str(token._asdict()) + "/" + weight

    @staticmethod
    def add_children(sentence_tree, depth):
        new_sentence_tree = []
        for token_children in sentence_tree:
            new_children = set(token_children)
            for child, child_label in token_children:
                if sentence_tree[child]:
                    for grand_child, grand_child_label  in sentence_tree[child]:
                        new_label = child_label + '+' + grand_child_label
                        if len(new_label.split("+")) == depth:
                            new_children.add((grand_child, new_label))
            new_sentence_tree.append(new_children)
        return new_sentence_tree

    @staticmethod
    def add_length_path(sentence_tree, depth):
        new_sentence_tree = []

        parents = [set() for _ in range(len(sentence_tree))]
        visited = [set() for _ in range(len(sentence_tree))]

        for i, token_children in enumerate(sentence_tree):
            traversal.find_parents(0, depth, i, "", sentence_tree, parents[i], visited[i])
        
        # add undirectionaly 
        for i, token_children in enumerate(sentence_tree):
            for child, child_label in token_children:
                sentence_tree[child].add((i, child_label))

        for i, token_children in enumerate(sentence_tree): 
            new_children = set(token_children)

            for node, label, parent_far in parents[i]:
                traversal.dfs(depth-parent_far, node, label, visited[i], new_children, sentence_tree)

            new_sentence_tree.append(new_children)
        return new_sentence_tree

    @staticmethod
    def ud_sentence_parser(sentence, collapse=False):
        """Returns an adjacency list from a sentence annotated with Universal Dependencies

        Examples
        --------
        >>> source = mangoes.corpus.CONLLU(["1	australian	australian	ADJ	JJ	_	2	amod	_	_",
        >>>                                 "2	scientist	scientist	NOUN	NN	_	3	nsubj	_	_",
        >>>                                 "3	discovers	discover	VERB	VBZ	_	0	root	_	_",
        >>>                                 "4	star	star	NOUN	NN	_	3	dobj	_	_",
        >>>                                 "5	with	with	ADP	IN	_	6	case	_	_",
        >>>                                 "6	telescope	telescope	NOUN	NN	_	3	nmod	_	_"])
        >>> sentence = source.sentences().__next__()
        >>> mangoes.context.DependencyBasedContext.ud_sentence_parser(sentence)
        [set(),
         {(0, 'amod')},
         {(1, 'nsubj'), (3, 'dobj'), (5, 'nmod')},
         set(),
         set(),
         {(4, 'case')}]

        Parameters
        ----------
        sentence: list of Tokens

        collapse: boolean
            Whether or not to collapse prepositions

        Returns
        -------
        list of same size as sentence
            Returns the dependents of each token in the sentence.

        """
        root_label = "root"
        preposition_label = "case"
        preposition_object_label = "nmod"

        relations = [set() for _ in sentence]

        for token in sentence:
            # weight = depth
            try:
                if token.dependency_relation == root_label:
                    continue
                elif collapse and token.dependency_relation == preposition_label:
                    preposition_object_position = int(token.head) - 1
                    preposition_object = sentence[preposition_object_position]

                    head_position = int(preposition_object.head) - 1
                    relations[head_position].add(
                        (preposition_object_position, preposition_label + "_" + token.form))
                elif collapse and token.dependency_relation == preposition_object_label:
                    continue  # done in the previous case
                else:
                    token_position = int(token.id) - 1
                    head_position = int(token.head) - 1

                    relations[head_position].add((token_position, token.dependency_relation))
            except ValueError:
                logging.warning("Unable to parse token %s in sentence %s (ValueError)",
                                token, " ".join([t.form for t in sentence]))
                continue
            except IndexError:
                logging.warning("Unable to parse token %s in sentence %s (IndexError)",
                                token, " ".join([t.form for t in sentence]))
                continue

        return relations

    @staticmethod
    def stanford_dependencies_sentence_parser(sentence, collapse=False):
        """Returns an adjacency list from a sentence annotated with Stanford Dependencies

        Parameters
        ----------
        sentence: list of Tokens

        collapse: boolean
            Whether or not to collapse prepositions
        
        Returns
        -------
        list of same size as sentence
            Returns the dependents of each token in the sentence.

        """
        root_label = "root"
        preposition_label = "prep"
        preposition_object_label = "pobj"

        relations = [set() for _ in sentence]

        for token in sentence:
            # weight = depth 
            if token.dependency_relation == root_label:
                continue
            elif collapse and token.dependency_relation == preposition_label:
                continue
            elif collapse and token.dependency_relation == preposition_object_label:
                preposition = sentence[(int(token.head) - 1)]
                token_position = int(token.id) - 1
                head_position = int(preposition.head) - 1

                relations[head_position].add((token_position, preposition_label + "_" + preposition.form))
            else:
                token_position = int(token.id) - 1
                head_position = int(token.head) - 1

                relations[head_position].add((token_position, token.dependency_relation))

        return relations
