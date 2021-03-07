# -*- coding: utf-8 -*-
"""Functions to visualize embeddings or some properties of Embeddings.

"""
import logging

import numpy as np
import matplotlib.pyplot as plt

import mangoes.evaluation.statistics

logger = logging.getLogger(__name__)


# ########################################
# t-SNE
def tsne(embeddings):
    """Create a 2d projections of the embeddings using t-SNE

    Parameters
    ----------
    embeddings: an instance of Embeddings
        Instance of :class:`mangoes.Embeddings` to project
    """

    try:
        matrix = embeddings.matrix.toarray()
    except AttributeError:
        # already dense
        matrix = embeddings.matrix

    import sklearn.manifold
    model = sklearn.manifold.TSNE()
    return model.fit_transform(matrix)


def plot_tsne(embeddings, words_to_display=None):
    """Plot a 2d projection of the embeddings using t-SNE

    Parameters
    ----------
    embeddings: an instance of Embeddings
        Instance of :class:`mangoes.Embeddings` to project
    words_to_display: list of str, optional
        list of words to display. If None (default), the first 100 words are displayed

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> mangoes.visualize.plot_tsne(embedding)
    >>> plt.show()
    """
    emb_2d = tsne(embeddings)

    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], linewidth=0)

    if not words_to_display:
        words_to_display = embeddings.words[:min(100, len(embeddings.words))]
    for word in words_to_display:
        index = embeddings.words.index(word)
        plt.annotate(word, (emb_2d[index, 0], emb_2d[index, 1]), xycoords='data',
                     xytext=(10, 0), textcoords='offset points',
                     size=15, va="center", color='black',
                     bbox=dict(boxstyle="round", alpha=0.3, fc="white", ec="none"))


def plot_distances(embeddings, ax, nb_intervals=8, title="Distances between words", scale="symlog"):
    """Plot an histogram of the distances between all the words of the Embeddings

    Notes
    -----
    Use projection='polar' in matplotlib to get a circular histogram

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = plt.subplot(111, projection='polar')
    >>> mangoes.visualize.plot_distances(embeddings, ax)

    Parameters
    ----------
    embeddings : an instance of Embeddings
        Instance of :class:`mangoes.Embeddings`
    ax :
        A matplotlib axes object
    nb_intervals : int, optional
        number of intervals (default : 8)
    title : str, optional
        title to use for the plot
    scale : {‘symlog’, ‘linear’, ‘log’, ‘logit’}
        scaling of the y-axis (default : symlog)
    """
    theta = np.linspace(0.0, np.pi, nb_intervals + 1, endpoint=True)
    ax.set_title(title)
    radii = mangoes.evaluation.statistics.distances_histogram(embeddings, theta)
    width = (np.pi / nb_intervals)
    bars = ax.bar(theta[:-1] + width/2, radii, width=width, bottom=0.0)
    ax.set_yscale(scale)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize('xx-small')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize('xx-small')

    for r, bar in zip(theta, bars):
        bar.set_facecolor(plt.cm.viridis(r / np.pi))


def plot_isotropy(embeddings, ax, title="Partition function value"):
    """Plot an histogram of the repartition of the values of a partition function

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = plt.subplot(111)
    >>> mangoes.visualize.plot_isotropy(embeddings, ax)

    Parameters
    ----------

    embeddings :
        Instance of :class:`mangoes.Embeddings`
    ax :
        A matplotlib axes object
    title : str, optional
        title to use for the plot
    """
    # TODO : works only for dense matrices
    concentration, mean_value, values = mangoes.evaluation.statistics.isotropy_from_partition_function(embeddings)

    ax.set_title(title)
    ax.hist(values / mean_value, bins=40, range=(0, 2))

    ax.axvline(0.9, linestyle='dashed')
    ax.axvline(1.1, linestyle='dashed')
    ax.text(1, ax.get_ylim()[1] * 0.95, "{:.2%}".format(concentration), horizontalalignment='center', va='center')
