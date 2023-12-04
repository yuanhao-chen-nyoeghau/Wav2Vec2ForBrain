import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.stats import describe
import scipy.cluster.hierarchy as shc
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd


def show_hist(
    data, xlabel: str, title: str, log_scale: bool = False, ylabel: str = "Frequency"
):
    if log_scale:
        data = np.log1p(data)

    # Create a histogram
    plt.hist(data, bins=30, density=True, alpha=0.7, color="blue", edgecolor="black")

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title + " converted to log-scale" if log_scale else "")

    plt.show()


def show_statistics(data):
    describe_result = describe(data)

    print(f"Summary")
    print(f"Min: {describe_result.minmax[0]}, Max: {describe_result.minmax[1]}")
    print(f"Mean: {describe_result.mean}")
    print(f"Var: {describe_result.variance}")


# Taken from https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def show_heatmap(
    data,
    title: str,
):
    fig, ax = plt.subplots()

    im, cbar = heatmap(
        data,
        range(0, data.shape[0]),
        range(0, data.shape[1]),
        ax=ax,
        cmap="YlGn",
        cbarlabel="Spike Power (mean of squared voltage)",
    )
    # texts = annotate_heatmap(im, valfmt="{x:.1f}")
    plt.title(title)
    plt.show()


def pca_most_valuable_features(pca, data_percentage):
    return np.where(pca.explained_variance_ratio_.cumsum() >= data_percentage)[0][0]


def show_clusters(data):
    # PCA components
    pca = PCA(n_components=128)
    pca.fit_transform(data)
    index_95 = pca_most_valuable_features(pca, 0.95)

    pca = PCA(n_components=index_95)
    pcs = pca.fit_transform(data)

    index_50 = pca_most_valuable_features(pca, 0.50)

    sns.pairplot(pd.DataFrame(pcs[:, :index_50]).sample(n=100000))
    plt.show()

    pcs_frame = pd.DataFrame(pcs[:, :index_95]).sample(10000)

    clusters = shc.linkage(pcs_frame, method="ward")
    shc.dendrogram(Z=clusters)
    plt.axhline(y=250000, color="r", linestyle="-")
    plt.show()

    num_cluster = 7
    agg_model = AgglomerativeClustering(
        n_clusters=num_cluster, affinity="euclidean", linkage="ward"
    )
    agg_model.fit(pcs_frame)

    pcs_frame[index_50 + 1] = agg_model.labels_

    sns.set(style="ticks", color_codes=True)
    sns.pairplot(
        pcs_frame.iloc[:, : index_50 + 2], hue=index_50 + 1, palette="husl", markers="."
    )
    plt.show()
