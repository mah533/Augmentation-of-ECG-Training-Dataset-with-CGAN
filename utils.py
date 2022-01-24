import os

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe


def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    # calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def grid_plot_save(n_row, n_col, signal, path, f_name, all_classes, labels=None):
    idx = 0
    fig, ax = plt.subplots(n_row, n_col, sharex="col", sharey="row")
    for i in range(n_row):
        for j in range(n_col):
            if idx >= len(signal):
                break
            ax[i, j].plot(signal[idx])
            ax[i, j].grid()
            ax[i, j].set_title("cl: {}, {}".format(labels[idx], all_classes[labels[idx]]))
            idx += 1
    fig.tight_layout()
    fig.savefig(os.path.join(path, f_name))
    plt.close()


def grid_plot(n_row, n_col, signal):
    idx = 0
    fig, ax = plt.subplots(n_row, n_col, sharex="col", sharey="row")
    for i in range(n_row):
        for j in range(n_col):
            if idx >= len(signal):
                break
            ax[i, j].plot(signal[idx])
            ax[i, j].grid()
            # ax[i, j].set_title("cl: {}, {}".format(labels[idx], all_classes[labels[idx]]))
            idx += 1
    fig.tight_layout()
    plt.show()


def normalize(x):
    # Ming, Li, 2014, "Verification Based ECG Biometrics with ..."
    x_min = min(x)
    x_max = max(x)
    x_norm = [2 * (x[i] - (x_max + x_min) / 2) / (x_max - x_min) for i in range(len(x))]
    return x_norm


def check_accuracy(loader, model, train=True):
    # to be deleted, double check
    device = "cuda"
    if train:
        print("\nChecking accuracy on training data")
    else:
        print("\nChecking accuracy on test data")
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], 1, -1)

            scores = model(x)
            _, predictions = scores.squeeze().max(dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f'Got {num_correct}   /   {num_samples}   with accuracy  {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()
    # return acc


def print_confusion_matrix(confusion_matrix, class_names, fig_name=None, figsize=(10, 7), fontsize=14):
    # from: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Note that due to returning the created figure object, when this funciton is called in a
    notebook the figure willl be printed twice. To prevent this, either append ; to your
    function call, or modify the function by commenting out the return expression.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(fig_name, figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Note that due to returning the created figure object, when this funciton is called in a notebook
    # the figure willl be printed twice. To prevent this, either append ; to your function call, or
    # modify the function by commenting out this return expression.
    return fig

def stat_momemts(beats):
    # beat must be a dictionary
    beat_keys = beats.keys()
    beat_vals = beats.values()

    vals = [[] for _ in range(len(beat_keys))]
    mu1 = dict(zip(beat_keys, vals))

    vals = [[] for _ in range(len(beat_keys))]
    mu2 = dict(zip(beat_keys, vals))

    vals = [[] for _ in range(len(beat_keys))]
    mu3 = dict(zip(beat_keys, vals))

    vals = [[] for _ in range(len(beat_keys))]
    mu4 = dict(zip(beat_keys, vals))

    for cl in beat_keys:
        desc = describe(beats[cl], axis=0)
        mu1[cl].extend(desc.mean)
        mu2[cl].extend(desc.variance)
        mu3[cl].extend(desc.skewness)
        mu4[cl].extend(desc.kurtosis)

    return mu1, mu2, mu3, mu4

