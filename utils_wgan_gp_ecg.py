import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

    # calculate critic scores
    mixed_scores = critic(interpolated_images)

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

def grid_plot_save(n_row, n_col, signal, path, f_name):
    idx = 0
    fig, ax = plt.subplots(n_row, n_col, sharex="col", sharey="row")
    for i in range(n_row):
        for j in range(n_col):
            if idx >= len(signal):
                break
            ax[i, j].plot(signal[idx])
            ax[i, j].grid()
            idx += 1

    fig.tight_layout()
    fig.savefig(os.path.join(path, f_name))
    plt.close()

def normalize(x):
    # Ming, Li, 2014, "Verification Based ECG Biometrics with ..."
    x_min = min(x)
    x_max = max(x)
    x_norm = [2 * (x[i] - (x_max + x_min)/2)/(x_max - x_min)  for i in  range(len(x))]
    return x_norm