import torch
import torch.nn as nn
from matplotlib import pyplot as plt


def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, L = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1)).repeat(1, C, L).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

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
    fig.savefig(path + f_name)
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
    x_norm = [2 * (x[i] - (x_max + x_min)/2)/(x_max - x_min)  for i in  range(len(x))]
    return x_norm

class net_cnn(nn.Module):
    """
        classifier network: Convolutional Network
        to be used in classification (main_classifier_ecg.py)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5)
        self.conv1x1 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5)
        self.fc1 = nn.Linear(57, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv1x1(x)

        # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class net_fc(nn.Module):
    """
        classifier network: Fully Connected Network
        to be used in classification (main_classifier_ecg.py)
    """
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
        self.leakyrelu = nn.LeakyReLU(.3)

    def forward(self, x):
        # 64x1x256
        x = self.leakyrelu(self.fc1(x))
        return x
