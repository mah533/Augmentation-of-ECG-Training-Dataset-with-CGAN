



import torch
import torch.nn as nn


class Disc_ac_wgan_gp_1d(nn.Module):
    def __init__(self, channels_img, features_d, num_classes, img_size):
        super().__init__()
        self.img_size = img_size
        self.disc = nn.Sequential(
            # Input: 16 X 2 X 256
            nn.Conv1d(channels_img+1, features_d, kernel_size=4, stride=2, padding=1),  # 16 x 64 x 128
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),       # 16 x 128 x 64
            self._block(features_d * 2, features_d * 4, 4, 2, 1),   # 16 x 256 x 32
            self._block(features_d * 4, features_d * 8, 4, 2, 1),   # 16 X 512 X 16

            # After all _block img output is *x* (Conv1d below makes into *x*)
            nn.Conv1d(features_d * 8, 1, kernel_size=4, stride=2, padding=0, dilation=1),
            # 16 X 1 X 7
            nn.Linear(7, 1)         # 3 for ECG500 and 7 for MIT_BIH
        )
        self.embed = nn.Embedding(num_classes, img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,  # since BN is used, no need to bias
            ),
            nn.InstanceNorm1d(out_channels, affine=True),    # LayerNorm ←→ InstanceNorm
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        out = self.disc(x)
        return out


class Gen_ac_wgan_gp_1d(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, num_classes, img_size, embed_size,):
        super().__init__()
        self.img_size = img_size
        self.gen = nn.Sequential(
            # input: 64, 200, 1
            self._block(channels_noise+embed_size, features_g * 16, 4, 1, 0),  # 64, 1024, 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 64, 512, 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 64, 256, 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 64, 128, 32
            nn.ConvTranspose1d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # 64, 1, 64
            nn.Linear(64, 256, bias=False),     # 64x1x256 for MIT-BIH and 64x1x140 for ECG5000
            nn.Tanh(),  # [-1, 1]
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, labels):
        # latent vector z: N x noise_dim x 1
        temp = self.embed(labels)
        embedding = temp.unsqueeze(2)
        x = torch.cat([x, embedding], dim=1)
        out = self.gen(x)
        return out


def initialize_weights_1d(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

"""
def runn():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Disc_ac_wgan_gp(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)

    gen = Gen_ac_wgan_gp(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn(N, z_dim, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success!")


runn()
"""