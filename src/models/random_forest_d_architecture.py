import numpy as np

import torch.nn as nn


from src.models.spectral_normalization import SpectralNorm


class Linear(nn.Module):
    def __init__(self, use_spectral_norm: bool, in_features: int, out_features: int):
        super(Linear, self).__init__()
        if use_spectral_norm:
            self.linear = SpectralNorm(nn.Linear(in_features, out_features))
        else:
            self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)


def create_d_head(use_sigmoid: bool, use_spec_norm: bool):
    if use_sigmoid:
        return nn.Sequential(Linear(use_spec_norm, 256, 1), nn.Sigmoid())
    else:
        return nn.Sequential(Linear(use_spec_norm, 256, 1))


def create_d_head_with_dropout(use_sigmoid: bool, use_spec_norm: bool):
    if use_sigmoid:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 1),
            nn.Dropout(),
            nn.Sigmoid())
    else:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 1),
            nn.Dropout())


def create_d_big_head(use_sigmoid: bool, use_spec_norm: bool):
    """
    Create a big head which has the capacity as 10 small heads
    """
    if use_sigmoid:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            Linear(use_spec_norm, 10, 1),
            nn.Sigmoid())
    else:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            Linear(use_spec_norm, 10, 1))


def create_d_big_head_with_dropout(use_sigmoid: bool, use_spec_norm: bool):
    """
    Create a big head which has the capacity as 10 small heads
    """
    if use_sigmoid:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            nn.Dropout(),
            Linear(use_spec_norm, 10, 1),
            nn.Sigmoid())
    else:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            nn.Dropout(),
            Linear(use_spec_norm, 10, 1))


class Discriminator(nn.Module):
    def __init__(self, use_sigmoid, use_spec_norm, img_shape=(1, 28, 28), n_heads=10, use_big_head_d=False, use_dropout=False):
        super(Discriminator, self).__init__()

        self.share_layers = nn.Sequential(
            Linear(use_spec_norm, int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            Linear(use_spec_norm, 512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.n = n_heads
        if not use_big_head_d:
            for head in range(self.n):
                if not use_dropout:
                    setattr(self, "head_%i" % head, create_d_head(use_sigmoid, use_spec_norm))
                else:
                    setattr(self, "head_%i" % head, create_d_head_with_dropout(use_sigmoid, use_spec_norm))
        else:
            assert n_heads == 1
            if not use_dropout:
                setattr(self, "head_%i" % 0, create_d_big_head(use_sigmoid, use_spec_norm))
            else:
                setattr(self, "head_%i" % 0, create_d_big_head_with_dropout(use_sigmoid, use_spec_norm))

    def forward(self, img, head_id):
        img_flat = img.view(img.size(0), -1)
        s1 = self.share_layers(img_flat)
        if head_id == -1:
            s = 0
            for i in range(self.n):
                s += getattr(self, "head_%i" % i)(s1)
            return (s / self.n).squeeze()
        elif head_id >= 0 and head_id < self.n:
            return getattr(self, "head_%i" % head_id)(s1).squeeze()
        else:
            RuntimeError(f"Invalid head id: {head_id}")
