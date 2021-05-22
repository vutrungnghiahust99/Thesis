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
        

class Generator(nn.Module):
    def __init__(self, use_spectral_norm=False, z_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()

        self.img_shape = img_shape

        def block(in_features, out_features, normalize=True):
            layers = [Linear(use_spectral_norm, in_features, out_features)]

            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim, 128, normalize=False),
            *block(128, 256, normalize=False),
            *block(256, 512, normalize=False),
            *block(512, 1024, normalize=False),
            Linear(use_spectral_norm, 1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


def create_normal_head(use_spec_norm, use_sigmoid):
    if use_sigmoid:
        return nn.Sequential(Linear(use_spec_norm, 256, 1), nn.Sigmoid())
    else:
        return Linear(use_spec_norm, 256, 1)


def create_big_head(use_spec_norm, use_sigmoid):
    if use_sigmoid:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            Linear(use_spec_norm, 10, 1),
            nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            Linear(use_spec_norm, 256, 10),
            Linear(use_spec_norm, 10, 1)
        )


class Discriminator(nn.Module):
    def __init__(self, use_sigmoid, m: int, use_big_head: bool, use_spec_norm=True, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()

        self.shared_layers = nn.Sequential(
            Linear(use_spec_norm, int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            Linear(use_spec_norm, 512, 256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.m = m
        if not use_big_head:
            for i in range(m):
                setattr(self, f"head_{i}", create_normal_head(use_spec_norm, use_sigmoid))
        else:
            assert self.n == 1
            setattr(self, f"head_{i}", create_big_head(use_spec_norm, use_sigmoid))

    def forward(self, img, head_id):
        img_flat = img.view(img.size(0), -1)
        img = self.shared_layers(img_flat)

        if head_id == -1:
            s = 0
            for i in range(self.m):
                s += getattr(self, f"head_{i}")(img)
            return (s / self.m).view(-1)

        assert head_id >= 0 and head_id < self.n
        p = getattr(self, f"head_{head_id}")(img)
        return p.view(-1)
