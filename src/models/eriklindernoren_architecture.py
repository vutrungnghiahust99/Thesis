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
    def __init__(self, use_spectral_norm, z_dim=100, img_shape=(1, 28, 28)):
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


class Discriminator(nn.Module):
    def __init__(self, use_sigmoid, use_spec_norm, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()

        if use_sigmoid:
            self.model = nn.Sequential(
                Linear(use_spec_norm, int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                Linear(use_spec_norm, 512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                Linear(use_spec_norm, 256, 1),
                nn.Sigmoid(),
            )
        else:
            self.model = nn.Sequential(
                Linear(use_spec_norm, int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                Linear(use_spec_norm, 512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                Linear(use_spec_norm, 256, 1),
                # nn.Sigmoid(),
            )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity.squeeze()
