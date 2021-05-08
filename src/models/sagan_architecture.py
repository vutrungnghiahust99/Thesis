import numpy as np

import torch.nn as nn

from src.models.spectral_normalization import SpectralNorm


class ConvTranspose2d(nn.Module):
    def __init__(self, use_spectral_norm: bool, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvTranspose2d, self).__init__()
        if use_spectral_norm:
            self.conv_transpose2d = SpectralNorm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv_transpose2d(x)


class Conv2d(nn.Module):
    def __init__(self, use_spectral_norm: bool, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(Conv2d, self).__init__()
        if use_spectral_norm:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv2d(x)


class Generator(nn.Module):
    """Generator."""

    def __init__(self, use_self_attention=False, image_size=28, z_dim=100, conv_dim=64, use_spectral_norm=False):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.use_self_attention = use_self_attention
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 2
        layer1.append(ConvTranspose2d(use_spectral_norm, z_dim, conv_dim * mult, 4, 1, 0))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(ConvTranspose2d(use_spectral_norm, curr_dim, int(curr_dim / 2), 4, 2, 1))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)
        layer3.append(ConvTranspose2d(use_spectral_norm, curr_dim, int(curr_dim / 2), 3, 2, 1))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)
        layer4.append(ConvTranspose2d(use_spectral_norm, curr_dim, int(curr_dim / 2), 2, 2, 1))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())
        
        curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        last.append(nn.Conv2d(curr_dim, 1, 3, 1, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        if self.use_self_attention:
            out = self.attn1(out)
        out = self.l4(out)
        if self.use_self_attention:
            out = self.attn2(out)
        out = self.last(out)

        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, use_sigmoid: bool, image_size=28, conv_dim=16, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        assert conv_dim == 16 or conv_dim == 64
        assert image_size == 28
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        layer4 = []
        last = []

        layer1.append(Conv2d(use_spectral_norm, 1, conv_dim, 4, 2, 1))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(Conv2d(use_spectral_norm, curr_dim, curr_dim * 2, 4, 2, 1))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(Conv2d(use_spectral_norm, curr_dim, curr_dim * 2, 4, 2, 1))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer4.append(Conv2d(use_spectral_norm, curr_dim, curr_dim * 2, 4, 2, 1))
        layer4.append(nn.LeakyReLU(0.1))
            
        curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        last.append(nn.Conv2d(curr_dim, 1, 1, 1, 0))
        if use_sigmoid:
            last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.last(out)

        return out.squeeze()

import torch
z = torch.randn(64, 100)
g = Generator()
d = Discriminator(use_sigmoid=True)
with torch.no_grad():
    gen_imgs = g(z)
with torch.no_grad():
    outs = d(gen_imgs)
