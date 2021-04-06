import logging

import torch
from torchvision.utils import save_image

from src.noise import Noise


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def sample_image(images_folder, bound, dist, generator, epoch, n_row=10, z_dim=100):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    z = Noise.sample_gauss_or_uniform_noise(dist, bound, n_row**2, z_dim)
    with torch.no_grad():
        gen_imgs = generator(z)
    path = f"{images_folder}/{epoch}.png"
    save_image(gen_imgs, path, nrow=n_row, normalize=True)
    logging.info(f'saved image at: {path}')
