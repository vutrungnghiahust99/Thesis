import logging

import torch
from torchvision.utils import save_image

from src.noise import Noise

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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


def get_frequent(s, a):
    count = 0
    for b in list(s):
        count += (b == a)
    return count


def get_gen_real_imgs_with_headID(gen_imgs, real_imgs, heads, head_id):
    gen = []
    real = []
    for i in range(len(heads)):
        count = get_frequent(heads[i], str(head_id))
        if count == 0:
            continue
        gen.append(gen_imgs[i].unsqueeze(0))
        real.append(real_imgs[i].unsqueeze(0))
    return torch.cat(gen), torch.cat(real)


def get_real_imgs_with_headID(real_imgs, heads, head_id):
    real = []
    for i in range(len(heads)):
        count = get_frequent(heads[i], str(head_id))
        if count == 0:
            continue
        real.append(real_imgs[i].unsqueeze(0))
    return torch.cat(real)


def get_gen_mask_with_headID(heads, head_id):
    mask = []
    for i in range(len(heads)):
        count = get_frequent(heads[i], str(head_id))
        mask.append(count != 0)
    return torch.tensor(mask).type(Tensor)
