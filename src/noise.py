import numpy as np

import torch

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Noise():
    @staticmethod
    def sample_gauss_or_uniform_noise(dist: str, bound: float, n_samples: int, z_dim: int):
        if dist == 'gauss':
            z = Tensor(np.random.normal(0, bound, (n_samples, z_dim)))
        elif dist == 'uniform':
            z = Tensor(np.random.uniform(-bound, bound, (n_samples, z_dim)))
        else:
            raise RuntimeError(f'{dist} is invalid!')
        return z
