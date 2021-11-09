import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import ToPILImage

from src.noise import Noise
from src.fid_score.fid_score import compute_fid_score

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Metrics():
    @staticmethod
    def compute_lossg_lossd(loss,
                            generator,
                            discriminator,
                            real_imgs_loader,
                            bound,
                            dist,
                            z_dim=100):
        lossgs = []
        lossds = []
        for img, _ in tqdm(real_imgs_loader):
            real_img = img.type(Tensor)
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
                lossg = loss.compute_lossg(discriminator, gen_img)
            lossgs.append(lossg.item())

            with torch.no_grad():
                lossd = loss.compute_lossd(discriminator, gen_img, real_img)
            lossds.append(lossd.item())

        lossgs = np.array(lossgs)
        lossds = np.array(lossds)

        return lossgs.mean(), lossgs.std(), lossds.mean(), lossds.std()

    @staticmethod
    def compute_lossg_lossd_dx_dgz(loss,
                                   generator,
                                   discriminator,
                                   real_imgs_loader,
                                   bound,
                                   dist,
                                   z_dim=100):
        lossgs = []
        lossds = []
        dxs = []
        dgzs = []
        for img, _ in tqdm(real_imgs_loader):
            real_img = img.type(Tensor)
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
                lossg = loss.compute_lossg(discriminator, gen_img)
            lossgs.append(lossg.item())

            with torch.no_grad():
                dx = discriminator(real_img).squeeze()
                dgz = discriminator(gen_img).squeeze()
                lossd = loss.compute_lossd(discriminator, gen_img, real_img)
            lossds.append(lossd.item())
            dxs.append(dx.item())
            dgzs.append(dgz.item())

        lossgs = np.array(lossgs)
        lossds = np.array(lossds)
        dxs = np.array(dxs)
        dgzs = np.array(dgzs)

        return lossgs.mean(), lossgs.std(), lossds.mean(), lossds.std(), dxs.mean(), dxs.std(), dgzs.mean(), dgzs.std()

    @staticmethod
    def compute_heads_statistics(generator,
                                 discriminator,
                                 real_imgs_loader,
                                 bound,
                                 dist,
                                 z_dim=100):
        dx_mean = []
        dx_std = []
        dx_min = []
        dx_max = []
        dx_max_min = []

        dgz_mean = []
        dgz_std = []
        dgz_min = []
        dgz_max = []
        dgz_max_min = []
        for img, _ in tqdm(real_imgs_loader):
            real_img = img.type(Tensor)
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                mean, std, min, max, max_min = discriminator.eval_heads(real_img)
                dx_mean.append(mean)
                dx_std.append(std)
                dx_min.append(min)
                dx_max.append(max)
                dx_max_min.append(max_min)
                
                mean, std, min, max, max_min = discriminator.eval_heads(generator(z))
                dgz_mean.append(mean)
                dgz_std.append(std)
                dgz_min.append(min)
                dgz_max.append(max)
                dgz_max_min.append(max_min)
        return (
            (np.array(dx_mean).mean(), np.array(dx_std).mean(), np.array(dx_min).mean(), np.array(dx_max).mean(), np.array(dx_max_min).mean()),
            (np.array(dgz_mean).mean(), np.array(dgz_std).mean(), np.array(dgz_min).mean(), np.array(dgz_max).mean(), np.array(dgz_max_min).mean())
        )

    @staticmethod
    def compute_fid(generator,
                    dist,
                    bound,
                    path_to_real_imgs='data/mnist/MNIST/processed/test.pt',
                    z_dim=100) -> float:
        real_pil_imgs = get_real_pil_images(path_to_real_imgs)
        gen_pil_imgs = get_gen_pil_images(generator, bound, dist, len(real_pil_imgs))
        fid_score = compute_fid_score(gen_pil_imgs, real_pil_imgs)
        return fid_score


def get_real_pil_images(path) -> [Image.Image]:
    dataset = torch.load(path)
    dataset = dataset[0]
    pil_images = []
    for i in range(dataset.shape[0]):
        pil_image = Image.fromarray(dataset[i, :, :].numpy())
        pil_images.append(pil_image)
    return pil_images


def get_real_pil_images_v2(path) -> [Image.Image]:
    dataset = torch.load(path)
    dataset = dataset['data']
    pil_images = []
    for i in range(dataset.shape[0]):
        pil_image = Image.fromarray(dataset[i])
        pil_images.append(pil_image)
    return pil_images


def get_gen_pil_images_v2(generator,
                          bound,
                          dist,
                          n_samples,
                          z_dim=100) -> [Image.Image]:
    pil_images = []

    for i in range(n_samples):
        z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
        with torch.no_grad():
            img = generator(z).squeeze()
        img = img * 0.5 + 0.5
        pil_image = ToPILImage()(img)
        pil_images.append(pil_image)
    return pil_images


def get_gen_pil_images(generator,
                       bound,
                       dist,
                       n_samples,
                       z_dim=100,
                       img_shape=(1, 28, 28)) -> [Image.Image]:
    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    pil_images = []

    assert n_samples % 10 == 0
    imgs = []
    for i in range(10):
        z = Noise.sample_gauss_or_uniform_noise(dist, bound, n_samples // 10, z_dim)
        with torch.no_grad():
            imgs.append(generator(z))
    imgs = torch.cat(imgs, dim=0)

    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    arr = imgs.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    height = img_shape[1]
    width = img_shape[2]
    for i in range(arr.shape[0]):
        pil_img = Image.fromarray(arr[i, :, :, :].squeeze(), mode='L')
        if pil_img.size[0] != width or pil_img.size[1] != height:
            pil_img = pil_img.resize((width, height))
        pil_images.append(pil_img)
    return pil_images
