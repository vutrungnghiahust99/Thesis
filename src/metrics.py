import numpy as np
from PIL import Image
from tqdm import tqdm
import scipy.linalg as sla
# import pickle

import torch

from src.noise import Noise
from src.losses import GAN1 as gan1_loss
from src.losses import LSGAN as lsgan_loss
from src.fid_score.fid_score import compute_fid_score

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Metrics():
    @staticmethod
    def compute_lossg_lossd_dx_dgz_rf(loss_name,
                                      generator,
                                      discriminator,
                                      real_imgs_loader,
                                      bound,
                                      dist,
                                      z_dim=100,
                                      head_id=-1):
        loss = select_loss(loss_name)
        lossgs = []
        lossds = []
        dxs = []
        dgzs = []
        for img, _ in tqdm(real_imgs_loader):
            real_img = img.type(Tensor)
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
            lossg = loss.compute_lossg_rf(discriminator, gen_img, head_id)
            lossgs.append(lossg.item())

            with torch.no_grad():
                real_pred = discriminator(real_img, head_id)
                fake_pred = discriminator(gen_img, head_id)
            dxs.append(real_pred.item())
            dgzs.append(fake_pred.item())

            lossd = loss.compute_lossd_rf(discriminator, gen_img, real_img, head_id)
            lossds.append(lossd.item())

        lossgs = np.array(lossgs)
        lossds = np.array(lossds)
        dxs = np.array(dxs)
        dgzs = np.array(dgzs)

        return lossgs.mean(), lossgs.std(), lossds.mean(), lossds.std(), dxs.mean(), dxs.std(), dgzs.mean(), dgzs.std()

    @staticmethod
    def compute_lossg_lossd_dx_dgz_rf_sep_heads(loss_name,
                                                generator,
                                                discriminator,
                                                real_imgs_loader,
                                                bound,
                                                dist,
                                                z_dim=100,
                                                n_heads=10):
        loss = select_loss(loss_name)
        out = []
        for head_id in range(n_heads):
            lossgs = []
            lossds = []
            dxs = []
            dgzs = []
            for img, _ in tqdm(real_imgs_loader):
                real_img = img.type(Tensor)
                z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
                with torch.no_grad():
                    gen_img = generator(z)
                lossg = loss.compute_lossg_rf(discriminator, gen_img, head_id)
                lossgs.append(lossg.item())

                with torch.no_grad():
                    real_pred = discriminator(real_img, head_id)
                    fake_pred = discriminator(gen_img, head_id)
                dxs.append(real_pred.item())
                dgzs.append(fake_pred.item())

                lossd = loss.compute_lossd_rf(discriminator, gen_img, real_img, head_id)
                lossds.append(lossd.item())

            lossgs = np.array(lossgs)
            lossds = np.array(lossds)
            dxs = np.array(dxs)
            dgzs = np.array(dgzs)
            out.append(
                [lossgs.mean(), lossgs.std(), lossds.mean(), lossds.std(), dxs.mean(), dxs.std(), dgzs.mean(), dgzs.std()]
            )
        return out

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


def select_loss(loss_name):
    if loss_name == 'lsgan':
        return lsgan_loss
    elif loss_name == 'gan1':
        return gan1_loss
    else:
        RuntimeError(f'{loss_name} is not supported!!')


def get_real_pil_images(path) -> [Image.Image]:
    dataset = torch.load(path)
    dataset = dataset[0]
    pil_images = []
    for i in range(dataset.shape[0]):
        pil_image = Image.fromarray(dataset[i, :, :].numpy())
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

    @staticmethod
    def computef_fid_resnet18(generator, resnet18, statistics, dist, bound, n=1000, batch_size=128, z_dim=100):
        z = Noise.sample_gauss_or_uniform_noise(dist, bound, n, z_dim)
        logits = []
        s = 0
        while(s < z.shape[0]):
            with torch.no_grad():
                x_gen = generator(z[s: s + batch_size])
                b = resnet18.forward(x_gen.cpu()).detach().numpy()
                logits.append(b)
            s += batch_size
        logits = np.concatenate(logits, axis=0)

        m = logits.mean(0)
        C = np.cov(logits, rowvar=False)

        fid_score = ((statistics['m'] - m) ** 2).sum() + np.matrix.trace(C + statistics['C'] - 2 * sla.sqrtm(np.matmul(C, statistics['C'])))
        return fid_score

# from src.models.hgan import Generator
# from src.fid_score.fid_model_hgan import ResNet18

# generator = Generator()
# resnet18 = ResNet18()
# statistics = pickle.load(open('data/cifar10_64/test_data_statistics.p', 'rb'))

# fid = Metrics.computef_fid_resnet18(generator, resnet18, statistics)
