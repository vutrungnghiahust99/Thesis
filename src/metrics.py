import numpy as np
from PIL import Image
from tqdm import tqdm

import torch

from src.noise import Noise
from src.losses import GAN1 as gan1_loss
from src.losses import LSGAN as lsgan_loss
from src.fid_score.fid_score import compute_fid_score

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Metrics():
    @staticmethod
    def compute_l2norm_derivative_g_wrt_z(generator,
                                          bound,
                                          dist,
                                          n_samples=5000,
                                          z_dim=100,
                                          img_shape=(1, 28, 28)) -> np.array:
        def compute_jacobian(generator, z, z_dim, flatted_dim) -> torch.tensor:
            z1 = z.clone()
            z1 = z1.squeeze()
            z1 = z1.repeat(flatted_dim // 2, 1)
            z1.requires_grad_(True)
            y1 = generator(z1)
            t1 = y1.reshape(flatted_dim // 2, flatted_dim)
            if torch.cuda.is_available():
                t1.backward(torch.eye(flatted_dim)[0: flatted_dim // 2, :].cuda())
            else:
                t1.backward(torch.eye(flatted_dim)[0: flatted_dim // 2, :])

            z2 = z.clone()
            z2 = z2.squeeze()
            z2 = z2.repeat(flatted_dim - flatted_dim // 2, 1)
            z2.requires_grad_(True)
            y2 = generator(z2)
            t2 = y2.reshape(flatted_dim - flatted_dim // 2, flatted_dim)
            if torch.cuda.is_available():
                t2.backward(torch.eye(flatted_dim)[flatted_dim // 2:, :].cuda())
            else:
                t2.backward(torch.eye(flatted_dim)[flatted_dim // 2:, :])

            grad = torch.cat((z1.grad, z2.grad), dim=0)
            return grad

        l2norms = []
        flatted_dim = np.prod(img_shape)
        for _ in tqdm(range(n_samples)):
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            jacobian_matrix = compute_jacobian(generator, z, z_dim, flatted_dim)
            assert jacobian_matrix.shape[0] == flatted_dim
            assert jacobian_matrix.shape[1] == z_dim
            l2norm = jacobian_matrix.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)
        return l2norms

    @staticmethod
    def compute_l2norm_derivative_d_wrt_x(real_imgs_loader, discriminator):
        l2norms = []
        for r_img, _ in tqdm(real_imgs_loader):
            assert r_img.shape[0] == 1

            real_img = r_img.clone()
            real_img = real_img.type(Tensor)
            real_img.requires_grad_(True)
            real_pred = discriminator(real_img)
            # real_pred = real_pred.squeeze()
            real_pred.backward()
            
            l2norm = real_img.grad.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)
        return l2norms
    
    @staticmethod
    def compute_l2norm_derivative_d_wrt_x_gz(real_imgs_loader, discriminator, generator, bound, dist, n_samples=5000, z_dim=100):
        counter = 0
        l2norms = []
        for r_img, _ in tqdm(real_imgs_loader):
            assert r_img.shape[0] == 1
            if counter == 2500:
                break
            real_img = r_img.clone()
            real_img = real_img.type(Tensor)
            real_img.requires_grad_(True)
            real_pred = discriminator(real_img)
            real_pred.backward()
            
            l2norm = real_img.grad.norm()
            l2norms.append(l2norm.cpu().numpy())
            counter += 1

        for _ in tqdm(range(n_samples - counter)):
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
            gen_img.requires_grad_(True)
            gen_img = gen_img.type(Tensor)
            fake_pred = discriminator(gen_img)
            # fake_pred = fake_pred.squeeze()
            fake_pred.backward()
            
            l2norm = gen_img.grad.norm()
            l2norms.append(l2norm.cpu().numpy())

        l2norms = np.array(l2norms)
        return l2norms

    @staticmethod
    def compute_l2norm_derivative_d_wrt_gz(discriminator,
                                           generator,
                                           bound,
                                           dist,
                                           n_samples=5000,
                                           z_dim=100):
        l2norms = []
        for _ in tqdm(range(n_samples)):
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
            gen_img.requires_grad_(True)
            gen_img = gen_img.type(Tensor)
            fake_pred = discriminator(gen_img)
            # fake_pred = fake_pred.squeeze()
            fake_pred.backward()
            
            l2norm = gen_img.grad.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)
        return l2norms

    @staticmethod
    def compute_l2norm_derivative_lossg_wrt_z(loss_name,
                                              generator,
                                              discriminator,
                                              bound,
                                              dist,
                                              n_samples=5000,
                                              z_dim=100):
        loss = select_loss(loss_name)
        l2norms = []
        for _ in tqdm(range(n_samples)):
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            z.requires_grad_(True)
            gen_img = generator(z)
            lossg = loss.compute_lossg(discriminator, gen_img)
            lossg.backward()

            l2norm = z.grad.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)
        return l2norms

    @staticmethod
    def compute_lsnorm_derivative_lossg_wrt_theta_g(loss_name,
                                                    generator,
                                                    discriminator,
                                                    bound,
                                                    dist,
                                                    z_dim=100,
                                                    n_samples=5000):
        loss = select_loss(loss_name)
        generator.requires_grad_(True)
        discriminator.requires_grad_(True)

        l2norms = []
        optimizer_G = torch.optim.Adam(generator.parameters())
        optimizer_D = torch.optim.Adam(discriminator.parameters())
        for _ in tqdm(range(n_samples)):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            gen_img = generator(z)
            lossg = loss.compute_lossg(discriminator, gen_img)
            lossg.backward()
            s = torch.cat([x.grad.flatten() for x in generator.parameters() if x.grad is not None])
            l2norm = s.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)

        generator.requires_grad_(False)
        discriminator.requires_grad_(False)
        return l2norms

    @staticmethod
    def compute_norms_derivative_lossd_wrt_theta_d(loss_name,
                                                   generator,
                                                   discriminator,
                                                   real_imgs_loader,
                                                   bound,
                                                   dist,
                                                   z_dim=100):
        loss = select_loss(loss_name)
        generator.requires_grad_(True)
        discriminator.requires_grad_(True)
        optimizer_G = torch.optim.Adam(generator.parameters())
        optimizer_D = torch.optim.Adam(discriminator.parameters())
        l2norms = []
        for img, _ in tqdm(real_imgs_loader):
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            real_img = img.type(Tensor)
            z = Noise.sample_gauss_or_uniform_noise(dist, bound, 1, z_dim)
            with torch.no_grad():
                gen_img = generator(z)
            
            lossd = loss.compute_lossd(discriminator, gen_img, real_img)
            lossd.backward()
            s = torch.cat([x.grad.flatten() for x in discriminator.parameters() if x.grad is not None])
            l2norm = s.norm()
            l2norms.append(l2norm.cpu().numpy())
        l2norms = np.array(l2norms)

        generator.requires_grad_(False)
        discriminator.requires_grad_(False)
        return l2norms

    @staticmethod
    def compute_lossg_lossd_dx_dgz(loss_name,
                                   generator,
                                   discriminator,
                                   real_imgs_loader,
                                   bound,
                                   dist,
                                   z_dim=100):
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
            lossg = loss.compute_lossg(discriminator, gen_img)
            lossgs.append(lossg.item())

            with torch.no_grad():
                real_pred = discriminator(real_img)
                fake_pred = discriminator(gen_img)
            dxs.append(real_pred.item())
            dgzs.append(fake_pred.item())

            lossd = loss.compute_lossd(discriminator, gen_img, real_img)
            lossds.append(lossd.item())

        lossgs = np.array(lossgs)
        lossds = np.array(lossds)
        dxs = np.array(dxs)
        dgzs = np.array(dgzs)

        return lossgs.mean(), lossgs.std(), lossds.mean(), lossds.std(), dxs.mean(), dxs.std(), dgzs.mean(), dgzs.std()

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
