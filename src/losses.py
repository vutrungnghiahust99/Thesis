import numpy as np

import torch
import torch.autograd as autograd

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class LSGAN():
    adversarial_loss = torch.nn.MSELoss()

    @staticmethod
    def compute_lossg(discriminator, gen_imgs) -> float:
        """
        Args:
            gen_imgs ([tensor]): B x C x H x W
        """
        batch_size = gen_imgs.shape[0]
        real_preds = Tensor(batch_size).fill_(1.0)
        lossg = LSGAN.adversarial_loss(discriminator(gen_imgs).squeeze(), real_preds.squeeze())
        return lossg

    @staticmethod
    def compute_lossg_rf(discriminator, gen_imgs, head_id) -> float:
        """
        Args:
            gen_imgs ([tensor]): B x C x H x W
        """
        batch_size = gen_imgs.shape[0]
        real_preds = Tensor(batch_size).fill_(1.0)
        lossg = LSGAN.adversarial_loss(discriminator(gen_imgs, head_id).squeeze(), real_preds.squeeze())
        return lossg

    @staticmethod
    def compute_lossg_rf_eval(discriminator, gen_imgs) -> float:
        """
        Args:
            gen_imgs ([tensor]): B x C x H x W
        """
        batch_size = gen_imgs.shape[0]
        real_preds = Tensor(batch_size).fill_(1.0)
        lossg = LSGAN.adversarial_loss(discriminator(gen_imgs, -1).squeeze(), real_preds.squeeze())
        return lossg

    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs) -> float:
        """
        Args:
            gen_imgs (tensor): B x C x H x W
            real_imgs (tensor): B x C x H x W
        """
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs)
        real_loss = LSGAN.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs)
        fake_loss = LSGAN.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd

    @staticmethod
    def compute_lossd_rf(discriminator, gen_imgs, real_imgs, head_id) -> float:
        """
        Args:
            gen_imgs (tensor): B x C x H x W
            real_imgs (tensor): B x C x H x W
        """
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs, head_id)
        real_loss = LSGAN.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs, head_id)
        fake_loss = LSGAN.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd

    @staticmethod
    def compute_lossd_rf_eval(discriminator, gen_imgs, real_imgs) -> float:
        """
        Args:
            gen_imgs (tensor): B x C x H x W
            real_imgs (tensor): B x C x H x W
        """
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs, -1)
        real_loss = LSGAN.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs, -1)
        fake_loss = LSGAN.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd


class GAN1():
    adversarial_loss = torch.nn.BCELoss()
    
    @staticmethod
    def compute_lossg(discriminator, gen_imgs) -> float:
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        lossg = -1.0 * GAN1.adversarial_loss((real_labels - discriminator(gen_imgs)).squeeze(), real_labels.squeeze())
        return lossg
    
    @staticmethod
    def compute_lossg_rf(discriminator, gen_imgs, head_id) -> float:
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        lossg = -1.0 * GAN1.adversarial_loss((real_labels - discriminator(gen_imgs, head_id)).squeeze(), real_labels.squeeze())
        return lossg

    @staticmethod
    def compute_lossg_rf_eval(discriminator, gen_imgs) -> float:
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        lossg = -1.0 * GAN1.adversarial_loss((real_labels - discriminator(gen_imgs, -1)).squeeze(), real_labels.squeeze())
        return lossg

    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs):
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs)
        real_loss = GAN1.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs)
        fake_loss = GAN1.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd

    @staticmethod
    def compute_lossd_rf(discriminator, gen_imgs, real_imgs, head_id):
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs, head_id)
        real_loss = GAN1.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs, head_id)
        fake_loss = GAN1.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd

    @staticmethod
    def compute_lossd_rf_eval(discriminator, gen_imgs, real_imgs):
        assert gen_imgs.shape[0] == gen_imgs.shape[0]
        batch_size = gen_imgs.shape[0]
        real_labels = Tensor(batch_size).fill_(1.0)
        fake_labels = Tensor(batch_size).fill_(0.0)

        real_preds = discriminator(real_imgs, -1)
        real_loss = GAN1.adversarial_loss(real_preds.squeeze(), real_labels.squeeze())

        fake_preds = discriminator(gen_imgs, -1)
        fake_loss = GAN1.adversarial_loss(fake_preds.squeeze(), fake_labels.squeeze())

        lossd = (real_loss + fake_loss) / 2
        return lossd


class WGANGP():
    @staticmethod
    def compute_lossg(discriminator, gen_imgs) -> float:
        fake_preds = discriminator(gen_imgs)
        lossg = -torch.mean(fake_preds)
        return lossg
    
    @staticmethod
    def compute_lossd(discriminator, gen_imgs, real_imgs) -> float:
        fake_preds = discriminator(gen_imgs)
        real_preds = discriminator(real_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
        lossd = -torch.mean(real_preds) + torch.mean(fake_preds) + 10 * gradient_penalty
        return lossd


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Tensor(real_samples.shape[0], 1).fill_(1.0)
    # Get gradient w.r.t. interpolates
    # print(d_interpolates.shape, interpolates.shape)
    if len(d_interpolates.shape) == 0:
        d_interpolates = d_interpolates.unsqueeze(-1).unsqueeze(-1)
    else:
        d_interpolates = d_interpolates.unsqueeze(-1)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
