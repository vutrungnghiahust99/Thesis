import torch

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
