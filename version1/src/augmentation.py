import torch
import torch.nn.functional as F

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Augmentation():
    @staticmethod
    def translation(x, shift=4):
        """given img size H x W, this functions firstly pad zero to get image size (H+shift) x (H+shift), then crop size (HxW)
        """
        shift_x = shift
        shift_y = shift
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = grid_x + translation_x + shift
        grid_y = grid_y + translation_y + shift
        x_pad = F.pad(x, [shift, shift, shift, shift, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    @staticmethod
    def add_guassian_noise(tensor, std=0.2, mean=0):
        return tensor + torch.randn(tensor.size()).type(Tensor) * std + mean

    @staticmethod
    def rand_cutout(x, ratio=0.4):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x

    @staticmethod
    def rand_brightness(x):
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x

    @staticmethod
    def rand_saturation(x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x

    @staticmethod
    def rand_contrast(x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x

    @staticmethod
    def f1(x):
        return Augmentation.translation(x, shift=2)
    
    @staticmethod
    def f2(x):
        return Augmentation.translation(x, shift=4)

    @staticmethod
    def f3(x):
        return Augmentation.rand_cutout(x, ratio=0.2)

    def f4(x):
        return Augmentation.rand_cutout(x, ratio=0.4)

    @staticmethod
    def aug0(x):
        return Augmentation.f1(x)

    @staticmethod
    def aug1(x):
        return Augmentation.f2(x)

    @staticmethod
    def aug2(x):
        return Augmentation.f3(x)

    @staticmethod
    def aug3(x):
        return Augmentation.f4(x)

    @staticmethod
    def aug4(x):
        return Augmentation.f1(Augmentation.f3(x))

    @staticmethod
    def aug5(x):
        return Augmentation.f1(Augmentation.f4(x))

    @staticmethod
    def aug6(x):
        return Augmentation.f2(Augmentation.f3(x))

    @staticmethod
    def aug7(x):
        return Augmentation.f2(Augmentation.f4(x))

    @staticmethod
    def aug8(x):
        return Augmentation.f1(Augmentation.f1(x))

    @staticmethod
    def aug9(x):
        return Augmentation.f3(Augmentation.f3(x))
