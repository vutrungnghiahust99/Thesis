import torch
import torch.nn.functional as F

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Augmentation():
    @staticmethod
    def translation(x, shift: int):
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
    def add_guassian_noise(tensor, std, mean=0):
        return tensor + torch.randn(tensor.size()).type(Tensor) * std + mean
