
import os
import pandas as pd

root = "experiments/rfgan/augmentation"

# header = [
#     'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std',
#     'fid_score'
# ]

header = [
    'lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std',
    'dx_mean', 'dx_std', 'dx_min', 'dx_max', 'dx_max_min',
    'dgz_mean', 'dgz_std', 'dgz_min', 'dgz_max', 'dgz_max_min',
    'fid_score',
]

INPUT = [
    'rfgan_lossgan1_bagging0_augmentation1_10heads_usemaskD',
    'rfgan_lossgan1_bagging1_augmentation1_10heads_usemaskD',
]


def get_numeric(s):
    s = s.split('\n')
    s = [x for x in s if '[' in x and ']' in x and 'accuracy' not in x and '/s' not in x and '_' not in x and (len(x) < 700)]
    result = []
    for line in s:
        line = line.split('[')[1]
        line = line.split(']')[0]
        line = line.split(',')
        line = [float(x) for x in line]
        result = result + line
    assert len(result) == len(header), f'{len(result)} != {len(header)}'
    return result[0:len(header)]

for inp in INPUT:
    int_file = os.path.join(root, inp, inp + '.txt')
    OUTPUT = os.path.join(root, inp, inp + '.csv')

    logs = open(int_file).read()
    logs = logs.split('--------------------------------')
    logs = logs[1:-1]
    results = [get_numeric(x) for x in logs]
    df = pd.DataFrame(results, columns=header)
    print(df.shape)
    df.to_csv(OUTPUT, index=None)
    print(OUTPUT)
    print('Completed')


# import torch

# from torchvision import transforms
# from src.augmentation import Augmentation

# from torchvision.utils import save_image


# def rand_cutout(x, ratio=0.4):
#     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
#     offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
#     offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
#     grid_batch, grid_x, grid_y = torch.meshgrid(
#         torch.arange(x.size(0), dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
#         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
#     )
#     grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
#     grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
#     mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
#     mask[grid_batch, grid_x, grid_y] = 0
#     x = x * mask.unsqueeze(1)
#     return x

# s = torch.load('batch.pth')
# s = torch.cat((s, s), dim=0)
# s = s[0:100]

# s1 = rand_cutout(s)

# save_image(s, 'origin.png', nrow=10, normalize=True)
# save_image(s1, 'augmentation.png', nrow=10, normalize=True)
