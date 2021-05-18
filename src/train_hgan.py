import argparse
import logging
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm

import torchvision.transforms as transforms
import torch

from src.models.hgan import Generator, Discriminator_V1
from src.fid_score.fid_model_hgan import ResNet18
from src.dataset import CIFAR10, CIFAR10_TEST
from src.losses import LSGAN as lsgan_loss
from src.losses import GAN1 as gan1_loss
from src.utils import sample_image
from src.utils import get_gen_real_imgs_with_headID, get_gen_mask_with_headID
from src.metrics import Metrics
from src.augmentation import Augmentation
from src.noise import Noise

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--loss_name", type=str, choices=['gan1'], required=True)

# augmentation
parser.add_argument("--augmentation", type=int, choices=[0, 1], default=0)
parser.add_argument("--aug_times", type=int, default=-1)

# input noise z
parser.add_argument("--dist", type=str, choices=['gauss', 'uniform'], required=True)
parser.add_argument("--bound", type=float, default=1)

# No. heads in the discriminator
parser.add_argument("--n_heads", type=int, required=True)

parser.add_argument("--diff_data_for_heads", type=int, choices=[0, 1], default=0)

parser.add_argument("--n_epochs", type=int)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--weights_g", type=str, default='')
parser.add_argument("--weights_d", type=str, default='')

parser.add_argument('--resnet18_weights', type=str, default='data/cifar10_64/cifar_resnet.pt')
parser.add_argument('--statistics', type=str, default='data/cifar10_64/test_data_statistics.p')
parser.add_argument('--training_path', type=str, default='data/cifar10_64/new_data/training.pt')
parser.add_argument('--test_path', type=str, default='data/cifar10_64/new_data/test.pt')

# unchanged configs
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
args = parser.parse_args()

# make exp_folder
exp_folder = f'experiments/hgan/augmentation/{args.exp_name}'
if not args.weights_g and not args.weights_d:
    os.makedirs(exp_folder, exist_ok=False)
    mode = 'w'
else:
    mode = 'a'

logging.basicConfig(filename=f'{exp_folder}/{args.exp_name}.txt',
                    filemode=mode,
                    format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

# logging general information
logging.info(f'\n***************** {args.exp_name.upper()} **************')
for k, v in args._get_kwargs():
    logging.info(f'{k}: {v}')
logging.info('--------------------------------')

logging.info(f'exp_folder: {exp_folder}')
images_folder = os.path.join(exp_folder, 'images')
models_folder = os.path.join(exp_folder, 'models')
os.makedirs(images_folder, exist_ok=True)
os.makedirs(models_folder, exist_ok=True)
logging.info(f'images_folder: {images_folder}')
logging.info(f'models_folder: {models_folder}')


generator = Generator()
assert args.loss_name == 'gan1'
discriminator = Discriminator_V1(n=args.n_heads)

logging.info(generator)
logging.info(discriminator)

# load weights if possible
start_epoch = 0
if args.weights_g:
    checkpoint = torch.load(args.weights_g)
    generator.load_state_dict(checkpoint)
    logging.info(f"loaded g at: {args.weights_g}")
    s = args.weights_g.split('/')[-1].split('.pth')[0].split('_')[-1]
    start_epoch = int(s) + 1

if args.weights_d:
    checkpoint = torch.load(args.weights_d)
    discriminator.load_state_dict(checkpoint)
    logging.info(f"loaded d at: {args.weights_d}")

logging.info(f'start at epoch: {start_epoch}')

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# Configure data loader
dataloader = torch.utils.data.DataLoader(
    CIFAR10(
        root=args.training_path,
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# for evaluate
real_imgs_loader = torch.utils.data.DataLoader(
    CIFAR10_TEST(
        root=args.test_path,
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    ),
    batch_size=1,
    shuffle=False,
)

resnet18 = ResNet18().eval()
if torch.cuda.is_available():
    checkpoint = torch.load(args.resnet18_weights)
    resnet18.cuda()
else:
    checkpoint = torch.load(args.resnet18_weights, map_location='cpu')
logging.info(f'Loaded resnet18 checkpoint at: {args.resnet18_weights}')
resnet18.load_state_dict(checkpoint['model_state'])

statistics = pickle.load(open(args.statistics, 'rb'))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = [
    torch.optim.Adam(discriminator.head_0.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_1.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_2.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_3.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_4.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_5.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_6.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
    torch.optim.Adam(discriminator.head_7.parameters(), lr=args.lr, betas=(args.b1, args.b2)),
]

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# loss functions
if args.loss_name == 'gan1':
    loss = gan1_loss
elif args.loss_name == 'lsgan':
    loss = lsgan_loss
else:
    RuntimeError(f'{args.loss_name} is not supported!!!')

# ----------
#  Training
# ----------

mapping = {
    0: Augmentation.aug0,
    1: Augmentation.aug1,
    2: Augmentation.aug2,
    3: Augmentation.aug3,
    4: Augmentation.aug4,
    5: Augmentation.aug5,
    6: Augmentation.aug6,
    7: Augmentation.aug7,
    8: Augmentation.aug8,
    9: Augmentation.aug9,
}

header = None
results = []
for epoch in range(start_epoch, args.n_epochs):
    logging.info(f'epoch: {epoch}')
    generator.train()
    discriminator.train()
    generator.requires_grad_(True)
    discriminator.requires_grad_(True)

    for (imgs, _, heads) in tqdm(dataloader):
        batch_size = imgs.shape[0]
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
        gen_imgs_ = generator(z)
        gen_imgs = {}
        lossg_total = 0
        random_seq = list(range(args.n_heads))
        random.shuffle(random_seq)
        for head_id in random_seq:
            s = gen_imgs_
            if args.diff_data_for_heads:
                mask = get_gen_mask_with_headID(heads, head_id)
                mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                s = s * mask
            if args.augmentation:
                s = torch.cat([s] + [mapping[head_id](s) for _ in range(args.aug_times)])
            lossg = loss.compute_lossg_rf(discriminator, s, head_id)
            lossg_total += lossg
        lossg_total.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for head_id in random_seq:
            optimizer_D[head_id].zero_grad()
            g = gen_imgs_.detach()
            r = real_imgs
            if args.diff_data_for_heads:
                g, r = get_gen_real_imgs_with_headID(g, r, heads, head_id)
            if args.augmentation:
                g = torch.cat([g] + [mapping[head_id](g) for _ in range(args.aug_times)])
                r = torch.cat([r] + [mapping[head_id](r) for _ in range(args.aug_times)])
            lossd = loss.compute_lossd_rf(discriminator, g, r, head_id)
            lossd.backward()
            optimizer_D[head_id].step()

    if epoch % args.interval != 0:
        continue

    generator.eval()
    discriminator.eval()
    generator.requires_grad_(False)
    discriminator.requires_grad_(False)

    # save checkpoint
    g_path = os.path.join(models_folder, f'generator_{epoch}.pth')
    d_path = os.path.join(models_folder, f'discriminator_{epoch}.pth')
    assert os.path.isfile(g_path) is False, f"Can not overwrite G at: {g_path}"
    assert os.path.isfile(d_path) is False, f"Can not overwrite D at: {d_path}"
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)
    logging.info(f'saved g at: {g_path}')
    logging.info(f'saved d at: {d_path}')
    sample_image(images_folder, args.bound, args.dist, generator, epoch)

    # statistic

    entry_0 = [epoch]
    header_0 = ['epoch']

    logging.info('lossg_mean, lossg_std, lossd_mean, lossd_std, dx_mean, dx_std, dgz_mean, dgz_std')
    lossg_mean, lossg_std, lossd_mean, lossd_std, dx_mean, dx_std, dgz_mean, dgz_std = \
        Metrics.compute_lossg_lossd_dx_dgz_rf(
            args.loss_name,
            generator,
            discriminator,
            real_imgs_loader,
            args.bound,
            args.dist,
            head_id=-1)  # compute the average of all heads
    entry_7 = [lossg_mean, lossg_std, lossd_mean, lossd_std, dx_mean, dx_std, dgz_mean, dgz_std]
    header_7 = ['lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std', 'dx_mean', 'dx_std', 'dgz_mean', 'dgz_std']
    logging.info(entry_7)

    logging.info('fid score')
    fid_score = Metrics.compute_fid_resnet18(generator, resnet18, statistics, args.dist, args.bound)
    entry_8 = [fid_score]
    header_8 = ['fid_score']
    logging.info(entry_8)
    print(fid_score)

    logging.info('fid score [10000]')
    fid_score = Metrics.compute_fid_resnet18(generator, resnet18, statistics, args.dist, args.bound, n=10000)
    entry_6 = [fid_score]
    header_6 = ['fid_score_10000']
    logging.info(entry_6)
    print(fid_score)

    lossg_mean__lossg_std__lossd_mean__lossd_std__dx_mean__dx_std__dgz_mean__dgz_std_all = \
        Metrics.compute_lossg_lossd_dx_dgz_rf_sep_heads(
            args.loss_name,
            generator,
            discriminator,
            real_imgs_loader,
            args.bound,
            args.dist,
            n_heads=args.n_heads)  # compute metrics for each head in D
    entry_9 = []
    header_9 = []
    for head_id in range(args.n_heads):
        h = [f'lossg_mean_{head_id}', f'lossg_std_{head_id}', f'lossd_mean_{head_id}', f'lossd_std_{head_id}',
             f'dx_mean_{head_id}', f'dx_std_{head_id}', f'dgz_mean_{head_id}', f'dgz_std_{head_id}']
        e = lossg_mean__lossg_std__lossd_mean__lossd_std__dx_mean__dx_std__dgz_mean__dgz_std_all[head_id]
        logging.info(h)
        logging.info(e)
        header_9 = header_9 + h
        entry_9 = entry_9 + e

    entry = entry_7 + entry_8 + entry_9 + entry_6
    if header is None:
        header = header_7 + header_8 + header_9 + header_6

    results.append(entry)
    logging.info('--------------------------------')

csv_path = f'{exp_folder}/{args.exp_name}.csv'
df = pd.DataFrame(results, columns=header)
df.to_csv(csv_path, index=None)
logging.info('Completed')
