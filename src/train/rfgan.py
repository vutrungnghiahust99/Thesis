import argparse
import logging
import os
import random
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision import datasets
import torch

from src.models.eriklindernoren import Generator, Discriminator
from src.losses.gan1 import RFLoss as gan1_loss
from src.losses.lsgan import RFLoss as lsgan_loss
from src.dataset import MNIST
from src.utils import sample_image
from src.utils import get_real_imgs_with_headID
from src.metrics import Metrics
from src.augmentation import Augmentation
from src.noise import Noise

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--loss_name", type=str, choices=['gan1', 'lsgan'], required=True)

parser.add_argument('--use_mask_d', type=int, choices=[0, 1], default=0)

# augmentation
parser.add_argument("--augmentation", type=int, choices=[0, 1], default=0)
parser.add_argument("--aug_times", type=int, default=-1)

# No. heads in the discriminator
parser.add_argument("--n_heads", type=int, required=True)

parser.add_argument("--bagging", type=int, choices=[0, 1], default=0)

parser.add_argument("--n_epochs", type=int)
parser.add_argument("--interval", type=int, default=1)

parser.add_argument("--weights_g", type=str, default='')
parser.add_argument("--weights_d", type=str, default='')

# unchanged configs
parser.add_argument("--dist", type=str, choices=['gauss', 'uniform'], default='gauss')
parser.add_argument("--bound", type=float, default=1)
parser.add_argument('--data_path', type=str, default='data/mnist')
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--z_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
args = parser.parse_args()

# make exp_folder
if args.augmentation:
    exp_folder = f'experiments/rfgan/augmentation/{args.exp_name}'
else:
    exp_folder = f'experiments/rfgan/{args.exp_name}'
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


# Initialize generator and discriminator

generator = Generator()
if args.loss_name == 'gan1':
    loss = gan1_loss
    discriminator = Discriminator(
        use_sigmoid=True,
        m=args.n_heads,
        use_big_head=False,
        use_mask_d=args.use_mask_d)
else:
    loss = lsgan_loss
    discriminator = Discriminator(
        use_sigmoid=False,
        m=args.n_heads,
        use_big_head=False,
        use_mask_d=args.use_mask_d)

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
os.makedirs(args.data_path, exist_ok=True)
logging.info(f'data at: {args.data_path}')

dataloader = torch.utils.data.DataLoader(
    MNIST(
        root='data/mnist/MNIST/processed/training_rf_v2.pt',
        transform=transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# for evaluate
real_imgs_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        args.data_path,
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1,
    shuffle=False,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
shared_layers_D_otim = torch.optim.Adam(discriminator.shared_layers.parameters(), lr=args.lr, betas=(args.b1, args.b2))
heads_D_optim = []
for i in range(args.n_heads):
    optim = torch.optim.Adam(getattr(discriminator, f'head_{i}').parameters(), lr=args.lr, betas=(args.b1, args.b2))
    heads_D_optim.append(optim)


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# ----------
#  Training
# ----------

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
        gen_imgs = generator(z)
        s = gen_imgs
        if args.augmentation:
            s = torch.cat([s] + [Augmentation.translation(s) for _ in range(args.aug_times)])
        lossg = loss.compute_lossg(discriminator, s)
        lossg.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        random_seq = list(range(args.n_heads))
        random.shuffle(random_seq)
        for head_id in random_seq:
            shared_layers_D_otim.zero_grad()
            heads_D_optim[head_id].zero_grad()
            g = gen_imgs.detach()
            r = real_imgs
            if args.bagging:
                r = get_real_imgs_with_headID(r, heads, head_id)
            if args.augmentation:
                g = torch.cat([g] + [Augmentation.translation(g) for _ in range(args.aug_times)])
                r = torch.cat([r] + [Augmentation.translation(r) for _ in range(args.aug_times)])
            lossd = loss.compute_lossd(discriminator, g, r)
            lossd.backward()
            shared_layers_D_otim.step()
            heads_D_optim[head_id].step()

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
    lossg_mean, lossg_std, lossd_mean, lossd_std = Metrics.compute_lossg_lossd(
        loss, generator, discriminator, real_imgs_loader, args.bound, args.dist)
    entry_1 = [lossg_mean, lossg_std, lossd_mean, lossd_std]
    header_1 = ['lossg_mean', 'lossg_std', 'lossd_mean', 'lossd_std']
    logging.info(' '.join(header_1))
    logging.info(entry_1)

    dx, dgz = Metrics.compute_heads_statistics(generator, discriminator, real_imgs_loader, args.bound, args.dist)
    entry_2 = list(dx)
    header_2 = ['dx_mean', 'dx_std', 'dx_min', 'dx_max', 'dx_max_min']
    logging.info(' '.join(header_2))
    logging.info(entry_2)
    entry_3 = list(dgz)
    header_3 = ['dgz_mean', 'dgz_std', 'dgz_min', 'dgz_max', 'dgz_max_min']
    logging.info(' '.join(header_3))
    logging.info(entry_3)

    fid = Metrics.compute_fid(generator, args.dist, args.bound)
    entry_4 = [fid]
    header_4 = ['fid_score']
    logging.info(' '.join(header_4))
    logging.info(entry_4)

    logging.info('--------------------------------')

logging.info('Completed')
