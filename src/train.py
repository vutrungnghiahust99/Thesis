import argparse
import logging
import os
import pandas as pd
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision import datasets
import torch

from src.models.eriklindernoren_architecture import Generator
from src.models.random_forest_d_architecture import Discriminator
from src.dataset import MNIST
from src.losses import LSGAN as lsgan_loss
from src.losses import GAN1 as gan1_loss
from src.utils import sample_image
# from src.utils import get_gen_real_imgs_with_headID
from src.metrics import Metrics
from src.augmentation import Augmentation
from src.noise import Noise

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--loss_name", type=str, choices=['gan1', 'lsgan'])

# augmentation
parser.add_argument("--augmentation", type=int, choices=[0, 1], default=0)
parser.add_argument("--translation_shift", type=int, choices=[2, 4, 8, -1], default=-1)
parser.add_argument("--gaussian_noise_std", type=float, choices=[-1, 0.001, 0.01, 0.02, 0.04, 0.1, 0.5], default=-1)
parser.add_argument("--aug_times", type=int, default=-1)

# spectral normalization
parser.add_argument("--spec_g", type=int, choices=[0, 1], default=0)
parser.add_argument("--spec_d", type=int, choices=[0, 1], default=0)

# input noise z
parser.add_argument("--dist", type=str, choices=['gauss', 'uniform'], default='gauss')
parser.add_argument("--bound", type=float, default=1)

# No. heads in the discriminator
parser.add_argument("--n_heads", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=10)

parser.add_argument("--n_epochs", type=int)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--weights_g", type=str, default='')
parser.add_argument("--weights_d", type=str, default='')

# unchanged configs
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


s1 = f'{args.exp_name}_loss_{args.loss_name}_spec_g_{args.spec_g}_spec_d_{args.spec_d}_dist_{args.dist}'
if args.augmentation:
    s1 = s1 + f'_translation_shift_{args.translation_shift}_gaussian_noise_std_{args.gaussian_noise_std}_aug_times_{args.aug_times}'
args.exp_id = s1

# make exp_folder
exp_folder = f'experiments/{args.exp_id}'
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
generator = Generator(use_spectral_norm=args.spec_g)
if args.loss_name == 'gan1':
    discriminator = Discriminator(use_sigmoid=True, use_spec_norm=args.spec_d, n_heads=args.n_heads)
else:
    discriminator = Discriminator(use_sigmoid=False, use_spec_norm=args.spec_d, n_heads=args.n_heads)

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
        root='data/mnist/MNIST/processed/training_rf.pt',
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
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

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
        if args.augmentation:
            real_imgs = torch.cat([real_imgs] + [Augmentation.add_guassian_noise(real_imgs, args.std) for _ in range(args.aug_times)])
            heads = heads * (1 + args.aug_times)

        for head_id in range(args.n_heads):
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()
            if args.augmentation:
                z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
                gen_imgs = generator(z)
                gen_imgs = torch.cat([gen_imgs] + [Augmentation.add_guassian_noise(gen_imgs, args.std) for _ in range(args.aug_times)])
                lossg = loss.compute_lossg_rf(discriminator, gen_imgs, head_id)

                lossg.backward()
                optimizer_G.step()
            else:
                z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
                gen_imgs = generator(z)
                lossg = loss.compute_lossg_rf(discriminator, gen_imgs, head_id)

                lossg.backward()
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            optimizer_D.zero_grad()
            # g, r = get_gen_real_imgs_with_headID(gen_imgs.detach(), real_imgs, heads, head_id)
            # lossd = loss.compute_lossd_rf(discriminator, g, r, head_id)
            lossd = loss.compute_lossd_rf(discriminator, gen_imgs.detach(), real_imgs, head_id)
            lossd.backward()
            optimizer_D.step()

    if epoch % args.interval != 0:
        continue

    generator.eval()
    discriminator.eval()
    generator.requires_grad_(False)
    discriminator.requires_grad_(False)

    # save checkpoint
    g_path = os.path.join(models_folder, f'generator_{epoch}.pth')
    d_path = os.path.join(models_folder, f'discriminator_{epoch}.pth')
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
    fid_score = Metrics.compute_fid(generator, args.dist, args.bound)
    entry_8 = [fid_score]
    header_8 = ['fid_score']
    logging.info(entry_8)
    print(fid_score)

    entry = entry_7 + entry_8
    if header is None:
        header = header_7 + header_8

    results.append(entry)
    logging.info('--------------------------------')

csv_path = f'{exp_folder}/{args.exp_name}.csv'
df = pd.DataFrame(results, columns=header)
df.to_csv(csv_path, index=None)
logging.info('Completed')
