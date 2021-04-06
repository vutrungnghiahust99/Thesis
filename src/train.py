import argparse
import logging
import os
import pandas as pd

import torchvision.transforms as transforms
from torchvision import datasets
import torch

from src.models.eriklindernoren_architecture import Generator, Discriminator
from src.losses import LSGAN as lsgan_loss
from src.losses import GAN1 as gan1_loss
from src.losses import WGANGP as wgangp_loss
from src.utils import sample_image
from src.metrics import Metrics
from src.augmentation import Augmentation
from src.noise import Noise

parser = argparse.ArgumentParser()
parser.add_argument('--exp_id', type=str, required=True)
parser.add_argument("--base_exp_name", type=str, required=True)
parser.add_argument("--loss_name", type=str, choices=['gan1', 'lsgan', 'wgangp'])
parser.add_argument("--shift", type=int, default=-1)
parser.add_argument("--std", type=float, choices=[-1, 0.001, 0.01, 0.02, 0.04, 0.1, 0.5, 2, 4])
parser.add_argument("--spec_g", type=int, choices=[0, 1])
parser.add_argument("--spec_d", type=int, choices=[0, 1])
parser.add_argument("--dist", type=str, choices=['gauss', 'uniform'])
parser.add_argument("--bound", type=float, choices=[0.01, 1, 100])
parser.add_argument("--augmentation", type=int, choices=[0, 1])
parser.add_argument("--aug_times", type=int)
parser.add_argument("--train_g_more", type=int, choices=[0, 1])
parser.add_argument("--bs_g", type=int, choices=[64, 128, 256, -1])
parser.add_argument("--n_epochs", type=int)
parser.add_argument("--n_critic", type=int, default=5)
parser.add_argument("--interval", type=int, choices=[1, 5, 10])
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

s1 = f'exp_id_{args.exp_id}_base_exp_name_{args.base_exp_name}_loss_name_{args.loss_name}_shift_{args.shift}_spec_g_{args.spec_g}_spec_d_{args.spec_d}_dist_{args.dist}_std_{args.std}'
s2 = f'bound_{args.bound}_aug_{args.augmentation}_aug_times_{args.aug_times}_train_g_more_{args.train_g_more}_bs_g_{args.bs_g}_n_epochs_{args.n_epochs}_n_critic_{args.n_critic}_interval_{args.interval}'
args.exp_name = s1 + "_" + s2

# make exp_folder
exp_folder = f'experiments/{args.base_exp_name}/{args.exp_name}'
os.makedirs(exp_folder, exist_ok=True)
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
    discriminator = Discriminator(use_sigmoid=True, use_spec_norm=args.spec_d)
else:
    discriminator = Discriminator(use_sigmoid=False, use_spec_norm=args.spec_d)

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
    datasets.MNIST(
        args.data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
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
elif args.loss_name == 'wgangp':
    loss = wgangp_loss
else:
    RuntimeError(f'{args.loss_name} is not supported!!!')

# ----------
#  Training
# ----------
if args.train_g_more + args.augmentation == 2:
    RuntimeError("Can not augmentation and train g more at the same time!!!!")

header = None
results = []
batch = 0
for epoch in range(start_epoch, start_epoch + args.n_epochs):
    logging.info(f'epoch: {epoch}')
    generator.train()
    discriminator.train()
    generator.requires_grad_(True)
    discriminator.requires_grad_(True)

    for i, (imgs, _) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        real_imgs = imgs.type(Tensor)
        if args.augmentation:
            real_imgs = torch.cat([real_imgs] + [Augmentation.translation(real_imgs, args.shift) for _ in range(args.aug_times)])
 
        # -----------------
        #  Train Generator
        # -----------------

        if i % args.n_critic == 0:
            optimizer_G.zero_grad()
            if args.train_g_more:
                z_g = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, args.bs_g, args.z_dim)
                gen_imgs_g = generator(z_g)
                with torch.no_grad():
                    z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
                    gen_imgs = generator(z)
                lossg = loss.compute_lossg(discriminator, gen_imgs_g)
                lossg.backward()
                optimizer_G.step()

            elif args.augmentation:
                z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
                gen_imgs = generator(z)
                gen_imgs = torch.cat([gen_imgs] + [Augmentation.translation(gen_imgs, args.shift) for _ in range(args.aug_times)])
                lossg = loss.compute_lossg(discriminator, gen_imgs)

                lossg.backward()
                optimizer_G.step()
            else:
                z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
                gen_imgs = generator(z)
                lossg = loss.compute_lossg(discriminator, gen_imgs)

                lossg.backward()
                optimizer_G.step()
        else:
            z = Noise.sample_gauss_or_uniform_noise(args.dist, args.bound, batch_size, args.z_dim)
            gen_imgs = generator(z)
            if args.augmentation:
                gen_imgs = torch.cat([gen_imgs] + [Augmentation.translation(gen_imgs, args.shift) for _ in range(args.aug_times)])
        
        if i % 5 != 0:
            continue

        logging.info(f"batch: {batch}")
        generator.eval()
        discriminator.eval()
        generator.requires_grad_(False)
        discriminator.requires_grad_(False)

        entry_0 = [batch]
        header_0 = ['batch']

        logging.info('g_wrt_z')
        g_wrt_z = Metrics.compute_l2norm_derivative_g_wrt_z(generator, args.bound, args.dist)
        entry_1 = [g_wrt_z.mean(), g_wrt_z.std(), g_wrt_z.min(), g_wrt_z.max()]
        header_1 = ['g_wrt_z_mean', 'g_wrt_z_std', 'g_wrt_z_min', 'g_wrt_z_max']
        logging.info(entry_1)

        # logging.info('lossg_wrt_theta_g')
        # lossg_wrt_theta_g = Metrics.compute_lsnorm_derivative_lossg_wrt_theta_g(args.loss_name, generator, discriminator, args.bound, args.dist)
        # entry_2 = [lossg_wrt_theta_g.mean(), lossg_wrt_theta_g.std(), lossg_wrt_theta_g.min(), lossg_wrt_theta_g.max()]
        # header_2 = ['lossg_wrt_theta_g_mean', 'lossg_wrt_theta_g_std', 'lossg_wrt_theta_g_min', 'lossg_wrt_theta_g_max']
        # logging.info(entry_2)

        logging.info('lossg_wrt_z')
        lossg_wrt_z = Metrics.compute_l2norm_derivative_lossg_wrt_z(args.loss_name, generator, discriminator, args.bound, args.dist)
        entry_6 = [lossg_wrt_z.mean(), lossg_wrt_z.std(), lossg_wrt_z.min(), lossg_wrt_z.max()]
        header_6 = ['lossg_wrt_z_mean', 'lossg_wrt_z_std', 'lossg_wrt_z_min', 'lossg_wrt_zmax']
        logging.info(entry_6)

        entry = entry_1 + entry_6
        if header is None:
            header = header_1 + header_6

        results.append(entry)

        logging.info(f'epoch: {epoch}')
        generator.train()
        discriminator.train()
        generator.requires_grad_(True)
        discriminator.requires_grad_(True)

        batch += 1

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

csv_path = f'{exp_folder}/{args.exp_name}.csv'
df = pd.DataFrame(results, columns=header)
df.to_csv(csv_path, index=None)
logging.info('Completed')
