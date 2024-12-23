import torch
import os
import gc
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm

from src.loss import DiscriminatorLoss, GeneratorLoss
from src.stylegan2 import Discriminator, Generator

from src.constants import ROOT_DIR
import src.utils as utils

import torchinfo

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from torcheval.metrics import FrechetInceptionDistance

import time


# TODO: Implement training loop. Utilize at least DataParallel to split 
# batch computaion to multiple GPUs. Consider using DistributedDataParallel
# for even better multi-GPU performance. 

def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        dest='load_model_path',
        help='Path to the directory containing the model to be loaded. Defaults to None',
    )

    parser.add_argument(
        '--epochs',
        '-ep',
        type=int,
        default=100,
        dest='epochs',
        help='Number of epochs to train for.',
    )

    parser.add_argument(
        '--snap_freq',
        type=int,
        dest='snapshot_frequency',
        default=10,
        help='The frequency in terms of epochs with which to take snapshot of current model.'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        dest='num_workers',
        default=0,
        help='The number of worker processes for the dataloaders to use.'
    )

    parser.add_argument(
        '--z_dim',
        type=int,
        dest='z_dim',
        default=512,
        help='Dimension of latent vector z.'
    )

    parser.add_argument(
        '--w_dim',
        type=int,
        dest='w_dim',
        default=512,
        help='Dimension of intermediate latent vector w.'
    )

    parser.add_argument(
        '--resolution',
        '-res',
        type=int,
        dest='resolution',
        default=256,
        help='The resolution of the generated images.'
    )

    parser.add_argument(
        '--batch_size',
        '-bs',
        type=int,
        dest='batch_size',
        default=32,
        help='The size of a mini-batch.'
    )

    parser.add_argument(
        '--r1_interval',
        '-r1',
        type=int,
        dest='r1_interval',
        default=16,
        help='The interval at which (lazy) R1 regularization is performed.'
    )

    parser.add_argument(
        '--pl_interval',
        '-pl',
        type=int,
        dest='pl_interval',
        default=8,
        help='The interval at which (lazy) path length regularization is performed.'
    )

    parser.add_argument(
        '--learning_rate',
        '-lr',
        type=float,
        dest='learning_rate',
        default=0.001,
        help='The learning rate of the optimizers.'
    )

    parser.add_argument(
        '--beta1',
        '-b1',
        type=float,
        dest='beta1',
        default=0.,
        help='The first moment parameter beta1 of Adam optimizer.'
    )

    parser.add_argument(
        '--beta2',
        '-b2',
        type=float,
        dest='beta2',
        default=0.99,
        help='The second moment parameter beta1 of Adam optimizer.'
    )

    return parser.parse_args()


def main():
    print()
    args = parser()
    utils.print_training_config(args)

    epochs = args.epochs
    snap_freq = args.snapshot_frequency
    num_workers = args.num_workers
    z_dim = args.z_dim
    w_dim = args.w_dim
    res = args.resolution
    batch_size = args.batch_size
    r1_interval = args.r1_interval
    pl_interval = args.pl_interval
    lr = args.learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    c_r1 = r1_interval / (r1_interval + 1) # R1 lazy regularization correction term for optimizer hyperparams
    c_pl = pl_interval / (pl_interval + 1) # Path length lazy regularization correction term for optimizer hyperparams

    # Create name for current training run
    run_name = 'run-' + datetime.today().strftime('%Y-%m-%d')

    # Enable cuDNN auto-tuner to automatically select kernel
    # for best performance when computing convolutions. 
    # Significantly increases speed of training.
    torch.backends.cudnn.benchmark = True 

    # Initialize tensorboard writer
    tb_writer = SummaryWriter(log_dir=os.path.join('runs', run_name))

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    act_kwargs = {'negative_slope': 0.2}
    map_kwargs = synt_kwargs = {'act_kwargs': act_kwargs}
    # Initialize discrimator and generator networks
    D_net = Discriminator(res, act_kwargs=act_kwargs)
    G_net = Generator(z_dim, w_dim, res, map_kwargs=map_kwargs, synt_kwargs=synt_kwargs)
    
    # Load pre-trained discriminator and generator networks if provided.
    if args.load_model_path is not None:
        print('Loading pre-trained models...', end=' ')
        D_net.load_state_dict(torch.load(os.path.join(args.load_model_path, 'discriminator.pth'), weights_only=True))
        G_net.load_state_dict(torch.load(os.path.join(args.load_model_path, 'generator.pth'), weights_only=True))
        print('Done! \n')
    else:
        print('No pre-trained models provided. Training from scratch. \n')

    # Move networks to DEVICE
    D_net.to(DEVICE)
    G_net.to(DEVICE)
    
    # Print summaries of the networks
    torchinfo.summary(G_net.mapNet, input_size=(batch_size, z_dim), device=DEVICE)
    print()
    torchinfo.summary(G_net.syntNet, input_size=(batch_size, G_net.syntNet.num_ws, w_dim), device=DEVICE)
    print()
    torchinfo.summary(D_net, input_size=(batch_size, 3, res, res), device=DEVICE)
    print()

    # Initialize discrimator and generator optimizers
    D_opt = torch.optim.Adam(D_net.parameters(), lr=c_r1*lr, betas=(beta1**c_r1, beta2**c_r1))
    G_opt = torch.optim.Adam(G_net.parameters(), lr=c_pl*lr, betas=(beta1**c_pl, beta2**c_pl))

    # Initialize discrimator and generator loss
    D_loss = DiscriminatorLoss().to(DEVICE)
    G_loss = GeneratorLoss().to(DEVICE)

    # Load data
    dataset, _ = utils.load_images(os.path.join(ROOT_DIR, 'data', 'PokemonData'), res=res)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # Initialize Frechet Inception Distance object
    FID = FrechetInceptionDistance(device=DEVICE)
    # Load FID object if it exists, otherwise compute the statistics of 
    # real image dataset and save FID object.
    fid_path = os.path.join(ROOT_DIR, 'models', 'fid_pokemon.pth')
    if os.path.exists(fid_path):
        print('Loading FID...', end=' ', flush=True)
        FID.load_state_dict(torch.load(fid_path, weights_only=True))
        FID.to(DEVICE) # Make sure all attributes of loaded FID object are on the same device
        print('Done! \n')
    else:
        print('Computing real image FID statistics...', flush=True)
        for imgs, _ in iter(tqdm(dataloader)):
            # Move images to DEVICE and scale pixel values to range [0, 1]
            imgs = imgs.to(DEVICE) / 255.0
            FID.update(imgs, is_real=True)

        os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
        torch.save(FID.state_dict(), fid_path)
        print('Done! \n')

    print('Starting training! \n')
    for ep in range(epochs):
        # Initialize running statistics
        running_d_loss = 0
        running_r1_penalty = 0
        running_g_loss = 0
        running_pl_penalty = 0
        running_D_real = 0
        running_D_fake = 0

        for i, (imgs, _)  in enumerate(tqdm(iter(dataloader), desc=f'Epoch {ep}'), 1):
            # Move images to DEVICE and scale pixel values to range [0, 1]
            real_imgs = imgs.to(DEVICE) / 255.0
            batch_size = real_imgs.shape[0]
            
            # Generate mini-batch of fake images and intermediate latent vectors (ws)
            z = torch.randn((batch_size, z_dim), device=DEVICE)
            fake_imgs, ws = G_net(z, style_mix_prob=0.9)
            del z

            # Perform main discriminator loss pass 
            d_loss, D_real, D_fake = D_loss(D_net, real_imgs, fake_imgs)
            running_d_loss += d_loss.item() * batch_size
            running_D_real += D_real * batch_size
            running_D_fake += D_fake * batch_size  

            D_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            D_opt.step()
            del d_loss

            # Perform R1 regularization pass
            if i % r1_interval == 0:
                r1_penalty = r1_interval * D_loss.r1_reg(real_imgs, D_net)
                running_r1_penalty += r1_penalty.item() * batch_size

                D_opt.zero_grad(set_to_none=True)
                r1_penalty.backward()
                D_opt.step()
                del r1_penalty
            del real_imgs

            # Perform main generator loss pass
            g_loss = G_loss(D_net, fake_imgs)
            running_g_loss += g_loss.item() * batch_size

            G_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            G_opt.step()
            del g_loss

            # Perform path length regularization pass
            if i % pl_interval == 0:
                # Generate new mini-batch of fake images and intermediate latent vectors (ws)
                # with updated generator
                z = torch.randn((batch_size, z_dim), device=DEVICE)
                fake_imgs, ws = G_net(z, style_mix_prob=0.9)
                del z   
                
                pl_penalty = pl_interval * G_loss.pl_reg(fake_imgs, ws)
                running_pl_penalty += pl_penalty.item() * batch_size

                G_opt.zero_grad(set_to_none=True)
                pl_penalty.backward()
                G_opt.step()
                del pl_penalty
            del fake_imgs, ws

            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            stats = {}

            if ep % snap_freq == 0:
                # Compute and store current FID score
                stats['FID score'] = utils.compute_FID_score(FID, G_net)

                # Create path to snapshot folder
                save_path = os.path.join(ROOT_DIR, 'snapshots', run_name, 'epoch_' + str(ep))
                os.makedirs(save_path, exist_ok=True)

                # Generate and save grid of fake images to snapshot folder
                fake_imgs = G_net.generate_images(num_imgs=16)
                utils.save_image_grid(fake_imgs, save_path, nrow=4)
                del fake_imgs

                # Save models to snapshot folder
                torch.save(G_net.state_dict(), os.path.join(save_path, 'generator.pth'))
                torch.save(D_net.state_dict(), os.path.join(save_path, 'discriminator.pth'))

            # Compute and store mean of running statistics
            stats['Discrimator loss'] = running_d_loss / len(dataloader.sampler)
            stats['Generator loss'] = running_g_loss / len(dataloader.sampler)
            stats['R1 penalty'] = running_r1_penalty / (len(dataloader.sampler) / r1_interval)
            stats['PL penalty'] = running_pl_penalty / (len(dataloader.sampler) / pl_interval)
            stats['D real'] = running_D_real / len(dataloader.sampler)
            stats['D fake'] = running_D_fake / len(dataloader.sampler)

            utils.print_training_statistics(stats, ep)
            utils.record_training_statistics(stats, tb_writer, ep)

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()