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
from src.utils import load_images, unnormalize_images, print_training_config, print_training_statistics, record_training_statistics

from torchsummary import summary
import torchinfo

from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

from torcheval.metrics import FrechetInceptionDistance


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
        default=8,
        help='The interval at which (lazy) R1 regularization is performed.'
    )

    parser.add_argument(
        '--pl_interval',
        '-pl',
        type=int,
        dest='pl_interval',
        default=16,
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
    print_training_config(args)

    epochs = args.epochs
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

    # Create folder where model of current training run will be stored
    exp_folder = 'run-' + datetime.today().strftime('%Y-%m-%d')
    os.makedirs(os.path.join(ROOT_DIR, 'models', exp_folder), exist_ok=True)

    # Enable cuDNN auto-tuner to automatically select kernel
    # for best performance when computing convolutions. 
    # Significantly increases speed of training.
    torch.backends.cudnn.benchmark = True 

    # Initialize tensorboard writer
    tb_writer = SummaryWriter(log_dir=os.path.join('runs', exp_folder))

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    # Initialize discrimator and generator networks
    D_net = Discriminator(res).to(DEVICE)
    G_net = Generator(z_dim, w_dim, res).to(DEVICE)
    
    torchinfo.summary(G_net.mapNet, input_size=(batch_size, z_dim))
    print()
    torchinfo.summary(G_net.syntNet, input_size=(batch_size, G_net.syntNet.num_ws, w_dim))
    print()
    torchinfo.summary(D_net, input_size=(batch_size, 3, res, res))
    print()

    # Initialize discrimator and generator optimizers
    D_opt = torch.optim.Adam(D_net.parameters(), lr=c_r1*lr, betas=(beta1**c_r1, beta2**c_r1))
    G_opt = torch.optim.Adam(G_net.parameters(), lr=c_pl*lr, betas=(beta1**c_pl, beta2**c_pl))

    # Initialize discrimator and generator loss
    D_loss = DiscriminatorLoss().to(DEVICE)
    G_loss = GeneratorLoss().to(DEVICE)

    # Load data
    dataset, _ = load_images(os.path.join(ROOT_DIR, 'data', 'PokemonData'), res=res, test_size=0.99)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Initialize Frechet Inception Distance object
    FID = FrechetInceptionDistance(device=DEVICE)

    # Load FID object if it exists, otherwise compute the statistics of 
    # real image dataset and save FID object.
    fid_path = os.path.join(ROOT_DIR, 'models', 'fid.pth')
    if os.path.exists(fid_path):
        print('Loading FID...', end=' ')
        FID.load_state_dict(torch.load(fid_path, weights_only=True))
        print('Done! \n')
    else:
        print('Computing real FID statistics...', end=' ')
        for imgs, _ in iter(torch.utils.data.DataLoader(dataset, batch_size=100)):
            imgs = unnormalize_images(imgs.to(DEVICE))
            FID.update(imgs, is_real=True)
        torch.save(FID.state_dict(), fid_path)
        print('Done! \n')
    
    best_fid_score = np.inf
    # Load pre-trained discriminator and generator models if provided 
    # and compute current best FID score.
    if args.load_model_path is not None:
        print('Loading pre-trained models...', end=' ')
        D_net.load_state_dict(torch.load(os.path.join(args.load_model_path, 'discriminator.pth'), weights_only=True))
        G_net.load_state_dict(torch.load(os.path.join(args.load_model_path, 'generator.pth'), weights_only=True))
        print('Done! \n')

        # Generate fake images
        z = torch.randn((100, z_dim), device=DEVICE)
        fake_imgs, _ = G_net(z)
        del z

        # Update the FID statistics of fake images and
        # compute current best FID score.
        print('Computing fake FID statistics...', end=' ')
        fake_imgs = unnormalize_images(fake_imgs)
        FID.update(fake_imgs, is_real=False)
        best_fid_score = FID.compute()
        print('Done! \n')
        print(f'Current FID score: {best_fid_score} \n')
    else:
        print('No pre-trained models provided. Training from scratch. \n')

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
            # Move images to DEVICE
            real_imgs = imgs.to(DEVICE)
            
            # Generate mini-batch of fake images
            z = torch.randn((batch_size, z_dim), device=DEVICE)
            fake_imgs, ws = G_net(z, style_mix_prob=0.9)
            del z
           
            # Compute discriminator loss and regularization terms
            do_r1_reg = i % r1_interval == 0
            d_loss, r1_penalty, D_real, D_fake = D_loss(D_net, real_imgs, fake_imgs, do_reg=do_r1_reg)

            # Update running discriminator statistics
            running_d_loss += d_loss.item() * real_imgs.shape[0]
            running_r1_penalty += r1_penalty.item() * real_imgs.shape[0] 
            running_D_real += D_real * real_imgs.shape[0]
            running_D_fake += D_fake * real_imgs.shape[0]  
            del real_imgs

            # Perform backward pass of loss and optimization step on discriminator
            D_opt.zero_grad(set_to_none=True)
            d_loss.backward(retain_graph=do_r1_reg)
            D_opt.step()
            del d_loss

            # Perform backward pass of R1 penalty and optimization step on discriminator
            if do_r1_reg:
                D_opt.zero_grad(set_to_none=True)
                r1_penalty = r1_interval * r1_penalty
                r1_penalty.backward()
                D_opt.step()
            del r1_penalty

            # Compute generator loss and regularization terms
            do_pl_reg = i % pl_interval == 0
            g_loss, pl_penalty = G_loss(D_net, fake_imgs, ws, do_reg=do_pl_reg)
            
            # Update running generator statistics
            running_g_loss += g_loss.item() * fake_imgs.shape[0]
            running_pl_penalty += pl_penalty.item() * fake_imgs.shape[0]
            del fake_imgs, ws

            # Perform backward pass of loss and optimization step on generator
            G_opt.zero_grad(set_to_none=True)
            g_loss.backward(retain_graph=do_pl_reg)
            G_opt.step()
            del g_loss

            # Perform backward pass of path length penalty and optimization step on generator
            if do_pl_reg:
                G_opt.zero_grad(set_to_none=True)
                pl_penalty = pl_interval * pl_penalty
                pl_penalty.backward()
                G_opt.step()
            del pl_penalty

            gc.collect()
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Generate fake images
            z = torch.randn((50, z_dim), device=DEVICE)
            fake_imgs, _ = G_net(z)
            del z

            # Update the FID statistics of fake images and
            # compute current FID score.
            fake_imgs = unnormalize_images(fake_imgs)
            FID.update(fake_imgs, is_real=False)
            fid_score = FID.compute()

            # if (ep+1) % 10 == 0:
            #     for i in range(3):
            #         plt.imshow(F.to_pil_image(fake_imgs[i]))
            #         plt.show()
            del fake_imgs  

            if fid_score < best_fid_score:
                best_fid_score = fid_score
                torch.save(G_net.state_dict(), os.path.join(ROOT_DIR, 'models', exp_folder, 'generator.pth'))
                torch.save(D_net.state_dict(), os.path.join(ROOT_DIR, 'models', exp_folder, 'discriminator.pth'))

            stats = {}
            stats['FID score'] = fid_score
            stats['Discrimator loss'] = running_d_loss / len(dataloader.sampler)
            stats['Generator loss'] = running_g_loss / len(dataloader.sampler)
            stats['R1 penalty'] = running_r1_penalty / (len(dataloader.sampler) / r1_interval)
            stats['PL penalty'] = running_pl_penalty / (len(dataloader.sampler) / pl_interval)
            stats['D real'] = running_D_real / len(dataloader.sampler)
            stats['D fake'] = running_D_fake / len(dataloader.sampler)

            print_training_statistics(stats, ep)
            record_training_statistics(stats, tb_writer, ep)

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()