import torch
import os
import gc

from training.loss import DiscriminatorLoss, GeneratorLoss
from training.stylegan2 import Discriminator, Generator

from training.constants import ROOT_DIR
from training.utils import load_images, unnormalize_images

from torchsummary import summary
import torchinfo

from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

from torcheval.metrics import FrechetInceptionDistance


# TODO: Implement training loop. Utilize at least DataParallel to split 
# batch computaion to multiple GPUs. Consider using DistributedDataParallel
# for even better multi-GPU performance. 


def main():
    epochs = 100
    z_dim = 512
    w_dim = 512
    res = 256
    batch_size = 8
    r1_interval = 8
    pl_interval = 16
    lr = 0.001
    beta1 = 0.
    beta2 = 0.99
    # epsilon = 1e-08
    c_r1 = r1_interval / (r1_interval + 1) # R1 lazy regularization correction term for optimizer hyperparams
    c_pl = pl_interval / (pl_interval + 1) # Path length lazy regularization correction term for optimizer hyperparams

    # Enable cuDNN auto-tuner to automatically select kernel
    # for best performance when computing convolutions. 
    # Significantly increases speed of training
    torch.backends.cudnn.benchmark = True 

    # Initialize tensorboard writer
    tb_writer = SummaryWriter()

    # Set device
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    # Initialize discrimator and generator networks
    D_net = Discriminator(res).to(DEVICE)
    G_net = Generator(z_dim, w_dim, res).to(DEVICE)
    
    # summary(G.mapNet, input_size=(z_dim,))
    # torchinfo.summary(G.mapNet, input_size=(z_dim,), batch_dim=0)
    # print()
    # summary(G.syntNet, input_size=(G.syntNet.num_ws, w_dim))
    # torchinfo.summary(G.syntNet, input_size=(G.syntNet.num_ws, w_dim), batch_dim=0)
    # print()
    # summary(D, input_size=(3, res, res))
    # torchinfo.summary(D, input_size=(3, res, res), batch_dim=0)
    # print()

    # Initialize discrimator and generator optimizers
    D_opt = torch.optim.Adam(D_net.parameters(), lr=c_r1*lr, betas=(beta1**c_r1, beta2**c_r1))
    G_opt = torch.optim.Adam(G_net.parameters(), lr=c_pl*lr, betas=(beta1**c_pl, beta2**c_pl))

    # D_opt_pen = torch.optim.Adam(D_net.parameters(), lr=c_r1*lr, betas=(beta1**c_r1, beta2**c_r1))
    # G_opt_pen = torch.optim.Adam(G_net.parameters(), lr=c_pl*lr, betas=(beta1**c_pl, beta2**c_pl))

    # Initialize discrimator and generator loss
    D_loss = DiscriminatorLoss().to(DEVICE)
    G_loss = GeneratorLoss().to(DEVICE)

    # Load data
    dataset, _ = load_images(os.path.join(ROOT_DIR, 'data/PokemonData'), res=res, test_size=0.99)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Initialize Frechet Inception Distance object
    FID = FrechetInceptionDistance(device=DEVICE)
    for imgs, _ in iter(torch.utils.data.DataLoader(dataset, batch_size=100)):
        imgs = unnormalize_images(imgs.to(DEVICE))
        FID.update(imgs, is_real=True)

    # print(torch.cuda.memory_summary())

    print('Starting training!')
    for ep in range(epochs):
        running_d_loss = 0
        running_r1_penalty = 0
        running_g_loss = 0
        running_pl_penalty = 0
        running_D_real = 0
        running_D_fake = 0

        for i, (imgs, _)  in enumerate(iter(dataloader), 1):
            # print(f"Iteration: {i}")
            real_imgs = imgs.to(DEVICE)
            
            z = torch.randn((batch_size, z_dim), device=DEVICE)
            fake_imgs, ws = G_net(z, style_mix_prob=0.9)
            del z
           
            do_r1_reg = i % r1_interval == 0
            d_loss, r1_penalty, D_real, D_fake = D_loss(D_net, real_imgs, fake_imgs, do_reg=do_r1_reg)
            running_d_loss += d_loss.item() * real_imgs.shape[0]
            running_D_real += D_real * real_imgs.shape[0]
            running_D_fake += D_fake * real_imgs.shape[0]
            if do_r1_reg:
                # r1_penalty = r1_interval * r1_penalty
                running_r1_penalty += r1_penalty.item() * real_imgs.shape[0]   
            del real_imgs

            # d_loss = d_loss + r1_interval * r1_penalty

            D_opt.zero_grad(set_to_none=True)
            d_loss.backward(retain_graph=do_r1_reg)
            D_opt.step()
            del d_loss

            if do_r1_reg:
                # print("Doing r1 regularization backpass")
                D_opt.zero_grad(set_to_none=True)
                # D_opt_pen.zero_grad(set_to_none=True)
                # D_opt_pen.load_state_dict(D_opt.state_dict())
                r1_penalty = r1_interval * r1_penalty
                r1_penalty.backward()
                # D_opt_pen.step()
                D_opt.step()
                del r1_penalty
                # print("Success!!!")

            do_pl_reg = i % pl_interval == 0
            g_loss, pl_penalty = G_loss(D_net, fake_imgs, ws, do_reg=do_pl_reg)
            running_g_loss += g_loss.item() * fake_imgs.shape[0]
            if do_pl_reg:
                # pl_penalty = pl_interval * pl_penalty
                running_pl_penalty += pl_penalty.item() * fake_imgs.shape[0]
            del fake_imgs, ws

            # g_loss = g_loss + pl_interval * pl_penalty

            G_opt.zero_grad(set_to_none=True)
            g_loss.backward(retain_graph=do_pl_reg)
            G_opt.step()
            del g_loss

            if do_pl_reg:
                # print("Doing pl regularization backpass")
                G_opt.zero_grad(set_to_none=True)
                # G_opt_pen.zero_grad(set_to_none=True)
                # G_opt_pen.load_state_dict(G_opt.state_dict())
                pl_penalty = pl_interval * pl_penalty
                pl_penalty.backward()
                # G_opt_pen.step()
                G_opt.step()
                del pl_penalty
                # print("Success!!!")
            
            gc.collect()
            torch.cuda.empty_cache()

        # print(torch.cuda.memory_summary())

        with torch.no_grad():
            z = torch.randn((50, z_dim), device=DEVICE)
            fake_imgs, _ = G_net(z)
            del z

            fake_imgs = unnormalize_images(fake_imgs)
            FID.update(fake_imgs, is_real=False)
            fid_score = FID.compute()

            # if (ep+1) % 10 == 0:
            #     for i in range(3):
            #         plt.imshow(F.to_pil_image(fake_imgs[i]))
            #         plt.show()
            del fake_imgs 

            d_loss_mean = running_d_loss / len(dataloader.sampler)
            g_loss_mean = running_g_loss / len(dataloader.sampler)
            r1_penalty_mean = running_r1_penalty / (len(dataloader.sampler) / r1_interval)
            pl_penalty_mean = running_pl_penalty / (len(dataloader.sampler) / pl_interval)
            D_real_mean = running_D_real / len(dataloader.sampler)
            D_fake_mean = running_D_fake / len(dataloader.sampler)

            print(50*'-' + '\n')
            print(f'Epoch {ep} completed with:')
            print(f'FID score {fid_score}')
            print(f'Discrimator loss {d_loss_mean}')
            print(f'Generator loss {g_loss_mean}')
            print(f'R1 penalty {r1_penalty_mean}')
            print(f'PL penalty {pl_penalty_mean}')
            print(f'D_real {D_real_mean}')
            print(f'D_fake {D_fake_mean}')
            print()

            tb_writer.add_scalar('FID_score', fid_score, ep)
            tb_writer.add_scalar('D_loss', d_loss_mean, ep)
            tb_writer.add_scalar('G_loss', g_loss_mean, ep)
            tb_writer.add_scalar('R1_penalty', r1_penalty_mean, ep)
            tb_writer.add_scalar('PL_penalty', pl_penalty_mean, ep)
            tb_writer.add_scalar('D_real', D_real_mean, ep)
            tb_writer.add_scalar('D_fake', D_fake_mean, ep)

            gc.collect()
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary())




if __name__ == '__main__':
    main()