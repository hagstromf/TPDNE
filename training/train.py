import torch
import os

from training.loss import DiscriminatorLoss, GeneratorLoss
from training.stylegan2 import Discriminator, Generator

from utils.constants import DEVICE, ROOT_DIR
from utils.load import load_images

from torchsummary import summary
import torchinfo


# TODO: Implement training loop. Utilize at least DataParallel to split 
# batch computaion to multiple GPUs. Consider using DistributedDataParallel
# for even better multi-GPU performance. Implement logging using tensorboard
# and some rudimentary logger for console output.


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

    D_opt = torch.optim.Adam(D_net.parameters(), lr=c_r1*lr, betas=(beta1**c_r1, beta2**c_r1))
    G_opt = torch.optim.Adam(G_net.parameters(), lr=c_pl*lr, betas=(beta1**c_pl, beta2**c_pl))

    D_loss = DiscriminatorLoss().to(DEVICE)
    G_loss = GeneratorLoss().to(DEVICE)

    trainloader, testloader = load_images(os.path.join(ROOT_DIR, 'data/PokemonData'), res=res, batch_size=batch_size)

    for ep in range(epochs):
        running_d_loss = 0
        running_r1_penalty = 0
        running_g_loss = 0
        running_pl_penalty = 0

        for i, (imgs, _)  in enumerate(iter(trainloader), 1):
            # print(f"Iteration: {i}")
            real_imgs = imgs.to(DEVICE)
            
            z = torch.randn((batch_size, z_dim), device=DEVICE)
            fake_imgs, ws = G_net(z, style_mix_prob=0.9)
            del z
           
            do_r1_reg = i % r1_interval == 0
            d_loss, r1_penalty, D_real, D_fake = D_loss(D_net, real_imgs, fake_imgs, do_reg=do_r1_reg)
            running_d_loss += d_loss.item() * real_imgs.shape[0]
            if do_r1_reg:
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
                r1_penalty = r1_interval * r1_penalty
                r1_penalty.backward()
                D_opt.step()
                del r1_penalty
                # print("Success!!!")

            do_pl_reg = i % pl_interval == 0
            g_loss, pl_penalty = G_loss(D_net, fake_imgs, ws, do_reg=do_pl_reg)
            running_g_loss += g_loss.item() * fake_imgs.shape[0]
            if do_pl_reg:
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
                pl_penalty = pl_interval * pl_penalty
                pl_penalty.backward()
                G_opt.step()
                del pl_penalty
                # print("Success!!!")

        with torch.no_grad():
            pass




if __name__ == '__main__':
    main()