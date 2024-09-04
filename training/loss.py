import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import REAL_LABEL, FAKE_LABEL


class GeneratorLoss(nn.Module):
    def __init__(self, pl_gamma, pl_beta=0.99):

        super().__init__()

        self.pl_gamma = pl_gamma
        self.pl_beta = pl_beta
        self.pl_ema = torch.zeros([], dtype=torch.float32)

    # TODO: Implement path length regularization
    def pl_reg(self):
        print(self.pl_ema)
        self.pl_ema += 1
        print(self.pl_ema)

    def forward(self, netD, fake_images, do_reg=False):
        batch_size = fake_images.shape[0]
        targets = REAL_LABEL * torch.ones((batch_size, 1)).to(fake_images.device)

        output = netD(fake_images)

        loss = F.binary_cross_entropy(output, targets)

        if do_reg:
            self.pl_reg()

        return loss
    

class DiscriminatorLoss(nn.Module):
    def __init__(self,
                 r1_gamma=10,
                 r1_interval=16):

        super().__init__()

        self.r1_gamma = r1_gamma
        self.r1_interval = r1_interval

    def r1_reg(self, real_images, logits_real):
        # print(real_images.requires_grad)
        # print(logits_real.requires_grad)
        grad = torch.autograd.grad(logits_real,
                                   real_images,
                                   grad_outputs=torch.ones_like(logits_real),
                                   create_graph=True)[0]
        # print(grad.shape)
        penalty = grad.square().sum([1, 2, 3])
        penalty = self.r1_gamma / 2 * grad.square().mean()
        # print(penalty.shape)
        return penalty
    
    def forward(self, netD, real_images, fake_images, do_reg=False):
        batch_size = real_images.shape[0]
    
        real_images_tmp = real_images.detach().requires_grad_(do_reg)

        targets_real = REAL_LABEL * torch.ones((batch_size, 1)).to(real_images_tmp.device)
        targets_fake = FAKE_LABEL * torch.ones((batch_size, 1)).to(fake_images.device)
        
        logits_real = netD(real_images_tmp)
        logits_fake = netD(fake_images)
        # print(logits_real.shape)
        # print(logits_real)

        output_real = F.sigmoid(logits_real)
        output_fake = F.sigmoid(logits_fake)
        # print(output_real.shape)
        # print(output_real)
        
        D_real = torch.mean(output_real).to(real_images_tmp.device)
        D_fake = torch.mean(output_fake).to(fake_images.device)
        # print(D_real.shape)
        # print(D_real)
        
        loss_real = F.binary_cross_entropy(output_real, targets_real)
        loss_fake = F.binary_cross_entropy(output_fake, targets_fake)
        # print(loss_real.shape)
        # print(loss_real)
        # print()

        r1_penalty = 0
        if do_reg:
            r1_penalty = self.r1_reg(real_images_tmp, logits_real)
        # print(r1_penalty.shape)
        # print(r1_penalty)

        loss = loss_real + loss_fake
        # loss = loss_real + r1_penalty + loss_fake
        # print(loss.shape)
        # print(loss)
        
        return loss, r1_penalty, D_real, D_fake
        # return loss_real, loss_fake, r1_penalty, D_real, D_fake
        # return d_loss_real, D_real, d_loss_fake, D_fake
