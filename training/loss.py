import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class GeneratorLoss(nn.Module):
    def __init__(self, pl_weight=2.0, pl_beta=0.99):
        super().__init__()

        self.pl_weight = pl_weight
        self.pl_beta = pl_beta
        self.pl_ema = torch.zeros([])
        self.t = 1

    def pl_reg(self, fake_images, ws):
        *_, img_H, img_W = fake_images.shape
        y = torch.randn_like(fake_images) / np.sqrt(img_H * img_W)

        grad = torch.autograd.grad(outputs=fake_images * y,
                                   inputs=ws,
                                   grad_outputs=torch.ones_like(fake_images),
                                   create_graph=True)[0]

        norm = grad.square().sum(2).mean(0).sqrt()

        self.pl_ema = self.pl_beta * self.pl_ema + (1 - self.pl_beta) * norm.detach().mean()
        pl_ema_hat = self.pl_ema / (1 - self.pl_beta**self.t)
        self.t += 1

        pl_penalty = (norm - pl_ema_hat).square().mean()
     
        return self.pl_weight * pl_penalty

    def forward(self, netD, fake_images, ws, do_reg=False):
        logits = netD(fake_images)
        loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

        pl_penalty = 0
        if do_reg:
            pl_penalty = self.pl_reg(fake_images, ws)

        return loss, pl_penalty
    

class DiscriminatorLoss(nn.Module):
    def __init__(self, r1_gamma=10):
        super().__init__()

        self.r1_gamma = r1_gamma

    def r1_reg(self, real_images, logits_real):
        grad = torch.autograd.grad(logits_real,
                                   real_images,
                                   grad_outputs=torch.ones_like(logits_real),
                                   create_graph=True)[0]

        penalty = grad.square().sum([1, 2, 3])
        penalty = self.r1_gamma / 2 * grad.square().mean()

        return penalty
    
    def forward(self, netD, real_images, fake_images, do_reg=False):
        real_images_tmp = real_images.detach().requires_grad_(do_reg)
        fake_images_tmp = fake_images.detach()
        
        logits_real = netD(real_images_tmp)
        logits_fake = netD(fake_images_tmp)

        D_real = F.sigmoid(logits_real).mean()
        D_fake = F.sigmoid(logits_fake).mean()

        loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))
        loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

        loss = loss_real + loss_fake

        r1_penalty = 0
        if do_reg:
            r1_penalty = self.r1_reg(real_images_tmp, logits_real)
        
        return loss, r1_penalty, D_real, D_fake
