import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Dict, Any, Optional, List

class EqualizedLinear(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]] = None
                ) -> None:
        
        super().__init__()
        
        if act_kwargs is None:
            act_kwargs = {}
        self.activation = getattr(nn, act_fn)(**act_kwargs)

        self.weight = nn.Parameter(torch.randn([out_dim, in_dim], dtype=torch.float32) / lr_multiplier)
        self.weight_scale = lr_multiplier / np.sqrt(in_dim)

        self.bias = nn.Parameter(torch.full([out_dim], bias_init, dtype=torch.float32)) if bias else None
        self.bias_scale = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight * self.weight_scale
        bias = self.bias
        if bias is not None:
            bias = bias * self.bias_scale

        return self.activation(F.linear(x, weight=W, bias=bias))
    

class EqualizedConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 groups: int = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]] = None
                ) -> None:
        
        super().__init__()
        
        if act_kwargs is None:
            act_kwargs = {}
        self.activation = getattr(nn, act_fn)(**act_kwargs)

        kH, kW = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        assert type(kH) == int, "Kernel height kH needs to be an integer!"
        assert type(kW) == int, "Kernel width kW needs to be an integer!"

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kH, kW], dtype=torch.float32) / lr_multiplier)
        self.weight_scale = lr_multiplier / np.sqrt(in_channels * kH * kW)

        self.bias = nn.Parameter(torch.full([out_channels], bias_init, dtype=torch.float32)) if bias else None
        self.bias_scale = lr_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.weight * self.weight_scale
        bias = self.bias
        if bias is not None:
            bias = bias * self.bias_scale

        return self.activation(F.conv2d(x, 
                                        weight=W, 
                                        bias=bias, 
                                        stride=self.stride, 
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=self.groups))
    

class EqualizedConv2dModulated(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 lr_multiplier: float = 1.,
                 demodulate: bool = True
                ) -> None:
        
        super().__init__()

        kH, kW = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        assert type(kH) == int, "Kernel height kH needs to be an integer!"
        assert type(kW) == int, "Kernel width kW needs to be an integer!"

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn([out_channels, in_channels, kH, kW], dtype=torch.float32) / lr_multiplier)
        self.weight_scale = lr_multiplier / np.sqrt(in_channels * kH * kW)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # Scale weight
        W = self.weight * self.weight_scale

        # Modulate weight
        s = s[:, None, :, None, None]
        W = s * W.unsqueeze(0)

        if self.demodulate:
            # Demodulate weight
            sigma = torch.sqrt(torch.sum(W**2, dim=(-3, -2, -1), keepdim=True) + 1e-8)
            W = W / sigma

        # Reshape input and weight such that convolution layer sees one sample 
        # of batch_size groups (as detailed in Appendix B of StyleGan2 paper).
        batch_size, _, height, width = x.shape
        x = x.reshape(1, -1, height, width)

        out_channels = W.shape[1]
        W = W.reshape(batch_size * out_channels, *W.shape[-3:])

        x = F.conv2d(x, 
                    weight=W, 
                    bias=None,
                    stride=self.stride, 
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=batch_size)
        
        # Reshape and return output as a minibatch of batch_size samples
        return x.reshape(batch_size, out_channels, *x.shape[-2:])
    

class ToRGB(nn.Module):
    def __init__(self,
                 in_channels: int,
                 w_dim: int,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]] = None
                ) -> None:
        
        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        self.activation = getattr(nn, act_fn)(**act_kwargs)

        self.style = EqualizedLinear(w_dim, 
                                     in_channels, 
                                     bias=bias, 
                                     bias_init=1,
                                     lr_multiplier=lr_multiplier,
                                     act_fn=act_fn,
                                     act_kwargs=act_kwargs)

        self.conv = EqualizedConv2dModulated(in_channels,
                                             out_channels=3,
                                             kernel_size=1,
                                             padding=0,
                                             lr_multiplier=lr_multiplier,
                                             demodulate=False)
        
        self.bias = nn.Parameter(torch.full([3], bias_init, dtype=torch.float32)) if bias else None
        self.bias_scale = lr_multiplier
    
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = self.style(w)
        x = self.conv(x, s)

        if self.bias is not None:
            bias = self.bias * self.bias_scale
            x = x + bias[None, :, None, None]

        return self.activation(x)
    

class MappingNet(nn.Module):
    def __init__(self, 
                 z_dim: int, 
                 w_dim: int, 
                 num_layers: int = 8,
                 lr_multiplier: float = 0.01,
                 act_fn: str = 'LeakyReLU', 
                 act_kwargs: Optional[Dict[Any, Any]] = None
                ):
        
        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        self.activation = getattr(nn, act_fn)(**act_kwargs)

        layers = nn.ModuleList([EqualizedLinear(z_dim, 
                                                w_dim,  
                                                lr_multiplier=lr_multiplier, 
                                                act_fn=act_fn,
                                                act_kwargs=act_kwargs
                                                )])
        layers.extend([EqualizedLinear(w_dim, 
                                    w_dim,  
                                    lr_multiplier=lr_multiplier, 
                                    act_fn=act_fn,
                                    act_kwargs=act_kwargs
                                    ) for i in range(num_layers - 1)])
        
        self.seq = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = z / torch.sqrt(torch.mean(z**2, dim=1, keepdim=True) + 1e-8) # normalize input latent vectors (sampled from unit Gaussian)
        w = self.seq(w)

        # TODO: Implement exponential moving average and truncation

        return w


class StyleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 w_dim: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 noise: bool = True,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:
        
        super().__init__()

        if act_kwargs is None:
            act_kwargs = {}
        self.activation = getattr(nn, act_fn)(**act_kwargs)

        self.style = EqualizedLinear(w_dim, 
                                     in_channels, 
                                     bias=bias, 
                                     bias_init=1,
                                     lr_multiplier=lr_multiplier,
                                     act_fn=act_fn,
                                     act_kwargs=act_kwargs)

        self.conv = EqualizedConv2dModulated(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             lr_multiplier=lr_multiplier,
                                             demodulate=True)
        
        self.bias = nn.Parameter(torch.full([out_channels], bias_init, dtype=torch.float32)) if bias else None
        self.bias_scale = lr_multiplier

        self.noise = nn.Parameter(torch.zeros([], dtype=torch.float32)) if noise else None
        
    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = self.style(w)
        x = self.conv(x, s)

        if self.bias is not None:
            b = self.bias * self.bias_scale
            x = x + b[None, :, None, None]

        if self.noise is not None:
            batch_size = x.shape[0]
            noise = torch.randn([batch_size, 1, *x.shape[-2:]], dtype=torch.float32, device=x.device) * self.noise
            x = x + noise

        return self.activation(x)


class ResolutionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 w_dim: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 noise: bool = True,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:

        super().__init__()

        self.style_block1 = StyleBlock(in_channels,
                                       out_channels,
                                       w_dim,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       bias=bias,
                                       bias_init=bias_init,
                                       lr_multiplier=lr_multiplier,
                                       noise=noise,
                                       act_fn=act_fn,
                                       act_kwargs=act_kwargs) 
        
        self.style_block2 = StyleBlock(out_channels,
                                       out_channels,
                                       w_dim,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       bias=bias,
                                       bias_init=bias_init,
                                       lr_multiplier=lr_multiplier,
                                       noise=noise,
                                       act_fn=act_fn,
                                       act_kwargs=act_kwargs) 
        
        self.tRGB = ToRGB(out_channels,
                          w_dim,
                          bias=bias,
                          bias_init=bias_init,
                          lr_multiplier=lr_multiplier,
                          act_fn=act_fn,
                          act_kwargs=act_kwargs) 

    def forward(self, x: torch.Tensor, ws: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        x = self.style_block1(x, ws[:, 0])
        x = self.style_block2(x, ws[:, 1])
        rgb = self.tRGB(x, ws[:, 2])
        return x, rgb


class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim: int,
                 out_res: int = 256,
                 n_channels: Optional[List[int]] = None,
                 kernel_size: int | tuple[int, int] = 3,
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 noise: bool = True,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:

        super().__init__()

        self.num_blocks = int(np.log2(out_res)) - 1
        self.num_ws = self.num_blocks * 3 - 1

        # If no custom per resolution block output channels were given, build default number 
        # of channels (folowing the structure of ProgressiveGan)
        if n_channels is None:
            n_channels = []
            n = 512
            for i in range(self.num_blocks):
                if i > 3:
                    n = n // 2 if n > 4 else n # Stop n from going below 4 channels
                n_channels.append(n)

        self.constant = nn.Parameter(torch.randn([1, n_channels[0], 4, 4], dtype=torch.float32))
        self.style_block = StyleBlock(n_channels[0],
                                      n_channels[0],
                                      w_dim,
                                      kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      bias=bias,
                                      bias_init=bias_init,
                                      lr_multiplier=lr_multiplier,
                                      noise=noise,
                                      act_fn=act_fn,
                                      act_kwargs=act_kwargs)
        self.tRGB = ToRGB(n_channels[0],
                          w_dim,
                          bias=bias,
                          bias_init=bias_init,
                          lr_multiplier=lr_multiplier,
                          act_fn=act_fn,
                          act_kwargs=act_kwargs)
        
        self.blocks = nn.ModuleList([ResolutionBlock(n_channels[i-1],
                                                     n_channels[i],
                                                     w_dim,
                                                     kernel_size,
                                                     stride=stride,
                                                     padding=padding,
                                                     dilation=dilation,
                                                     bias=bias,
                                                     bias_init=bias_init,
                                                     lr_multiplier=lr_multiplier,
                                                     noise=noise,
                                                     act_fn=act_fn,
                                                     act_kwargs=act_kwargs)
                                                     for i in range(1, self.num_blocks)])
        
    def forward(self, ws: torch.Tensor) -> torch.Tensor:
        batch_size = ws.shape[0]

        x = self.constant.repeat(batch_size, 1, 1, 1)
        x = self.style_block(x, ws[:, 0])
        rgb = self.tRGB(x, ws[:, 1])

        idx = 2
        for block in self.blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear')

            x, tRGB = block(x, ws[:, idx:idx+3])
            rgb += tRGB

            idx += 3

        return F.tanh(rgb)


class Generator(nn.Module):
    def __init__(self,
                 z_dim: int,
                 w_dim: int,
                 out_res: int = 256,
                 synt_kwargs: Optional[Dict[Any, Any]]=None,
                 map_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:

        super().__init__()

        if synt_kwargs is None:
            synt_kwargs = {}

        if map_kwargs is None:
            map_kwargs = {}

        self.syntNet = SynthesisNet(w_dim, out_res, **synt_kwargs)
        self.mapNet = MappingNet(z_dim, w_dim, **map_kwargs)

    def forward(self, z: torch.Tensor, style_mix_prob: float = 0.) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.mapNet(z)
        ws = w.unsqueeze(1).repeat(1, self.syntNet.num_ws, 1)

        # Style mixing
        if style_mix_prob > 0.:
            cutoff = torch.randint(1, ws.shape[1], size=()) if torch.rand([]) < style_mix_prob else None
            if cutoff is not None: 
                ws[:, cutoff:] = self.mapNet(torch.randn_like(z)).unsqueeze(1).repeat(1, ws.shape[1], 1)[:, cutoff:]

        return self.syntNet(ws), ws
    

class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int],
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:
        
        super().__init__()

        self.conv1 = EqualizedConv2d(in_channels,
                                     in_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=bias,
                                     bias_init=bias_init,
                                     lr_multiplier=lr_multiplier,
                                     act_fn=act_fn,
                                     act_kwargs=act_kwargs)
        
        self.conv2 = EqualizedConv2d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=bias,
                                     bias_init=bias_init,
                                     lr_multiplier=lr_multiplier,
                                     act_fn=act_fn,
                                     act_kwargs=act_kwargs)
        
        self.residual_conv = EqualizedConv2d(in_channels,
                                             out_channels,
                                             kernel_size=1,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=bias,
                                             bias_init=bias_init,
                                             lr_multiplier=lr_multiplier,
                                             act_fn=act_fn,
                                             act_kwargs=act_kwargs)

        self.scale = 1 / np.sqrt(2) # Scale back the doubling of signal variance caused by residual connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        res = self.residual_conv(res)

        x = self.conv1(x)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear')

        x = x + res
        return x * self.scale
    

class Discriminator(nn.Module):
    def __init__(self,
                 in_res: int,
                 n_channels: Optional[List[int]] = None,
                 kernel_size: int | tuple[int, int] = 3,
                 stride: int | tuple[int, int] = 1,
                 padding: str | int | tuple[int, int] = 'same',
                 dilation: int | tuple[int, int] = 1,
                 bias: bool = True, 
                 bias_init: float = 0.,
                 lr_multiplier: float = 1.,
                 act_fn: str = 'LeakyReLU',
                 act_kwargs: Optional[Dict[Any, Any]]=None
                ) -> None:
        
        super().__init__()
        
        self.num_blocks = int(np.log2(in_res)) - 1

        # If no custom per discrimator block output channels were given, build default number 
        # of channels (folowing the structure of ProgressiveGan)
        if n_channels is None:
            n_channels = []
            n = 512
            for i in range(self.num_blocks):
                if i > 3:
                    n = n // 2 if n > 4 else n # Stop n from going below 4 channels
                n_channels.append(n)
        n_channels.reverse()

        self.fRGB = EqualizedConv2d(in_channels=3, 
                                    out_channels=n_channels[0],
                                    kernel_size=1,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias,
                                    bias_init=bias_init,
                                    lr_multiplier=lr_multiplier,
                                    act_fn=act_fn,
                                    act_kwargs=act_kwargs)

        self.seq = nn.Sequential(*[DiscriminatorBlock(n_channels[i-1],
                                                    n_channels[i],
                                                    kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    bias=bias,
                                                    bias_init=bias_init,
                                                    lr_multiplier=lr_multiplier,
                                                    act_fn=act_fn,
                                                    act_kwargs=act_kwargs)
                                                    for i in range(1, self.num_blocks)])

        # Post minibatch standard deviation layers
        self.conv = EqualizedConv2d(in_channels=n_channels[-1] + 1, 
                                    out_channels=n_channels[-1],
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias,
                                    bias_init=bias_init,
                                    lr_multiplier=lr_multiplier,
                                    act_fn=act_fn,
                                    act_kwargs=act_kwargs)
        
        self.fc = EqualizedLinear(in_dim=n_channels[-1] * 4 * 4, 
                                  out_dim=n_channels[-1],
                                  bias=bias,
                                  bias_init=bias_init,
                                  lr_multiplier=lr_multiplier,
                                  act_fn=act_fn,
                                  act_kwargs=act_kwargs)
        
        self.out = EqualizedLinear(in_dim=n_channels[-1], 
                                  out_dim=1,
                                  bias=bias,
                                  bias_init=bias_init,
                                  lr_multiplier=lr_multiplier,
                                  act_fn=act_fn,
                                  act_kwargs=act_kwargs)
        
    def minibatch_std(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.std(x, dim=0)
        y = torch.mean(y)
        y = y.repeat(x.shape[0], 1, *x.shape[-2:])

        x = torch.cat([x, y], dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fRGB(x)
        x = self.seq(x)
        x = self.minibatch_std(x) 
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return self.out(x)     


if __name__ == '__main__':
    pass