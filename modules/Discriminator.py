import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import WNConv1d, weights_init
from modules.pqmf import PQMF

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        super(DBlock, self).__init__()
        model = [
            nn.AvgPool1d(downsample_factor, stride=downsample_factor),
            nn.LeakyReLU(0.2, True),
            WNConv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            WNConv1d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2)
        ]
        self.block = nn.Sequential(*model)

        model = [
            WNConv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(downsample_factor, stride=downsample_factor)
        ]
        self.residual = nn.Sequential(*model)
    
    def forward(self, x):
        y = self.block(x) + self.residual(x)
        return y


class Discriminator(nn.Module):
    def __init__(self, n_subbands=4):
        super(Discriminator, self).__init__()
        self.pqmf = PQMF(subbands=n_subbands)
        self.pre_downsample = WNConv1d(n_subbands, 16, kernel_size=15, stride=1, padding=7)
        model = [
            DBlock(16, 64, 4),
            DBlock(64, 256, 4),
            DBlock(256, 512, 4),
            DBlock(512, 512, 1),
        ]
        self.db_stack = nn.Sequential(*model)
        self.final = WNConv1d(512, 1, kernel_size=3, stride=1, padding=1)         

    def forward(self, x):
        fb = self.pqmf.analysis(x)
        y = self.pre_downsample(fb)
        y = self.db_stack(y)
        y = self.final(y)
        return y


class RWDiscriminator(nn.Module):
    def __init__(self, num_D):
        super(RWDiscriminator, self).__init__()
        self.windows = [4096, 2048, 1024]
        model = nn.ModuleDict()
        for i in range(num_D):
            model["disc_%d" % i] = Discriminator()
        self.model = model        
        self.apply(weights_init)
    def forward(self, x):
        results = []
        for i, (key, disc) in enumerate(self.model.items()):
            w = self.windows[i]
            if i == 0:                
                results.append(disc(x))
            else:                
                ii = np.random.randint(x.shape[-1]-w-1)
                xwin = x[:, :, ii:ii + w]                
                results.append(disc(xwin))
        return results

if __name__ == "__main__":
    b = 2
    x = torch.randn(2, 1, 8192)
    D = RWDiscriminator(4)
    y = D(x)
    for yy in y:
        print(yy.shape)        