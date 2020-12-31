import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
from modules.layers import WNConv1d, WNConvTranspose1d, Tade, GatedWNConv1d, weights_init


class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        noise_dim = 128        
        model = []
        model += [
            WNConvTranspose1d(noise_dim, noise_dim//2, kernel_size=16, stride=8, padding=4, output_padding=0),
            nn.LeakyReLU(0.2, True),
            WNConvTranspose1d(noise_dim//2, noise_dim//2, kernel_size=8, stride=4, padding=2, output_padding=0),
            nn.LeakyReLU(0.2, True),            
        ]        
        self.noise_up = nn.Sequential(*model)

        hidden_dim = noise_dim // 2        
        model = []
        for _ in range(8):
            model += [
                Tade(input_size, hidden_dim)
            ]
        self.tade_up = nn.Sequential(*model)
        self.final = nn.Sequential(*
            [WNConv1d(hidden_dim, 1, kernel_size=9, stride=1, padding=4, dilation=1),
            nn.Tanh()])
        self.apply(weights_init)


    def forward(self, m, n):        
        n = self.noise_up(n)
        m, x = self.tade_up((m, n))
        y = self.final(x)
        return y


if __name__ == "__main__":    
    from modules.stft import Audio2Mel
    b = 1
    n_frames = 10
    fft = Audio2Mel()
    m = fft(torch.randn(2, 1, 8192*n_frames))
    n = torch.randn(b, 128, n_frames)    
    G = Generator(80)
    y = G(m,n)
    print(y.shape)