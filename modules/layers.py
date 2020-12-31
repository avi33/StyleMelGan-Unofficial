import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def weights_init(m):
    if isinstance(m, torch.nn.Conv1d):
        nn.init.xavier_normal_(m.weight)    
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)


class SafeSoftMax(nn.Module):
    def __init__(self):
        super(SafeSoftMax, self).__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=1)[0]
        x = torch.softmax(x-x_max, dim=1)
        return x


class GatedTanh(nn.Module):
    def __init__(self):
        super(GatedTanh, self).__init__()

    def forward(self, x):
        s, t = x.split(x.shape[1] // 2, dim=1)
        y = torch.sigmoid(s) * torch.tanh(t)
        return y


class GatedWNConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(GatedWNConv1d, self).__init__()        
        model = [
            WNConv1d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
            GatedTanh()
        ]
        self.conv = nn.Sequential(*model)
    
    def forward(self, x):
        y = self.conv(x)
        return y

class Tade(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tade, self).__init__()        
        self.inst_norm = nn.InstanceNorm1d(in_channels)        
        self.conv_post_interp = GatedWNConv1d(in_channels, in_channels, kernel_size=9, stride=1, padding=9//2)
        self.conv_gamma_beta = GatedWNConv1d(in_channels, out_channels*2, kernel_size=9, stride=1, padding=9//2)
        self.post_tade = GatedWNConv1d(out_channels, out_channels, kernel_size=9, stride=1, padding=4*2, dilation=2)

    def forward(self, mn):
        m = mn[0]
        n = mn[1]
        m2 = F.interpolate(m, scale_factor=2, mode='nearest')
        n2 = F.interpolate(n, scale_factor=2, mode='nearest')
        m2 = self.conv_post_interp(m2)
        gamma_beta = self.conv_gamma_beta(m2)
        gamma, beta = gamma_beta.split(gamma_beta.size(1) // 2, dim=1)        
        c = self.inst_norm(n2)                      
        y = c * gamma + beta
        y = self.post_tade(y)
        return m2, y
    

if __name__ == "__main__":
    SSM = SafeSoftMax()
    x = 1e-12*torch.rand(256, 1024)
    x = x.sum(0).view(1, -1)
    y1 = SSM(x)
    y2 = F.softmax(x, dim=1)
    print((y1-y2).sum())
