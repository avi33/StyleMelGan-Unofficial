import torch
import torch.nn as nn
import torchaudio
import numpy as np
import random


class RandomTimeShift(object):
    def __init__(self, p, max_time_shift=None):
        self.p = p
        self.max_time_shift = max_time_shift
    
    def forward(self, sample):
        if torch.rand(1) < self.p:
            if self.max_time_shift is None:
                self.max_time_shift = sample.shape[-1] // 20
            n_shift = random.randint(0, self.max_time_shift)
            if n_shift == 0:
                return sample
            else:
                pad = torch.zeros(n_shift, dtype=sample.dtype)
                direcion = random.random()
                if direcion > 0.5:
                    sample = torch.cat((pad, sample[:-n_shift]), dim=-1)
                else:
                    sample = torch.cat((sample[n_shift:], pad), dim=-1)
        return sample


class RandomAmp(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
    def __call__(self, sample):
        amp = torch.FloatTensor(1).uniform_(self.low, self.high)
        sample = sample * amp
        return sample


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):
        if torch.rand(1) > self.p:
            inv_idx = torch.arange(sample.size(0)-1, -1, -1).long()
            sample = sample[inv_idx]
        return sample


class RandomAdd180Phase(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, sample):
        if torch.rand(1) > self.p:     
            sample = sample * (-1)
        return sample


class RandomQuantNoise(object):
    def __init__(self, n_bits=16, p=0.5):
        self.p = p
        self.n_bits = n_bits
    def __call__(self, sample):
        if torch.rand(1) > self.p:     
            sample = torch.round(sample * 2**self.n_bits) / 2**self.n_bits
        return sample


class RandomAddAWGN(object):
    def __init__(self, snr_db=30, p=0.5):
        self.p = p
        self.snr_db = snr_db   
    def __call__(self, sample):
        if torch.rand(1) > self.p:     
            s = torch.sqrt(torch.mean(sample**2))
            sgm = s * 10**(-self.snr_db/20.)
            w = torch.randn_like(sample) * sgm
            sample = sample + w
        return sample


class RandomAddSine(object):
    def __init__(self, fs, snr_db=30, p=0.5):
        self.p = p
        self.fs = fs
        self.snr_db = snr_db

    def __call__(self, sample):
        if torch.rand(1) > self.p:     
            n = torch.arange(0, sample.shape[0], 1)
            f = 50 + 3*torch.randn(1)
            t = n*1./self.fs
            s = torch.sqrt(torch.mean(sample**2))
            sgm = s * 10**(-self.snr_db/20.)  
            b = sgm*s*torch.sin(2*np.pi*f*t+torch.rand(1)*np.pi)
            sample = sample + b           
        return sample


class AudioAugs(object):
    def __init__(self, augs, fs):        
        self.random_amp = RandomAmp(low=0.3, high=1)
        self.random_flip = RandomFlip(p=0.5)
        self.random_neg = RandomAdd180Phase(p=0.5)
        self.random_quantnoise = RandomQuantNoise(n_bits=16, p=0.5)
        self.awgn = RandomAddAWGN(snr_db=30, p=0.5)
        self.sine = RandomAddSine(fs=fs, snr_db=30, p=0.5)
        self.tshift = RandomTimeShift(p=0.5, max_time_shift=None)
        self.augs = augs
    
    def __call__(self, sample):
        for aug in self.augs:
            if aug=='amp':
                sample = self.random_amp(sample)
            elif aug=='flip':
                sample = self.random_flip(sample)
            elif aug=='neg':
                sample = self.random_neg(sample)
            elif aug=='quant':
                sample = self.random_quantnoise(sample)
            elif aug=='sine':
                sample = self.sine(sample)
            elif aug=='awgn':
                sample = self.awgn(sample)
            elif aug == 'tshift':
                sample = self.tshift(sample)
        return sample

if __name__ == "__main__":
    RA = RandomAmp(0.3, 1.)
    RF = RandomFlip(0.5)
    RN = RandomAdd180Phase(0.5)
    RQ = RandomQuantNoise(16, 0.5)
    RAW = RandomAddAWGN(30, 0.5)
    RS = RandomAddSine(30, 0.5)
    x = torch.randn(4)
    y1 = RF(x)
    y2 = RN(x)
    y3 = RQ(x)
    y4 = RA(x)
    y5 = RS(x)
    print(x-y4)