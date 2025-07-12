import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_ch, out_ch, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.ReflectionPad2d(padding),
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=0, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True)
    )

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            conv_block(ch, ch, kernel=3, stride=1, padding=1),
            conv_block(ch, ch, kernel=3, stride=1, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)

class SimpleTransformerNet(nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = strength

        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3,  32, 9, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True)
        )
        self.enc2 = conv_block(32, 64, kernel=3, stride=2, padding=1)
        self.enc3 = conv_block(64, 128, kernel=3, stride=2, padding=1)

        self.res = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x, alpha=None):
        x0 = x * 2 - 1
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        y  = self.res(x3)
        y  = self.dec2(y)
        y  = self.dec1(y)
        y  = self.final(y)
        y  = (y + 1) / 2
        if alpha is None: alpha = self.strength
        return alpha * y + (1 - alpha) * x
