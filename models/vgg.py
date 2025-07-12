import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

class VGG16Features(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        self.relu1_2 = nn.Sequential(*vgg_pretrained[:4])
        self.relu2_2 = nn.Sequential(*vgg_pretrained[4:9])
        self.relu3_3 = nn.Sequential(*vgg_pretrained[9:16])
        self.relu4_3 = nn.Sequential(*vgg_pretrained[16:23])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = {}
        out['relu1_2'] = self.relu1_2(x)
        out['relu2_2'] = self.relu2_2(out['relu1_2'])
        out['relu3_3'] = self.relu3_3(out['relu2_2'])
        out['relu4_3'] = self.relu4_3(out['relu3_3'])
        return namedtuple("VGGOutputs", out.keys())(*out.values())
