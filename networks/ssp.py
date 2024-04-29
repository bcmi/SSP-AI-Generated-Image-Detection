import torch
from torch import nn
from networks.resnet import resnet50
from networks.srm_conv import SRMConv2d_simple
import torch.nn.functional as F


class ssp(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        self.srm = SRMConv2d_simple()
        self.disc = resnet50(pretrained=True)
        self.disc.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.srm(x)
        x = self.disc(x)
        return x


if __name__ == '__main__':
    model = ssp(pretrain=True)
    print(model)
