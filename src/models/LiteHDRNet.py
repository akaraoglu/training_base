import torch
import torch.nn as nn
import torch.nn.functional as F


# Simplified Activation Selection
def get_activation(act_type='leakyrelu'):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


# Simplified Convolution Block with Group Convolution
class GroupConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act_type='leakyrelu', groups=1):
        super(GroupConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, groups=groups)
        self.act = get_activation(act_type)

    def forward(self, x):
        return self.act(self.conv(x))


# Simplified Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels, act_type='leakyrelu'):
        super(ResBlock, self).__init__()
        self.conv1 = GroupConvBlock(channels, channels, kernel_size=3, act_type=act_type)
        self.conv2 = GroupConvBlock(channels, channels, kernel_size=3, act_type=act_type)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# Simplified Multi-Scale Network
class MultiScaleNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=32, act_type='leakyrelu'):
        super(MultiScaleNet, self).__init__()

        self.conv1 = GroupConvBlock(in_channels, nf, kernel_size=3, act_type=act_type)
        self.resblock = ResBlock(nf, act_type=act_type)
        self.conv2 = nn.Conv2d(nf, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resblock(x)
        x = self.conv2(x)
        return x


# Simplified LiteHDRNet
class LiteHDRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=32, act_type='leakyrelu'):
        super(LiteHDRNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, nf, kernel_size=1),
            get_activation(act_type),
            nn.Conv2d(nf, nf * 2, kernel_size=1),
            get_activation(act_type),
            nn.Conv2d(nf * 2, nf, kernel_size=1),
            get_activation(act_type),
            nn.Conv2d(nf, out_channels, kernel_size=1)
        )

        self.msnet = MultiScaleNet(in_channels=out_channels, out_channels=out_channels, nf=nf, act_type=act_type)

    def forward(self, x):
        out1 = self.mlp(x[0])
        out2 = self.msnet(out1)
        return out2
