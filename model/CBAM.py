import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class channels(nn.Module):
    def __init__(self, channel_size):
        super(channels, self).__init__()
        self.channels = channel_size
        self.max = nn.AdaptiveMaxPool2d(1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(self.channels, int(self.channels*0.5)),
                                 nn.ReLU(),
                                 nn.Linear(int(self.channels*0.5), self.channels)
                                 )

    def forward(self, input):
        max = self.mlp(self.max(input))
        avg = self.mlp(self.avg(input))
        mc = torch.sigmoid(max + avg).unsqueeze(dim=-1).unsqueeze(dim=-1) * input
        return mc

class spatial(nn.Module):
    def __init__(self):
        super(spatial, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), padding=(1, 1)),
                                  nn.BatchNorm2d(1)
                                  )

    def forward(self, input):
        max = torch.max(input, dim=1, keepdim=True)[0]
        avg = torch.mean(input, dim=1, keepdim=True)
        concat = torch.concat([max,avg], dim=1)
        ms = torch.sigmoid(self.conv(concat)) * input
        return ms

class CBAM(nn.Module):
    def __init__(self, channel_size):
        super(CBAM, self).__init__()
        self.channels = channels(channel_size)
        self.spatial = spatial()

    def forward(self, input):
        output = self.spatial(self.channels(input))
        return output


if __name__ == '__main__':
    print('Debug StackedAE')
    # model = channels(256)
    # model = spatial()
    model = CBAM(256)
    # model.init_weights()
    x = torch.rand(4, 256, 52, 52)
    # model.set_input(x)
    model.forward(x)
    # output_1 = model.get_outputs()
    # output = model.get_output()
    # print(output.shape)
    print('Debug StackedAE')