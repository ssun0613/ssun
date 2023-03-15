import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = x * self.sigmoid(x)
        return x * self.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
                                        nn.Linear(in_channels, in_channels * r),
                                        Swish(),
                                        nn.Linear(in_channels * r, in_channels),
                                        nn.Sigmoid()
                                        )
    def forward(self, x):
        x = self.squeeze(x)
        x_1 = x.view(x.size(0), -1)
        x = self.excitation(x_1)
        x_2 = x.view(x.size(0), x.size(1), 1, 1)
        return x_2

class MBConv(nn.Module):
    expand = 6
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
                                        nn.Conv2d(in_channels, in_channels * MBConv.expand, 1, stride=stride, padding=0, bias=False),
                                        nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
                                        Swish(),
                                        nn.Conv2d(in_channels * MBConv.expand, in_channels * MBConv.expand, kernel_size=kernel_size,
                                                  stride=1, padding=kernel_size//2, bias=False, groups=in_channels*MBConv.expand),
                                        nn.BatchNorm2d(in_channels * MBConv.expand, momentum=0.99, eps=1e-3),
                                        Swish()
                                    )
        self.se = SEBlock(in_channels * MBConv.expand, se_scale)
        self.project = nn.Sequential(
                                        nn.Conv2d(in_channels*MBConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
                                    )

        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

class SepConv(nn.Module):
    expand = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, se_scale=4, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
                                        nn.Conv2d(in_channels * SepConv.expand, in_channels * SepConv.expand, kernel_size=kernel_size,
                                                  stride=1, padding=kernel_size//2, bias=False, groups=in_channels*SepConv.expand),
                                        nn.BatchNorm2d(in_channels * SepConv.expand, momentum=0.99, eps=1e-3),
                                        Swish()
                                    )
        self.se = SEBlock(in_channels * SepConv.expand, se_scale)
        self.project = nn.Sequential(
                                     nn.Conv2d(in_channels*SepConv.expand, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
                                     )
        self.shortcut = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x= x_shortcut + x

        return x

class EfficientNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        repeats = [1, 2, 2, 3, 3, 4, 1]
        stochastic_depth = False
        p = 0.5

        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        self.threshold = opt.threshold
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(32),
                                 self._make_Block(SepConv, repeats=1, in_channels=32, out_channels=16, kernel_size=3, stride=1, se_scale=4),
                                 self._make_Block(MBConv, repeats=2, in_channels=16, out_channels=24, kernel_size=3, stride=2, se_scale=4),
                                 self._make_Block(MBConv, repeats=2, in_channels=24, out_channels=40, kernel_size=5, stride=2, se_scale=4),
                                 self._make_Block(MBConv, repeats=3, in_channels=40, out_channels=80, kernel_size=3, stride=2, se_scale=4),
                                 self._make_Block(MBConv, repeats=2, in_channels=80, out_channels=112, kernel_size=5, stride=1, se_scale=4),
                                 self._make_Block(MBConv, repeats=2, in_channels=112, out_channels=192, kernel_size=5, stride=2, se_scale=4),
                                 self._make_Block(MBConv, repeats=2, in_channels=193, out_channels=320, kernel_size=3, stride=1, se_scale=4),
                                 nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=False),
                                 nn.BatchNorm2d(1280, momentum=0.99, eps=1e-3),
                                 )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(1280, 8)

    def set_input(self, x):
        self.input = x

    def forward(self):
        self.out1 = self.CNN(self.input)
        x = self.GAP(self.out1)
        self.output = self.classification(self.flatten(x))
        return self.output

    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

    def get_output(self):
        return self.output

    def predict(self):
        y = torch.sigmoid(self.output)
        y[y >= self.threshold] = 1
        y[y < self.threshold] = 0
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_networks(self, net, loss_type, device, weight_path=None):
        load_filename = 'efficientnet_{}.pth'.format(loss_type)
        if weight_path is None:
            ValueError('Should set the weight_path, which is the path to the folder including weights')
        else:
            load_path = os.path.join(weight_path, load_filename)
        net = net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)

        state_dict = torch.load(load_path, map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        else:
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net


if __name__ == '__main__':
    print('Debug StackedAE')
    config = Config()
    model = EfficientNet(config.opt)
    model.init_weights()
    x = torch.randn(1, 3, 224, 224)
    model.set_input(x)
    model.forward()