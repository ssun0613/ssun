import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        out = int(out_channels/4)
        self.Conv = nn.Sequential(nn.Conv2d(in_channels, out, kernel_size = (1, 1), stride = (1, 1), padding = padding),
                                  nn.BatchNorm2d(out),
                                  nn.ReLU(),
                                  nn.Conv2d(out, out, kernel_size, stride = (1, 1), padding = 1),
                                  nn.BatchNorm2d(out),
                                  nn.ReLU(),
                                  nn.Conv2d(out, out_channels, kernel_size = (1, 1), stride = (1, 1), padding = 0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(),
                                  )
    def forward(self, x):
        return self.Conv(x)

class shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(shortcut, self).__init__()
        self.shortcut = Conv(in_channels, out_channels, kernel_size, stride, padding)

        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(nn.Conv2d(in_channels, out_channels, stride = stride, padding = 1, kernel_size = (1, 1)),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU())
        else:
            self.projection = nn.Sequential()

    def forward(self, x):
        out = self.shortcut(x)
        return out + self.projection(x)

class Resnet(nn.Module):
    def __init__(self, opt):
        super(Resnet, self).__init__()
        self.threshold = opt.threshold
        self.Resnet = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, stride = (1, 1), kernel_size = (3, 3)),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=(3, 3)),
                                    shortcut(in_channels = 64, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1),
                                    shortcut(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 1),
                                    shortcut(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 512, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 512, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 1),
                                    shortcut(in_channels = 1024, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 1024, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 1024, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 1024, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 1024, out_channels = 1024, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 1024, out_channels = 2048, kernel_size = (3, 3), stride = 1, padding = 1),
                                    shortcut(in_channels = 2048, out_channels = 2048, kernel_size = (3, 3), stride = 1, padding = 0),
                                    shortcut(in_channels = 2048, out_channels = 2048, kernel_size = (3, 3), stride = 1, padding = 0),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(in_features=2048, out_features=8)
                                    )

    def set_input(self, x):
        self.input = x

    def forward(self):
        self.output = self.Resnet(self.input)

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
        load_filename = 'resnet_2_{}.pth'.format(loss_type)
        if weight_path is None:
            ValueError('Should set the weight_path, which is the path to the folder including weights')
        else:
            load_path = os.path.join(weight_path, load_filename)
        net = net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        else:
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net


if __name__ == '__main__':
    config = Config()

    model = Resnet(config.opt)
    model.init_weights()
    x = torch.rand(1, 3, 52, 52)
    model.set_input(x)
    model.forward()
    output = model.get_output()
    print(output.shape)
    print('Debug StackedAE')
