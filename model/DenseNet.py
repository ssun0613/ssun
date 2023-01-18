import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

class Hook():
    def __init__(self, module, backward):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def closs(self):
        self.hook.remove()

class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBottleneck, self).__init__()
        self.DenseBottleneck = nn.Sequential(nn.BatchNorm2d(in_channels),
                                             nn.ReLU(),
                                             nn.Conv2d(in_channels, 4 * out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0),
                                             nn.BatchNorm2d(4 * out_channels),
                                             nn.ReLU(),
                                             nn.Conv2d(4 * out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.DenseBottleneck(x), self.shortcut(x)], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer):
        super(DenseBlock, self).__init__()
        self.DenseBlock = self.block(in_channels, out_channels, num_layer)

    def block(self, in_channels, out_channels, num_layer):
        block = []
        for i in range(num_layer):
            block.append(DenseBottleneck((in_channels + i * out_channels), out_channels))
        return nn.Sequential(*block)

    def forward(self, x):
        return self.DenseBlock(x)

class Trans_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Trans_layer, self).__init__()
        self.Trans_layer = nn.Sequential(nn.BatchNorm2d(in_channels),
                                         nn.ReLU(),
                                         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                         nn.ReLU(),
                                         nn.AvgPool2d(kernel_size = (2, 2), stride = 2))

    def forward(self, x):
        return self.Trans_layer(x)

class DenseNet(nn.Module):
    def __init__(self, opt):
        super(DenseNet, self).__init__()

        self.threshold = opt.threshold
        self.k = 12
        self.CNN = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 2 * self.k, kernel_size = (3, 3), stride = 1),
                                 nn.BatchNorm2d(2 * self.k),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size = (3, 3), stride = 1),
                                 DenseBlock(in_channels = 2 * self.k, out_channels = self.k, num_layer = 6),
                                 Trans_layer(in_channels = 2 * self.k + 6 * self.k, out_channels = self.k, kernel_size = (1, 1), stride = 1, padding = 0),
                                 DenseBlock(in_channels = self.k, out_channels = self.k, num_layer = 12),
                                 Trans_layer(in_channels = self.k + 12 * self.k, out_channels = self.k, kernel_size = (1, 1), stride = 1, padding = 0),
                                 DenseBlock(in_channels = self.k, out_channels = self.k, num_layer = 24),
                                 nn.BatchNorm2d(self.k + 24 * self.k),
                                 nn.ReLU()
                                 )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(self.k + 24 * self.k , 8)

        if 'GradCAM' in Config().opt.show_cam:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]
            self.hookB = [Hook(layers[1], backward=True) for layers in list(self._modules.items())]
        else:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]

    def set_input(self, x):
        self.input = x

    def forward(self):
        CNN = self.CNN(self.input)
        GAP = self.GAP(CNN)
        self.output = self.classification(self.flatten(GAP))

    def predict(self):
        y = torch.sqrt((self.output ** 2).sum(dim=2, keepdim=True))
        y[y > self.threshold] = 1
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

    def load_networks(self, net, net_type, device, weight_path=None):
        load_filename = 'Densenet_epoch_{}.pth'.format(net_type)
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
    print('Debug StackedAE')
    model = DenseNet()
    model.init_weights()
    x = torch.rand(1, 3, 52, 52)
    model.set_input(x)
    model.forward()
    output_1 = model.get_outputs()
    output = model.get_output()
    print(output.shape)
    print('Debug StackedAE')