import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

class LeNet(nn.Module):
    def __init__(self, opt):
        super(LeNet, self).__init__()
        self.threshold = opt.threshold
        self.CNN = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.Tanh(),
                                 nn.AvgPool2d(kernel_size=(2, 2)),
                                 nn.Tanh(),
                                 nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.Tanh(),
                                 nn.AvgPool2d(kernel_size=(2, 2)),
                                 nn.Tanh(),
                                 nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                 nn.Tanh()
                                 )
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(120, 8)

    def set_input(self, x):
        self.input = x
    def forward(self):
        CNN = self.CNN(self.input)
        GAP = self.GAP(CNN)
        self.output = self.classification(self.flatten(GAP))

    def get_output(self):
        return self.output

    def predict(self):
        y = torch.sigmoid(self.output)
        y[y >= self.threshold] = 1
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
        load_filename = 'lenet_{}.pth'.format(loss_type)
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
        net.load_state_dict(state_dict['net'])

        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
            net.load_state_dict(state_dict['net'])
        print('load completed...')

        return net

if __name__ == '__main__':
    print('Debug StackedAE')
    model = LeNet()
    model.init_weights()
    x = torch.rand(1, 3, 52, 52)
    model.set_input(x)
    model.forward()
    output = model.get_outputs()
    print(output.shape)
    print('Debug StackedAE')
