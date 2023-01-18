import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

# https://deep-learning-study.tistory.com/563

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
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        se_scale = 4
        stochastic_depth = False
        p = 0.5

        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        self.threshold = opt.threshold
        self.CNN = nn.Sequential(
                                    nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(channels[0]),
                                    self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale),
                                    self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale),
                                    self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale),
                                    self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale),
                                    self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale),
                                    self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale),
                                    self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale),
                                    nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
                                    )

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(channels[8], 8)

        if 'GradCAM' in Config().opt.show_cam:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]
            self.hookB = [Hook(layers[1], backward=True) for layers in list(self._modules.items())]
        else:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]

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
        load_filename = 'efficientnet_epoch.pth'
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

class EfficientNet_GAIN(nn.Module):
    def __init__(self, num_classes=16, width_coef=1., depth_coef=1., scale=1., dropout=0.2, se_scale=4, stochastic_depth=False, p=0.5):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_size = [3, 3, 5, 3, 5, 5, 3]
        depth = depth_coef
        width = width_coef

        if stochastic_depth:
            self.p = p
            self.step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.p = 1
            self.step = 0

        self.CNN = nn.Sequential(
                                    nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(channels[0]),
                                    self._make_Block(SepConv, repeats[0], channels[0], channels[1], kernel_size[0], strides[0], se_scale),
                                    self._make_Block(MBConv, repeats[1], channels[1], channels[2], kernel_size[1], strides[1], se_scale),
                                    self._make_Block(MBConv, repeats[2], channels[2], channels[3], kernel_size[2], strides[2], se_scale),
                                    self._make_Block(MBConv, repeats[3], channels[3], channels[4], kernel_size[3], strides[3], se_scale),
                                    self._make_Block(MBConv, repeats[4], channels[4], channels[5], kernel_size[4], strides[4], se_scale),
                                    self._make_Block(MBConv, repeats[5], channels[5], channels[6], kernel_size[5], strides[5], se_scale),
                                    self._make_Block(MBConv, repeats[6], channels[6], channels[7], kernel_size[6], strides[6], se_scale),
                                    nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
                                    nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
                                    )

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classification = nn.Linear(channels[8], num_classes)

        if 'GradCAM' in Config().opt.show_cam:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]
            self.hookB = [Hook(layers[1], backward=True) for layers in list(self._modules.items())]
        else:
            self.hookF = [Hook(layers[1], backward=False) for layers in list(self._modules.items())]

    def set_input(self, x, label):
        self.input = x
        self.label = label

    def _forward(self, input):
        CNN = self.CNN(input)
        x = self.GAP(CNN)
        self.output = self.classification(self.flatten(x))
        return self.output

    def forward(self):
        self.out1 = self._forward(self.input)
        self.attMap = self._make_att_map(self.out1)
        self.maskedImg = self._make_masked_img(self.attMap)
        with torch.no_grad():
            self.out2 = self._forward(self.maskedImg)

    def _make_Block(self, block, repeats, in_channels, out_channels, kernel_size, stride, se_scale):
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels, out_channels, kernel_size, stride, se_scale, self.p))
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

    def _make_att_map(self, out1):
        layer_index = np.argmax(np.array([name == 'CNN' for name in self._modules.keys()], dtype=np.int))
        feature_maps = self.hookF[layer_index].output

        class_1 = out1[:, 0:8].unsqueeze(dim=2)
        class_0 = out1[:, 8:].unsqueeze(dim=2)

        out1 = torch.concat([class_1, class_0], dim=2)

        # score = 1 / (1 + torch.abs(self.label - out1))
        score = (out1 * self.label)[:, :, 1]
        score.mean().backward(retain_graph=True)

        gradient = self.hookB[layer_index].output[0]
        weighted = gradient.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (gradient.shape[2] * gradient.shape[3])
        attention = F.relu((feature_maps.detach() * weighted.detach()).sum(dim=1).unsqueeze(dim=1))
        attention = F.interpolate(attention, size=(self.input.shape[2], self.input.shape[3]))

        attention = (attention - attention.min()) / (attention.max() - attention.min())

        return attention

    def _make_masked_img(self, attention):
        attention[attention >= 0.4] = 1
        attention[attention < 0.4] = 0
        maskedImg = self.input * (1 - attention) + attention

        return maskedImg

    def get_outputs(self):
        class_1 = self.out1[:,0:8].unsqueeze(dim=2)
        class_0 = self.out1[:,8:].unsqueeze(dim=2)
        self.out1 = torch.concat([class_1,class_0],dim=2)
        return self.out1

    def get_outputs_2(self):
        class_1 = self.out2[:, 0:8].unsqueeze(dim=2)
        class_0 = self.out2[:, 8:].unsqueeze(dim=2)
        self.out2 = torch.concat([class_1, class_0], dim=2)
        return self.out2

    def get_output(self):
        return self.output

    def predict(self):
        y = F.softmax(self.out1, dim=2)
        return y

    def get_att_maskmap(self):
        return self.attMap, self.maskedImg

    def accuracy(self, x, t):
        import numpy as np

        y = torch.argmax(t, dim=2)
        all_defect_idx = np.array(range(x.shape[1]))

        pos_collect = 0
        neg_collect = 0
        pos_all = 0
        neg_all = 0
        for batch_idx in range(x.shape[0]):
            temp_x = x[batch_idx].cpu().detach().numpy()
            temp_y = y[batch_idx].cpu().detach().numpy()
            defect_idx = np.where(temp_y == 1)[0]
            no_defect_idx = []
            for i in range(x.shape[1]):
                if np.sum(defect_idx == all_defect_idx[i]) == 0:
                    no_defect_idx.append(all_defect_idx[i])
            no_defect_idx = np.array(no_defect_idx)

            pos_collect += np.sum(temp_y[defect_idx] == temp_x[defect_idx])
            neg_collect += np.sum(temp_y[no_defect_idx] == temp_x[no_defect_idx])
            pos_all += len(temp_y[defect_idx])
            neg_all += len(temp_y[no_defect_idx])

        pos_acc = pos_collect / float(pos_all)
        neg_acc = neg_collect / float(neg_all)

        return pos_acc

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
        load_filename = 'efficientnet_gain_epoch.pth'
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

def efficientnet_b0(num_classes=10):
    return EfficientNet(num_classes=16, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)

def efficientnet_gain_b0(num_classes=10):
    return EfficientNet_GAIN(num_classes=16, width_coef=1.0, depth_coef=1.0, scale=1.0,dropout=0.2, se_scale=4)

if __name__ == '__main__':
    print('Debug StackedAE')
    model = efficientnet_gain_b0()
    model.init_weights()
    x = torch.randn(1, 3, 224, 224)
    y = torch.rand(1, 8, 2)
    model.set_input(x, y)
    model.forward()