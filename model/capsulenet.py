import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from options.config import Config

class Conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(Conv, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(32),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=32, out_channels=int(out_channels/4), kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(int(out_channels/4)),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=int(out_channels/4), out_channels=int(out_channels/2), kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(int(out_channels/2)),
                                 nn.LeakyReLU(),
                                 nn.Conv2d(in_channels=int(out_channels/2), out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU(),
                                 )

    def forward(self,x):
        return self.cnn(x)

class Primarycaps(nn.Module):
    def __init__(self, in_channels=1024, out_channels=16):
        super(Primarycaps, self).__init__()
        assert in_channels % out_channels == 0

        self.out_channels = out_channels  # dimension of a capsule vector in a primarycaps
        self.in_channels = in_channels
        self.num_primary_caps = int(self.in_channels / self.out_channels)

        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1) for _ in range(self.num_primary_caps)])

    def forward(self, x): # x.shape([1, 1024, 13, 13])
        out = [capsule(x) for capsule in self.capsules] # len(out) = self.num_primary_caps
        out = torch.stack(out, dim=2) # out.shape([1, self.out_channels , self.num_primary_caps, 10, 10])
        num_routes = out.shape[2]*out.shape[3]*out.shape[4] # 64*13*13
        out = out.view(x.size(0), num_routes, -1) # out.shape([1, 64*13*13, 16])
        return self.squash(out)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) # squared_norm.shape([1, 3200, 1])
        output_tensor = squared_norm * input_tensor / ((1.+squared_norm) * torch.sqrt(squared_norm)) # output_tensor.shape([1, 3200, 8])
        return output_tensor

class Digitcaps(nn.Module):
    def __init__(self, in_dim=8, in_caps=32*10*10, out_dim=16, out_caps=8, num_routing=3):
        super(Digitcaps, self).__init__()

        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.num_routing = num_routing

        self.W = nn.Parameter(0.01 * torch.randn(in_caps, out_caps, in_dim, out_dim)) # 0.01!!!!!!!!!!!!!!!!!!!!!!!!!!

    def forward(self, x): # x.shape([1, 3200, 8])
        batch_size = x.size(0) # batch_size=1
        # x: (batch_size, in_caps, in_dim) - bin
        # W: (in_caps, out_caps, in_dim, out_dim) - ijnm
        # W * x - (batch_size, in_caps, out_caps, out_dim) - bijm
        u_hat = torch.einsum('ijnm, bin -> bijm', self.W, x)
        # Detach
        temp_u_hat = u_hat.detach()

        # (batch_size, in_caps, out_caps) - bij
        b = torch.zeros(batch_size, self.in_caps, self.out_caps).to(x.device)
        for route_iter in range(self.num_routing - 1):
            c = F.softmax(b, dim=2)  # -> Softmax along num_caps (batch_size, in_caps, out_caps)
            s = torch.einsum('bij,bijm->bjm', c, temp_u_hat)
            # (batch_size, out_caps, out_dim)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, temp_u_hat)
            b = b + a

        c = F.softmax(b, dim=2)  # -> Softmax along num_caps (batch_size, in_caps, out_caps)
        s = torch.einsum('bij,bijm->bjm', c, u_hat)
        # (batch_size, out_caps, out_dim)
        v = self.squash(s)
        return v

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor


class Decoder(nn.Module):
    def __init__(self, input_width=52, input_height=52, input_channel=3, out_caps=16, num_classes=8):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.out_caps = out_caps
        self.num_classes = num_classes
        self.reconstraction_layers = nn.Sequential(nn.Linear(self.out_caps * self.num_classes, 512),
                                                   nn.ReLU(),
                                                   nn.Linear(512, 1024),
                                                   nn.ReLU(),
                                                   nn.Linear(1024, self.input_width * self.input_height * self.input_channel),
                                                   nn.Sigmoid())

    def forward(self, x):
        classes = torch.sqrt((x**2).sum(dim=2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(8)).to(x.device)

        masked = masked.index_select(dim=0, index=Variable(max_length_indices.data))
        t = (x * masked[:, :, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked

class capsnet(nn.Module):
    def __init__(self, opt):
        super(capsnet, self).__init__()
        self.img_shape = (opt.data_depth, opt.data_width, opt.data_height)
        self.threshold = opt.threshold
        self.conv_layer = Conv(in_channels=opt.data_depth, out_channels=opt.out_channels)
        self.primary_layer = Primarycaps(in_channels=opt.out_channels, out_channels=opt.in_dim)
        self.digit_capsules = Digitcaps(in_dim=opt.in_dim, in_caps=self.calc_in_caps(), out_dim=opt.out_dim, out_caps=8, num_routing=opt.num_routing)
        self.decoder = Decoder(input_width=opt.data_height, input_height=opt.data_width, input_channel=opt.data_depth, out_caps=opt.out_dim, num_classes=8)

        if opt.loss_name == 'mse':
            self.loss_name = nn.MSELoss()
        elif opt.loss_name == 'cross':
            self.loss_name = nn.CrossEntropyLoss()

    def calc_in_caps(self):
        temp_input = torch.zeros(1, self.img_shape[0], self.img_shape[1], self.img_shape[2])
        out = self.conv_layer(temp_input)
        out = self.primary_layer(out)
        return out.shape[1]

    def set_input(self, x):
        self.input = x

    def forward(self, mode='train'):
        self.output = self.digit_capsules(self.primary_layer(self.conv_layer(self.input)))
        if mode == 'train':
            self.reconstructions, masked = self.decoder(self.output)

    def predict(self):
        y = torch.norm(self.output, dim=2)
        y[y >= self.threshold] = 1
        y[y < self.threshold] = 0
        return y

    def margin_loss(self, v_j, label):
        batch_size = v_j.size(0)

        v_j_norm = torch.norm(v_j, dim=2, keepdim=True)

        left = F.relu(0.9 - v_j_norm).view(batch_size, -1) ** 2
        right = F.relu(v_j_norm - 0.1).view(batch_size, -1) ** 2

        loss = label * left + 0.5 * (1 - label) * right
        loss = loss.sum(dim=1).mean()

        if torch.isnan(loss):
            print('loss is nan...')

        return loss

    def margin_loss_ssun(self, x, labels):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9-v_c).view(batch_size, -1)
        right = F.relu(v_c-0.1).view(batch_size, -1)

        loss = labels*left + 0.5*(1.0-labels)*right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.loss_name(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))

        return loss*0.0005

    def get_loss(self, label):
        return self.margin_loss(self.output, label) + self.reconstruction_loss(self.input, self.reconstructions)

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

    def load_networks(self, net, loss_type , device, weight_path=None):
        load_filename = 'capsnet_{}_{}_{}.pth'.format(loss_type, Config().opt.in_dim, Config().opt.out_channels)
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
    config=Config()
    model = capsnet(config.opt)
    x = torch.rand(1, 3, 52, 52)
    model.set_input(x)
    model.forward()
