import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.conv(x))

class Primarycaps(nn.Module):
    def __init__(self, num_capsule=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32*6*6):
        super(Primarycaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(num_capsule)])

    def forward(self,x):
        out = [capsule(x) for capsule in self.capsules]
        out = torch.stack(out, dim=1)
        out = out.view(x.size(0), self.num_routes, -1)
        return self.squash(out)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1.+squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Digitcaps(nn.Module):
    def __init__(self, num_capsules = 10, num_routes=32*6*6, in_channels=8, out_channels=16):
        super(Digitcaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x]*self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W]*batch_size, dim=0)
        u_hat = torch.matmul(W,x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))

        num_iterations=3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij]*batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j]*self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
