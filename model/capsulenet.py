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
        self.in_channels = in_channels
        self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) for _ in range(num_capsule)])

    def forward(self, x): # x.shape([1, 256, 44, 44])
        out = [capsule(x) for capsule in self.capsules] # len(out)=8
        out = torch.stack(out, dim=1) # out.shape([1, 8, 32, 18, 18])
        out = out.view(x.size(0), self.num_routes, -1) # out.shape([1, 1152, 72])
        return self.squash(out)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True) # squared_norm.shape([1, 1152, 1])
        output_tensor = squared_norm * input_tensor / ((1.+squared_norm) * torch.sqrt(squared_norm)) # output_tensor.shape([1, 1152, 72])
        return output_tensor

class Digitcaps(nn.Module):
    def __init__(self, num_capsules = 10, num_routes=32*6*6, in_channels=1, out_channels=72):
        super(Digitcaps, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_capsules = num_capsules
        self.num_routes = num_routes

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, in_channels, out_channels))

    def forward(self, x): # x.shape([1, 1152, 72])
        batch_size = x.size(0) # batch_size=1
        x = torch.stack([x]*self.num_capsules, dim=2).unsqueeze(4) # x.shape([1, 1152, 10, 72, 1])

        W = torch.cat([self.W]*batch_size, dim=0) # W.shape([1, 1152, 10, 1, 72])
        u_hat = torch.matmul(W,x) # u_hat.shape([1, 1152, 10, 1, 1])

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1)) # b_ij.shape([1, 1152, 10, 1])

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
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.input_width * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x**2).sum(dim=2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))

        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x*masked[:,:,None,None]).view(x.size(0),-1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked

class capsnet(nn.Module):
    def __init__(self):
        super(capsnet, self).__init__()
        self.conv_layer = Conv(in_channels=1, out_channels=256, kernel_size=9)
        self.primary_layer = Primarycaps(num_capsule=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32*6*6)
        self.digit_capsules = Digitcaps(num_capsules = 10, num_routes=32*6*6, in_channels=1, out_channels=72)
        self.decoder = Decoder(input_width=28, input_height=28, input_channel=1)



        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_layer(self.conv_layer(data)))
        reconstructions, masked = self.decoder(output, data)

        return output, reconstructions, masked

    def margin_loss(self, x, labels):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9-v_c).view(batch_size, -1)
        right = F.relu(v_c-0.1).view(batch_size, -1)

        loss = labels*left + 0.5*(1.0-labels)*right
        loss=loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))

        return loss*0.0005

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

if __name__ == '__main__':
    print('Debug StackedAE')
    model = capsnet()
    x = torch.rand(1, 1, 52, 52)
    y = torch.rand(1,16)
    output, reconstructions, masked = model.forward(x)