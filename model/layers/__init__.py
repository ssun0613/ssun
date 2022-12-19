import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels, pool_features):
        super(InceptionA, self).__init__()
        conv_block = BasicConv2d

        out_channel_branch_1x1 = int((out_channels - pool_features - int(out_channels*3/7))/2)
        out_channel_branch_3x3_r = out_channel_branch_1x1
        out_channel_branch_3x3_l = out_channels - pool_features - 2*out_channel_branch_1x1

        self.branch1x1 = conv_block(in_channels=in_channels, out_channels=out_channel_branch_1x1, kernel_size=1)

        self.branch3x3_r_1 = conv_block(in_channels=in_channels, out_channels=int(out_channel_branch_3x3_r/2), kernel_size=1)
        self.branch3x3_r_2 = conv_block(in_channels=int(out_channel_branch_3x3_r/2), out_channels=out_channel_branch_3x3_r,
                                        kernel_size=3, padding=[1, 1])

        self.branch3x3_l_1 = conv_block(in_channels=in_channels, out_channels=int(out_channel_branch_3x3_l / 2),
                                        kernel_size=1)
        self.branch3x3_l_2 = conv_block(in_channels=int(out_channel_branch_3x3_l / 2), out_channels=out_channel_branch_3x3_l,
                                        kernel_size=3, padding=[1, 1])
        self.branch3x3_l_3 = conv_block(in_channels=out_channel_branch_3x3_l, out_channels=out_channel_branch_3x3_l,
                                        kernel_size=3, padding=[1, 1])

        self.branch_pool = conv_block(in_channels=in_channels, out_channels=pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3_r = self.branch3x3_r_1(x)
        branch3x3_r = self.branch3x3_r_2(branch3x3_r)

        branch3x3_l = self.branch3x3_l_1(x)
        branch3x3_l = self.branch3x3_l_2(branch3x3_l)
        branch3x3_l = self.branch3x3_l_3(branch3x3_l)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=[1, 1])
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3_r, branch3x3_l, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels, out_channels, pool_features):
        super(InceptionB, self).__init__()
        conv_block = BasicConv2d

        output_branch3x3_1 = 96 if int((out_channels-pool_features)/2) > 96 else int((out_channels-pool_features)/2)
        output_branch3x3 = (out_channels-pool_features) - output_branch3x3_1

        self.branch3x3 = conv_block(in_channels=in_channels, out_channels=output_branch3x3, kernel_size=3, stride=[1, 2], padding=[1, 0])

        self.branch3x3_1 = conv_block(in_channels=in_channels, out_channels=int(output_branch3x3_1*2/3), kernel_size=1)
        self.branch3x3_2 = conv_block(in_channels=int(output_branch3x3_1*2/3), out_channels=output_branch3x3_1,
                                      kernel_size=3, padding=1)
        self.branch3x3_3 = conv_block(in_channels=output_branch3x3_1, out_channels=output_branch3x3_1,
                                      kernel_size=3, stride=[1, 2], padding=[1, 0])

        self.branch_pool = conv_block(in_channels=in_channels, out_channels=pool_features, kernel_size=1)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3_1 = self.branch3x3_1(x)
        branch3x3_1 = self.branch3x3_2(branch3x3_1)
        branch3x3_1 = self.branch3x3_3(branch3x3_1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=[1, 2], padding=[1, 0])
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch3x3, branch3x3_1, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels, out_channels, pool_features):
        super(InceptionE, self).__init__()

        conv_block = BasicConv2d

        output_1x1 = int((out_channels - pool_features)/3)

        output_3x3_r = int(((out_channels - pool_features) - output_1x1)/2)
        output_3x3_ra = int(output_3x3_r/2)
        output_3x3_rb = output_3x3_r - output_3x3_ra

        output_3x3_l = (out_channels - pool_features) - output_1x1 - output_3x3_r
        output_3x3_la = int(output_3x3_l/2)
        output_3x3_lb = output_3x3_l - output_3x3_la

        self.branch1x1 = conv_block(in_channels=in_channels, out_channels=output_1x1, kernel_size=1)

        self.branch3x3_r_1 = conv_block(in_channels=in_channels, out_channels=output_3x3_r, kernel_size=1)
        self.branch3x3_r_2a = conv_block(in_channels=output_3x3_r, out_channels=output_3x3_ra,
                                         kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_r_2b = conv_block(in_channels=output_3x3_r, out_channels=output_3x3_rb,
                                         kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3_l_1 = conv_block(in_channels=in_channels, out_channels=int(output_3x3_l*1.16), kernel_size=1)
        self.branch3x3_l_2 = conv_block(in_channels=int(output_3x3_l * 1.16), out_channels=output_3x3_l,
                                        kernel_size=3, padding=1)
        self.branch3x3_l_3a = conv_block(in_channels=output_3x3_l, out_channels=output_3x3_la,
                                        kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_l_3b = conv_block(in_channels=output_3x3_l, out_channels=output_3x3_lb,
                                        kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels=in_channels, out_channels=pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3_r = self.branch3x3_r_1(x)
        branch3x3_r = [
            self.branch3x3_r_2a(branch3x3_r),
            self.branch3x3_r_2b(branch3x3_r),
        ]
        branch3x3_r = torch.cat(branch3x3_r, 1)

        branch3x3_l = self.branch3x3_l_1(x)
        branch3x3_l = self.branch3x3_l_2(branch3x3_l)
        branch3x3_l = [
            self.branch3x3_l_3a(branch3x3_l),
            self.branch3x3_l_3a(branch3x3_l),
        ]
        branch3x3_l = torch.cat(branch3x3_l, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3_r, branch3x3_l, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)
