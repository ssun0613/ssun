import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from options.config import Config

class gain():
    def _make_att_map(self, net, out1):
        layer_index = np.argmax(np.array([name == 'CNN' for name in net._modules.keys()], dtype=np.int))
        feature_maps = net.hookF[layer_index].output

        class_1 = out1[:, 0:8].unsqueeze(dim=2)
        class_0 = out1[:, 8:].unsqueeze(dim=2)

        out1 = torch.concat([class_1, class_0], dim=2)

        score = (out1 * net.label)[:, :, 1]
        score.mean().backward(retain_graph=True)

        gradient = net.hookB[layer_index].output[0]
        weighted = gradient.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (gradient.shape[2] * gradient.shape[3])
        attention = F.relu((feature_maps.detach() * weighted.detach()).sum(dim=1).unsqueeze(dim=1))
        attention = F.interpolate(attention, size=(net.input.shape[2], net.input.shape[3]))

        attention = (attention - attention.min()) / (attention.max() - attention.min())

        return attention

    def _make_masked_img(self, net, attention):
        attention[attention >= 0.4] = 1
        attention[attention < 0.4] = 0
        maskedImg = net.input * (1 - attention) + attention

        return maskedImg
