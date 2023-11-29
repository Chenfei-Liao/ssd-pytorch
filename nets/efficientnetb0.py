import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from torchvision.models.efficientnet import efficientnet_b0

if __name__ == "__main__":
    net = efficientnet_b0()
    print(net)
    print(net.features)
    print(net.features[5][2].out_channels)
    for i, layer in enumerate(net.features):
        print(i, layer)


