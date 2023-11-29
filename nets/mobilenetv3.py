import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models.mobilenetv3 import mobilenet_v3_small

if __name__ == "__main__":
    net = mobilenet_v3_large()
    print(net.features[12].out_channels)
    for i, layer in enumerate(net.features):
        print(i, layer)
    print(net.features[10].out_channels)

