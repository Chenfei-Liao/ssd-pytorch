import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5
from ssd import SSD300

if __name__ == "__main__":
    net = shufflenet_v2_x0_5()
    net.avgpool = nn.Identity()
    net.fc = nn.Identity()
    print(net)



