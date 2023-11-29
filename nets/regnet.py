import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from torchvision.models.regnet import regnet_x_1_6gf

if __name__ == "__main__":
    net = regnet_x_1_6gf()
    print(net)



