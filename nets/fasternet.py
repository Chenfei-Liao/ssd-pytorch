import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from nets.models.fasternet import FasterNet

if __name__ == "__main__":
    net =FasterNet(num_classes=100)
    print(net)



