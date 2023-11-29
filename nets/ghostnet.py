import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from nets.models.ghostnet import ghostnet

if __name__ == "__main__":
    net = ghostnet()
    print(net)



