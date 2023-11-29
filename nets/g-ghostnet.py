import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from functools import partial
from torch import nn, Tensor
from nets.models.gghostnet import g_ghost_regnetx_080

if __name__ == "__main__":
    net = g_ghost_regnetx_080()
    print(net)



