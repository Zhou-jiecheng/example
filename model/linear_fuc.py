#用来拓展高性能的线性层实现
import torch
import torch.nn as nn

def get_linear_cls():
    from torch.nn import Linear
    return Linear