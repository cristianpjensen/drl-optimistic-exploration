import torch.nn as nn


def disable_gradients(network: nn.Module):
    for param in network.parameters():
        param.requires_grad = False
