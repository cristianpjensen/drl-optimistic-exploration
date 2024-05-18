import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(BayesianConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self.weight_mean = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_mean = nn.Parameter(torch.empty(out_channels))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_channels))

        # https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/3555c30b00cc25cf0220658ac394cc1985fb27b4/torchbnn/modules/conv.py#L78
        init_std = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)

        self.weight_prior_mean = 0
        self.weight_prior_sigma = init_std
        self.weight_prior_log_sigma = math.log(self.weight_prior_sigma)
        self.bias_prior_mean = 0
        self.bias_prior_sigma = init_std
        self.bias_prior_log_sigma = math.log(self.bias_prior_sigma)

        self.weight_mean.data.uniform_(-init_std, init_std)
        self.weight_log_sigma.data.fill_(self.weight_prior_log_sigma)
        self.bias_mean.data.uniform_(-init_std, init_std)
        self.bias_log_sigma.data.fill_(self.bias_prior_log_sigma)

    def forward(self, x, train=True):
        if train:
            w_sigma = torch.exp(self.weight_log_sigma / 2)
            b_sigma = torch.exp(self.bias_log_sigma / 2)

            weight = self.weight_mean + w_sigma * torch.randn_like(w_sigma)
            bias = self.bias_mean + b_sigma * torch.randn_like(b_sigma)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.conv2d(x, weight, bias, stride=self.stride)

    def prior_kl_loss(self):
        weight_kls = self.weight_prior_log_sigma - self.weight_log_sigma + ((torch.exp(self.weight_log_sigma).square() + torch.square(self.weight_mean - self.weight_prior_mean)) / (2 * self.weight_prior_sigma ** 2)) - 0.5
        bias_kls = self.bias_prior_log_sigma - self.bias_log_sigma + ((torch.exp(self.bias_log_sigma).square() + torch.square(self.bias_mean - self.bias_prior_mean)) / (2 * self.bias_prior_sigma ** 2)) - 0.5
        return weight_kls.sum() + bias_kls.sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(BayesianLinear, self).__init__()

        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.empty(out_features))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))

        # Initialize the parameters: https://github.com/Harry24k/bayesian-neural-network-pytorch/blob/3555c30b00cc25cf0220658ac394cc1985fb27b4/torchbnn/modules/conv.py#L78
        init_std = 1.0 / math.sqrt(in_features)

        self.weight_prior_mean = 0
        self.weight_prior_sigma = init_std
        self.weight_prior_log_sigma = math.log(self.weight_prior_sigma)
        self.bias_prior_mean = 0
        self.bias_prior_sigma = init_std
        self.bias_prior_log_sigma = math.log(self.bias_prior_sigma)

        self.weight_mean.data.uniform_(-init_std, init_std)
        self.weight_log_sigma.data.fill_(self.weight_prior_log_sigma)
        self.bias_mean.data.uniform_(-init_std, init_std)
        self.bias_log_sigma.data.fill_(self.bias_prior_log_sigma)

    def forward(self, x, train=True):
        if train:
            w_sigma = torch.exp(self.weight_log_sigma)
            b_sigma = torch.exp(self.bias_log_sigma)

            weight = self.weight_mean + w_sigma * torch.randn_like(w_sigma)
            bias = self.bias_mean + b_sigma * torch.randn_like(b_sigma)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.linear(x, weight, bias)

    def prior_kl_loss(self):
        weight_kls = self.weight_prior_log_sigma - self.weight_log_sigma + ((torch.exp(self.weight_log_sigma).square() + torch.square(self.weight_mean - self.weight_prior_mean)) / (2 * self.weight_prior_sigma ** 2)) - 0.5
        bias_kls = self.bias_prior_log_sigma - self.bias_log_sigma + ((torch.exp(self.bias_log_sigma).square() + torch.square(self.bias_mean - self.bias_prior_mean)) / (2 * self.bias_prior_sigma ** 2)) - 0.5
        return weight_kls.sum() + bias_kls.sum()
