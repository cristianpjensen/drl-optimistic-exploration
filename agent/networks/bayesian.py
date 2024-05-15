import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(BayesianConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        self.weight_mean = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_std = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_mean = nn.Parameter(torch.empty(out_channels))
        self.bias_std = nn.Parameter(torch.empty(out_channels))

        # Initialize the parameters
        self.weight_mean.data.normal_(0, 0.1)
        self.weight_std.data.normal_(-3, 0.1)
        self.bias_mean.data.normal_(0, 0.1)
        self.bias_std.data.normal_(-3, 0.1)

    def forward(self, x, train=True):
        if train:
            w_std = torch.log1p(torch.exp(self.weight_std))
            b_std = torch.log1p(torch.exp(self.bias_std))

            weight = self.weight_mean + w_std * torch.randn_like(self.weight_mean)
            bias = self.bias_mean + b_std * torch.randn_like(self.bias_mean)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.conv2d(x, weight, bias, stride=self.stride)


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(BayesianLinear, self).__init__()

        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_std = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mean = nn.Parameter(torch.empty(out_features))
        self.bias_std = nn.Parameter(torch.empty(out_features))

        # Initialize the parameters
        self.weight_mean.data.normal_(0, 0.1)
        self.weight_std.data.normal_(-3, 0.1)
        self.bias_mean.data.normal_(0, 0.1)
        self.bias_std.data.normal_(-3, 0.1)

    def forward(self, x, train=True):
        if train:
            w_std = torch.log1p(torch.exp(self.weight_std))
            b_std = torch.log1p(torch.exp(self.bias_std))

            weight = self.weight_mean + w_std * torch.randn_like(self.weight_mean)
            bias = self.bias_mean + b_std * torch.randn_like(self.bias_mean)
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return F.linear(x, weight, bias)
