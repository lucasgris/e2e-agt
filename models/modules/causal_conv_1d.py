import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, groups=1, **kwargs
    ):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            **kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, : -self.padding]