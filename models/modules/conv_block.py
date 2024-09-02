import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.config = config

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        if "group_normalization" in self.config and self.config["group_normalization"] and in_channels > 1:
            self.group_normalization = nn.GroupNorm(num_groups=groups, num_channels=self.out_channels)
            self.batch_normalization = self.instance_normalization = None
        elif "batch_normalization" in self.config and self.config["batch_normalization"]:
            self.batch_normalization = nn.BatchNorm2d(self.out_channels)
            self.group_normalization = self.instance_normalization = None
        elif "instance_normalization" in self.config and self.config["instance_normalization"]:
            self.instance_normalization = nn.InstanceNorm2d(self.out_channels)
            self.batch_normalization = self.group_normalization = None
        else:
            self.group_normalization = self.batch_normalization = self.instance_normalization = None

        if self.config["activation"] == "relu":
            self.activation = nn.ReLU()
        elif self.config["activation"] == "mish":
            self.activation = nn.Mish()
        elif self.config["activation"] == "tanh":
            self.activation = nn.Tanh()
        elif self.config["activation"] == "gelu":
            self.activation = nn.GELU()
        elif self.config["activation"] == "glu":
            self.activation = nn.GLU(dim=1)
        elif self.config["activation"] == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation: {self.config['activation']}")
        if self.config["pooling"]:
            self.pooling = nn.MaxPool2d(tuple(self.config["pooling_ks"]))
        else:
            self.pooling = None

        if self.config["upsample"]:
            self.upsample = nn.Upsample(scale_factor=tuple(self.config["upsample_ks"]))
        else:
            self.upsample = None

        if self.config["dropout"] > 0:
            self.dropout = nn.Dropout2d(self.config["dropout"])
        else:
            self.dropout = None
    
    def forward(self, x):
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # x: [B, n_filters, T, num_feature]
        if self.group_normalization:
            x = self.group_normalization(x)
        elif self.batch_normalization:
            x = self.batch_normalization(x)
        x = self.activation(x)
        if self.pooling:
            x = self.pooling(x)
        if self.upsample:
            x = self.upsample(x)
        if self.dropout:
            x = self.dropout(x)
        # x: [B, n_filters, T, num_feature]
        return x
        
if __name__ == "__main__":
    pass