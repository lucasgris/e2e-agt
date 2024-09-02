import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_conv_1d import CausalConv1d

import logging
logger = logging.getLogger(__name__)

class LSTMContextNetwork(nn.Module):
    
    def __init__(self, config):
        super(LSTMContextNetwork, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=self.config["in_channels"],
            hidden_size=self.config["dim"],
            num_layers=self.config["n_layers"],
            batch_first=True,
            dropout=self.config["dropout"],
            bidirectional=self.config["bidirectional"],
        )

    def forward(self, x, mask=None, freeze=False):
        # x: B x C x T
        if freeze:
            logger.info("Freezing the context network")
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
        if mask is not None:
            raise NotImplementedError # TODO
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        return x


class CausalContextNetwork(nn.Module):
    def _make_layers(self, in_channels):
        layers = []
        for i in range(self.config["n_layers"]):
            layers.append(
                CausalConv1d(
                    in_channels=in_channels,
                    out_channels=self.config["dim"],
                    kernel_size=self.config["kernel_size"],
                    dilation=self.config["dilation"],
                    bias=True,
                    groups=self.config["groups"],
                )
            )
            if self.config["batch_normalization"]:
                layers.append(nn.BatchNorm1d(self.config["dim"]))
            if self.config["group_normalization"]:
                layers.append(
                    nn.GroupNorm(
                        num_groups=self.config["groups"], num_channels=self.config["dim"]
                    )
                )
            if self.config["instance_normalization"]:
                layers.append(nn.InstanceNorm1d(self.config["dim"]))
            if self.config["activation"] == "relu":
                layers.append(nn.ReLU())
            elif self.config["activation"] == "mish":
                layers.append(nn.Mish())
            elif self.config["activation"] == "gelu":
                layers.append(nn.GELU())
            elif self.config["activation"] == "glu":
                layers.append(nn.GLU(dim=1))
            else:
                _act = self.config["activation"]
                raise ValueError(f"Unsupported activation: {_act}")
            if self.config["dropout"] > 0:
                layers.append(nn.Dropout(self.config["dropout"]))

            in_channels = self.config["dim"]

        return nn.Sequential(*layers)

    def __init__(self, config, use_subnets=False):
        super(CausalContextNetwork, self).__init__()
        self.config = config
        self.num_strings = self.config["num_strings"]
        self.use_subnets = use_subnets
        if self.use_subnets:
            for s in range(self.num_strings):
                setattr(self, f"String_{s}", self._make_layers(self.config["in_channels"]))
        else:
            self.layers = self._make_layers(self.config["in_channels"])

    def forward(self, x, mask=None, freeze=False):
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        if mask is not None:
            raise NotImplementedError # Not used
        if self.use_subnets: 
            # x: B x S x C x T
            context_features = []
            for s in range(self.num_strings):
                context_features.append(getattr(self, f"String_{s}")(x[:, s, :, :]))
            context_features = torch.stack(context_features, dim=1)
        else:
            # x: B x C x T
            context_features = self.layers(x)
        return context_features


class TransformerContextNetwork(nn.Module):  # TODO
    def __init__(self, config):
        super(TransformerContextNetwork, self).__init__()
        self.config = config

        if config["use_pre_conv"]:
            assert self.config["kernel_size"] > 1, "kernel_size must be greater than 1"
            if config["use_causal_conv"]:
                self.conv = CausalConv1d(
                    in_channels=self.config["in_channels"],
                    out_channels=1,
                    kernel_size=self.config["kernel_size"],
                    dilation=self.config["dilation"],
                )
            else:
                self.conv = nn.Conv1d(
                    in_channels=self.config["in_channels"],
                    out_channels=1,
                    kernel_size=self.config["kernel_size"],
                    padding=self.config["padding"],
                )

        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config["dim"],
                nhead=self.config["n_heads"],
                dim_feedforward=self.config["dim"] * 4,
                dropout=self.config["dropout"],
                activation=self.config["activation"],
                batch_first=True,
            ),
            num_layers=self.config["n_layers"],
        )
        
    def _positional_encoding_1d(self, d_model, time):
        pe = torch.zeros(time, d_model)
        position = torch.arange(0, time).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mask=None, freeze=False):
        # x: B x L x T or B x C x L x T
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        else:
            for param in self.parameters():
                param.requires_grad = True
        
        if self.config["use_pre_conv"]:
            x = x.unsqueeze(1).squeeze(1)  # B x C x L x T
            x = self.conv(x).squeeze(1)  # B x C x T
            
        pos_emb = self._positional_encoding_1d(self.config["dim"], x.size(2))
        pos_emb = pos_emb.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x.permute(0, 2, 1) + pos_emb.to(x.device)
        x = self.enc(x)
        # B x T x C -> B x C x T
        return x.permute(0, 2, 1)