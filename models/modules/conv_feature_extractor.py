import ast
import math
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnAudio import features
from .conv_block import ConvBlock

import logging 
logger = logging.getLogger(__name__)

def _apply_spec_augment_2d(spec, max_mask_pct, n_time_masks, n_freq_masks):
    if 0 > max_mask_pct or max_mask_pct > 1:
        raise ValueError("max_mask_pct must be in [0, 1]")
    spec = spec.squeeze(1)
    _, n_freqs, n_frames = spec.shape
    mask_value = spec.mean()
    aug_spec = spec.clone()

    freq_mask_param = max(1, int(max_mask_pct * n_freqs))
    for _ in range(n_freq_masks):
        t = torch.randint(n_freqs - freq_mask_param + 1, (1,)).item()
        aug_spec[:, t:t + freq_mask_param, :] = mask_value

    time_mask_param = max(1, int(max_mask_pct * n_frames))
    for _ in range(n_time_masks):
        t = torch.randint(n_frames - time_mask_param + 1, (1,)).item()
        aug_spec[:, :, t:t + time_mask_param] = mask_value

    return aug_spec.unsqueeze(1)

def _apply_spec_augment_1d(x, max_mask_pct, n_time_masks):
    x = x.squeeze(1)
    _, n_frames = x.shape
    mask_value = x.mean()
    aug_x = x.clone()

    time_mask_param = max(1, int(max_mask_pct * n_frames))
    for _ in range(n_time_masks):
        t = torch.randint(n_frames - time_mask_param + 1, (1,)).item()
        aug_x[t:t + time_mask_param] = mask_value

    return aug_x.unsqueeze(1)

class RawConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config

        if type(self.config["conv_layers"]) == str:
            self.config["conv_layers"] = ast.literal_eval(self.config["conv_layers"])

        if type(self.config["activation"]) == str:
            if self.config["activation"] == "relu":
                self.activation = nn.ReLU()
            elif self.config["activation"] == "gelu":
                self.activation = nn.GELU()
            elif self.config["activation"] == "leaky_relu":
                self.activation = nn.LeakyReLU()
            else:
                raise Exception("unknown activation " + self.config["activation"])


        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(int(n_in), int(n_out), int(k), stride=stride, bias=False),
                nn.Dropout(p=float(self.config["dropout"])),
                self.activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in self.config["conv_layers"]:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.config["log_compression"] = self.config["log_compression"]
        self.config["skip_connections"] = self.config["skip_connections"]
        self.config["residual_scale"] = math.sqrt(self.config["residual_scale"])
        
        if self.config.get("load_weights_from", None) is not None:
            self.load_state_dict(torch.load(self.config["load_weights_from"]))

    def forward(self, x, freeze=False):
        
        if self.config.get("spec_augment", None) is not None and self.training:
            x = _apply_spec_augment_1d(x, 
                                       self.config["spec_augment"]["max_mask_pct"], 
                                       self.config["spec_augment"]["n_time_masks"])
        if freeze:
            self.conv_layers.eval()
            for param in self.conv_layers.parameters():
                param.requires_grad = False
        else:
            self.conv_layers.train()
            for param in self.conv_layers.parameters():
                param.requires_grad = True

        if len(x.size()) == 2:
            # BxT -> BxCxT
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.config["skip_connections"] and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.config["residual_scale"]

        if self.config["log_compression"]:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x, None, None  # z, residual, features


class ConvVHFFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ConvVHFFeatureExtractor, self).__init__()
        self.config = config
        
        if self.config["feature_extractor"] == "nnAudio":
            if self.config["nnAudio"]["feature"] == "CQT":
                self.feature_extractor = features.cqt.CQT(**self.config["nnAudio"]["params"])
            elif self.config["nnAudio"]["feature"] == "STFT":
                self.feature_extractor = features.stft.STFT(**self.config["nnAudio"]["params"])
            elif self.config["nnAudio"]["feature"] == "MelSpectrogram":
                self.feature_extractor = features.mel.MelSpectrogram(**self.config["nnAudio"]["params"])
            else:
                raise ValueError("Unknown feature extractor: {}".format(self.config["nnAudio"]["feature"]))
        else:
            self.feature_extractor = None

        # Setup convolutional blocks
        def setup_conv_sequence(cfg):
            conv_block_kernels=cfg["conv_block_kernels"].copy()
            conv_block_kernels.reverse()
            init_layer = ConvBlock(cfg["conv_block"],   
                                   in_channels=1,  
                                   out_channels=cfg["conv_block_dims"][0],
                                   kernel_size=tuple(conv_block_kernels.pop()))
            conv_blocks = [init_layer]
            inc = cfg["conv_block_dims"][0]
            for dim in cfg["conv_block_dims"][1:]:
                kernel_size=tuple(conv_block_kernels.pop())
                conv_blocks.append(ConvBlock(cfg["conv_block"], 
                                            kernel_size=kernel_size,
                                            in_channels=inc, 
                                            out_channels=dim))
                inc = dim   
            conv_blocks.append(nn.Conv2d(
                in_channels=inc, 
                out_channels=1, 
                kernel_size=3, 
                padding=0,
                stride=1
            ))
            return nn.Sequential(*conv_blocks)             

        self.conv_sequence_horizontal = setup_conv_sequence(self.config["horizontal_filter_conv"])
        self.conv_sequence_vertical = setup_conv_sequence(self.config["vertical_filter_conv"])
    

    def forward(self, x, freeze=False):                
        if self.feature_extractor is not None:
            assert len(x.size()) < 4, (
                "Input size must be (B, 1, T) or (B, T) for nnAudio feature extractor. "
                "Got: {}".format(x.size())
            )
            # x (B, 1, len_audio) -> B x C x T -> B x 1 x C x T
            features = self.feature_extractor(x).unsqueeze(1)
        else:
            features = x

        if self.config["spec_augment"] and self.training:
            features = _apply_spec_augment_2d(features, 
                                           self.config["spec_augment"]["max_mask_pct"], 
                                           self.config["spec_augment"]["n_time_masks"], 
                                           self.config["spec_augment"]["n_freq_masks"])

        if freeze:
            self.conv_blocks_module.eval()
            self.init_layer.eval()
            for param in self.conv_blocks_module.parameters():
                param.requires_grad = False
            for param in self.init_layer.parameters():
                param.requires_grad = False
        else:
            self.conv_blocks_module.train()
            self.init_layer.train()
            for param in self.conv_blocks_module.parameters():
                param.requires_grad = True
            for param in self.init_layer.parameters():
                param.requires_grad = True

        z = self.init_layer(features)
        
        if self.config["conv_skip_connection"]:
            z_h = z_v = z
            for layer in self.conv_sequence_horizontal:
                residual = z_h
                z_h = layer(z_h)
                if z_h.size() == residual.size():
                    z_h = z_h + residual
            for layer in self.conv_sequence_vertical:
                residual = z_v
                z_v = layer(z_v)
                if z_v.size() == residual.size():
                    z_v = z_v + residual
        else:
            residual = z
            z_h = self.conv_sequence_horizontal(z)
            z_v = self.conv_sequence_vertical(z)

        # H W
        assert z_h.size(2) >= z_v.size(2), "z_h.size(2) must be greater than or equal to z_v.size(2)"
        assert z_v.size(3) >= z_h.size(3), "z_v.size(3) must be greater than or equal to z_h.size(3)"
        target_size = (z.size(0), z.size(1), z_h.size(2), z_v.size(3))
        
        z_v = F.interpolate(
            z_v, 
            size=target_size[2:], 
            mode="nearest"
        )
        z_h = F.interpolate(
            z_h, 
            size=target_size[2:], 
            mode="nearest"
        )
        
        z = torch.cat([z_h, z_v], dim=1)
            
        if self.config.get("transpose_channels_height", False):
            # B C H T (B C H W) -> B H C T (B C H W)
            z = z.permute(0, 2, 1, 3)

        return z, residual, features


class ConvFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(ConvFeatureExtractor, self).__init__()
        self.config = config
        if self.config["feature_extractor"] == "nnAudio":
            if self.config["nnAudio"]["feature"] == "CQT":
                self.feature_extractor = features.cqt.CQT(**self.config["nnAudio"]["cqt_params"])
            elif self.config["nnAudio"]["feature"] == "STFT":
                self.feature_extractor = features.stft.STFT(**self.config["nnAudio"]["stft_params"])
            elif self.config["nnAudio"]["feature"] == "MelSpectrogram":
                self.feature_extractor = features.mel.MelSpectrogram(**self.config["nnAudio"]["params"])
            else:
                raise ValueError("Unknown feature extractor: {}".format(self.config["nnAudio"]["feature"]))
        else:
            self.feature_extractor = None

        self.init_layer = ConvBlock(
            self.config["init_layer"],
            padding=self.config["init_layer"]["padding"],
            kernel_size=self.config["init_layer"]["kernel"],
            in_channels=1,
            out_channels=int(self.config["init_layer"]["dim"]),
        )

        # Setup convolutional blocks
        conv_block_kernels = self.config["conv_block_kernels"]
        conv_block_kernels.reverse()
        conv_blocks = []
        inc = self.config["init_layer"]["dim"]
        for dim in self.config["conv_block_dims"]:
            kernel_size = tuple(conv_block_kernels.pop())
            conv_blocks.append(
                ConvBlock(
                    self.config["conv_block"],
                    padding=self.config["conv_block"]["padding"],
                    groups=self.config["conv_block"]["groups"],
                    kernel_size=kernel_size,
                    in_channels=inc,
                    out_channels=dim,
                )
            )
            inc = dim
        self.conv_blocks_module = nn.Sequential(*conv_blocks)

    def forward(self, x, freeze=False):            
        if self.feature_extractor is not None:
            assert len(x.size()) < 4, (
                "Input size must be (B, 1, T) or (B, T) for nnAudio feature extractor. "
                "Got: {}".format(x.size())
            )
            # x (B, 1, len_audio) -> B x C x T -> B x 1 x C x T
            features = self.feature_extractor(x).unsqueeze(1)
        else:
            features = x

        if self.config["spec_augment"] and self.training:
            features = _apply_spec_augment_2d(features, 
                                           self.config["spec_augment"]["max_mask_pct"], 
                                           self.config["spec_augment"]["n_time_masks"], 
                                           self.config["spec_augment"]["n_freq_masks"])

        if freeze:
            self.conv_blocks_module.eval()
            self.init_layer.eval()
            for param in self.conv_blocks_module.parameters():
                param.requires_grad = False
            for param in self.init_layer.parameters():
                param.requires_grad = False
        else:
            self.conv_blocks_module.train()
            self.init_layer.train()
            for param in self.conv_blocks_module.parameters():
                param.requires_grad = True
            for param in self.init_layer.parameters():
                param.requires_grad = True

        z = self.init_layer(features)

        if self.config["conv_skip_connection"]:
            for layer in self.conv_blocks_module:
                residual = z
                z = layer(z)
                if z.size() == residual.size():
                    z = z + residual
        else:
            residual = z
            z = self.conv_blocks_module(z)
            
        if self.config.get("transpose_channels_height", False):
            # B C H T (B C H W) -> B H C T (B C H W)
            z = z.permute(0, 2, 1, 3)
        
        return z, residual, features