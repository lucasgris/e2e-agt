from .conv_feature_extractor import ConvFeatureExtractor
from .linear_block import LinearBlock
from .fc_string_block import FCStringBlock
from .gumbel_vector_quantizer import GumbelVectorQuantizer

from torch import nn

# Setup linear layers
def setup_linear_block(cfg, in_channels, dim):    
    block = []
    block.append(
        nn.Linear(in_channels, dim)
    ) 
    if cfg["dropout"] > 0:
        block.append(nn.Dropout(cfg["dropout"]))
    if cfg["activation"] == "relu":
        block.append(nn.ReLU())
    elif cfg["activation"] == "mish":
        block.append(nn.Mish())
    elif cfg["activation"] == "tanh":
        block.append(nn.Tanh())
    else:
        raise ValueError(f'Unsupported activation: {cfg["activation"]}')
    
    return nn.Sequential(*block), dim