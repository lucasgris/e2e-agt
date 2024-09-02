import torch.nn as nn

class LinearBlock:
    def __init__(self, cfg, in_channels, dim):
        self.cfg = cfg
        self.in_channels = in_channels
        self.dim = dim

        block = []
        block.append(
            nn.Linear(self.in_channels, self.dim)
        ) 
        if self.cfg["dropout"] > 0:
            block.append(nn.Dropout(self.cfg["dropout"]))
        if self.cfg["activation"] == "relu":
            block.append(nn.ReLU())
        elif self.cfg["activation"] == "mish":
            block.append(nn.Mish())
        elif self.cfg["activation"] == "tanh":
            block.append(nn.Tanh())
        elif self.cfg["activation"] == "gelu":
            block.append(nn.GELU())
        else:
            raise ValueError(f'Unsupported activation: {self.cfg["activation"]}')
        
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        return self.block(x)