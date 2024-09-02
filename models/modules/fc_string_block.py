import torch
import torch.nn as nn
import torch.nn.functional as F


class FCStringBlock(nn.Module):
    def __init__(self, 
                 config, 
                 input_dim, 
                 out_channels=None, 
                 num_strings=6):
        super(FCStringBlock, self).__init__()
        self.config = config
        self.type = config["type"]
        self.bias = self.config["bias"] 
        if out_channels is None:
            self.output_dim = self.config["output_per_string"]
        else:
            self.output_dim = out_channels
        self.input_dim = input_dim
        self.num_strings = num_strings
        self.activation = self.config["activation"]
        self.target_len_frames = self.config["target_len_frames"]
        self.adaptative_pool_to_target_len = self.config["adaptative_pool_to_target_len"]
        self.interpolate_target = self.config["interpolate_target"]

        strings_layers = []
        for i in range(self.num_strings):   
            string_layer = []
            if self.type == "linear":
                string_layer.append(
                    nn.Linear(self.input_dim, self.output_dim, bias=self.bias)
                )
            elif self.type == "conv1d":
                string_layer.append(
                    nn.Conv1d(self.input_dim, self.output_dim, 1, bias=self.bias)
                )
            elif self.type == "conv2d":
                string_layer.append(
                    nn.Conv2d(self.input_dim, self.output_dim, 1, bias=self.bias)
                )

            strings_layers.append(string_layer)

        for i in range(self.num_strings):
            self.__setattr__(f"String_{i+1}", nn.Sequential(*strings_layers[i]))
    
    def forward(self, x_strings, return_logits=False):
        # [B, n_strings, T, C]
        x_strings_out = []
        for i in range(self.num_strings):
            if self.type == "conv2d":
                # [B, T, C] -> [B, C, 1, T] -> [B, C, T]
                x_string = self.__getattr__(f"String_{i+1}")(x_strings[:, i].transpose(1, 2).unsqueeze(2)).squeeze(2).transpose(1, 2)
            elif self.type == "conv1d":
                x_string = self.__getattr__(f"String_{i+1}")(x_strings[:, i].transpose(1, 2)).transpose(1, 2)
            else:
                x_string = self.__getattr__(f"String_{i+1}")(x_strings[:, i])
                if self.adaptative_pool_to_target_len and x_string.shape[1] > self.target_len_frames:
                    x_string = F.adaptive_avg_pool2d(x_string, (self.target_len_frames, self.output_dim))
                elif self.interpolate_target and x_string.shape[1] < self.target_len_frames:
                    x_string = F.interpolate(x_string.unsqueeze(0), size=(self.target_len_frames, self.output_dim)).squeeze(0)
            
            if not return_logits:
                if self.activation == "softmax":
                    x_string = F.softmax(x_string, dim=-1)
                elif self.activation == "log_softmax":
                    x_string = F.log_softmax(x_string, dim=-1)
                elif self.activation == "sigmoid":
                    x_string = F.sigmoid(x_string)

            x_string = x_string.unsqueeze(1)
            x_strings_out.append(x_string)
        x_strings_tensor = torch.cat(x_strings_out, dim=1)
        return x_strings_tensor
        
if __name__ == "__main__":
    from torchsummary import summary

    config = {
        "bias": True,
        "activation": False,
        "output_per_string": 22+2,
    }
    model = FCStringBlock(config, 64)
    print(model)
    summary(model, (6, 64))