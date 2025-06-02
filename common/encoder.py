import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np
import functools

class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, inp_channel, num_features: int, num_blocks: int, max_pooling: bool = True):
        super(ResnetStack, self).__init__()
        self.inp_channel = inp_channel
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling

        self.conv1 = nn.Conv2d(
            in_channels=inp_channel,  # Assuming RGB input
            out_channels=self.num_features,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1)
            ) for _ in range(self.num_blocks)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        conv_out = self.conv1(x)

        if self.max_pooling:
            conv_out = F.max_pool2d(conv_out, kernel_size=3, stride=2, padding=1)

        for block in self.blocks:
            block_input = conv_out
            conv_out = block(conv_out)
            conv_out += block_input

        return conv_out


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, 
                 hdim: int=512, num_layer:int=2, 
                 activate_final: bool = False,
                 activ: str='ELU', 
                 layer_norm: bool = False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        dims = [inp_dim]+ [hdim for i in range(num_layer)] + [out_dim]
        for i in range(num_layer+1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layer or activate_final:
                self.layers.append(getattr(nn, activ)())
                if layer_norm:
                    self.layers.append(nn.LayerNorm(dims[i+1]))
        self.apply(self._init_weights)
    # weight initiaization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(self, inp_channel:int = 3, width: int = 1, stack_sizes: tuple = (16, 32, 32), num_blocks: int = 2,
                 dropout_rate: float = None, hdim: int = 512, layer_norm: bool = False
                 ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.hdim = hdim
        self.layer_norm = layer_norm
        self.inp_channels = [inp_channel] + list(stack_sizes)
        self.stack_blocks = nn.ModuleList([
            ResnetStack(
                inp_channel= self.inp_channels[i],
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ])

        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.layer_norm:
            self.ln = nn.LayerNorm([32, 8, 8]) # 84X84 --> 32X11X11

        # Assuming the input shape is known, you'd calculate the flattened size here
        # For this example, let's assume it flattens to 1024
        
        self.fc = MLP(inp_dim=2048, out_dim=hdim, 
                    hdim=hdim, num_layer=0, 
                    activate_final= False,
                    activ='ELU', 
                    layer_norm = layer_norm)

    def forward(self, x, train=True):
        x = x / 255.0 - 0.5
        batch_size = x.shape[0]
        conv_out = x

        for idx, block in enumerate(self.stack_blocks):
            conv_out = block(conv_out)
            if self.dropout_rate is not None and train:
                conv_out = self.dropout(conv_out)
            # print(conv_out.shape)

        conv_out = F.relu(conv_out)
        if self.layer_norm:
            conv_out = self.ln(conv_out)

        out = conv_out.reshape(batch_size, -1)
        # out = self.mlp2(self.relu(self.mlp1(out)))
        out = self.fc(out)
        return out
    
encoder_modules = {
            'impala': ImpalaEncoder,
            'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
            'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
            'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
            }