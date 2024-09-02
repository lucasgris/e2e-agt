"""
MIT License

Copyright (c) 2021 PRIS-CV: Computer Vision Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

https://github.com/PRIS-CV/AdvancedDropout
"""
import torch
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class AdvancedDropout(Module):

    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super(AdvancedDropout, self).__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = Parameter(torch.Tensor([self.init_mu]))
        self.weight_sigma = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        print(input.size())
        if self.training:
            c, n = input.size()
            # parameterized prior
            h = F.linear(input, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            epsilon = mu + sigma * torch.randn([c, n]).cuda()
            mask = torch.sigmoid(epsilon)

            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
        else:
            out = input

        return out
        
