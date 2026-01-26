'''
利用Conv2former结构作为链接构建一个PFA结构，这个结构要包含norm操作，我们最后替换的是模型中的norm结构
'''

import torch
import torch.nn as nn
from Conv2Former import Dual_ConvMod,Dual_Block
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def concatenate(inputs, axis):
    h, w = 0, 0
    for i in inputs:
        if i.shape[2] > h:
            h = i.shape[2]
        if i.shape[3] > w:
            w = i.shape[3]
    upsample = []
    for i in inputs:
        upsample.append(nn.UpsamplingBilinear2d(size=(h, w))(i))
    return torch.cat(upsample, axis)

use_sync_bn = False
def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

class Conv2FPA_Norm(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.P1 = Dual_Block(channels[0], channels[1], 11)
        self.P2 = Dual_Block(channels[1], channels[2], 9)
        self.P3 = Dual_Block(channels[2], channels[3], 5)
        self.P4 = Dual_Block(channels[3], channels[3], 5)

        #通过1*1先调整通道数
        self.CONV1 = nn.Sequential(nn.Conv2d(sum(channels), channels[-1], 1),
                                     LayerNorm(channels[-1], eps=1e-6, data_format="channels_first"),
                                     nn.ReLU()
                                     )
        self.Downsample = nn.AvgPool2d(kernel_size=3,stride=2)

        self.norm = get_bn(channels[-1])

    def forward(self,C0,C1,C2,C3,C4):
        #C0:(128,56,56)
        #C1:(256,28,28)
        #C2:(512,14,14)
        #C3:(1024,7,7)
        #C4:(1024,7,7)

        C01 = self.P1(C0,C1) #(128,28,28)
        C12 = self.P2(C1,C2) #(256,14,14)
        C23 = self.P3(C2,C3) #(512,7,7)
        C34 = self.P4(C3,C4) #(1024,7,7)

        C234 = concatenate([C23,C34],1)
        _,_,h,w = C12.shape
        C234 = nn.UpsamplingBilinear2d(size=(h,w))(C234)
        C1234 = concatenate([C234,C12],1)
        _, _, h, w = C01.shape
        C1234 = nn.UpsamplingBilinear2d(size=(h,w))(C1234)
        C01234 = concatenate([C1234,C01],1)

        x = self.CONV1(C01234) #先降通道数
        x = self.Downsample(x)#再降空间尺寸
        x = self.norm(x)
        return x