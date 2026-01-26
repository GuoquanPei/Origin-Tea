# -*- coding: utf-8 -*-
# @Author  : NaiChuan
# @Time    : 2024/12/7 14:31
# @File    : MNv4_CGBlock.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import constant_init, kaiming_init,trunc_normal_init,normal_init
from timm.layers import DropPath


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)


    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(3,0))
        x1 = self._act_dn(x1)
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(3,0))
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,3))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3))
        x=x1+x3
        return x
# 定义CFBlock (Channel Fusion Block)
class CFBlock(nn.Module):
    """
    The CFBlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CFBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.):
        super(CFBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)

    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x))
        return x


# 定义MobileNetV4_small结构
class MNv4_CFBlock(nn.Module):
    def __init__(self, num_classes=1000):
        super(MNv4_CFBlock, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # 修正CFBlock初始化
        self.cfblock1 = CFBlock(32, 32)  # 正确的初始化方式，传递 in_channels 和 out_channels

        self.conv_dw1 = self._dwconv_block(32, 64, stride=2)
        self.cfblock2 = CFBlock(64, 64)  # 修正CFBlock初始化

        self.conv_dw2 = self._dwconv_block(64, 128, stride=2)
        self.cfblock3 = CFBlock(128, 128)  # 修正CFBlock初始化

        self.conv_dw3 = self._dwconv_block(128, 256, stride=2)
        self.cfblock4 = CFBlock(256, 256)  # 修正CFBlock初始化

        self.conv_dw4 = self._dwconv_block(256, 512, stride=2)
        self.cfblock5 = CFBlock(512, 512)  # 修正CFBlock初始化

        # 全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类头部
        self.fc = nn.Linear(512, num_classes)

    def _dwconv_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.cfblock1(x)  # CFBlock1

        x = self.conv_dw1(x)
        x = self.cfblock2(x)  # CFBlock2

        x = self.conv_dw2(x)
        x = self.cfblock3(x)  # CFBlock3

        x = self.conv_dw3(x)
        x = self.cfblock4(x)  # CFBlock4

        x = self.conv_dw4(x)
        x = self.cfblock5(x)  # CFBlock5

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 测试代码
if __name__ == '__main__':

    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 mobilenetv4_large 模型
    model = MNv4_CFBlock(num_classes=6).to(device)
    # print(model)
    print("Model loaded successfully.")
    # 模拟输入数据（batch_size=1, 3通道图像，大小为224x224）
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    # 模型推理
    with torch.no_grad():  # 关闭梯度计算
        model.eval()  # 设置模型为评估模式
        output = model(input_tensor)
    # 打印输出结果形状
    print(f"Output shape: {output.shape}")

    # Check the trainable params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
