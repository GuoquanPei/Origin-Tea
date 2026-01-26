# 移除 s1 层
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.coatnet import conv_3x3_bn, MBConv

# Assuming conv_3x3_bn and MBConv are defined elsewhere in your code

class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )
        if stride == 2:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

import torch
import torch.nn as nn
import torch.nn.init as init

class MobileNetV3_Tea_s1s2_4MB(nn.Module):
    def __init__(self, num_classes=6, act=nn.Hardswish):
        super(MobileNetV3_Tea_s1s2_4MB, self).__init__()

        # 仅保留 s0，直接连接到 bneck
        self.s0 = self._make_layer(conv_3x3_bn, 3, 64, 2, (224 // 2, 224 // 2))

        self.bneck = nn.Sequential(
            Block(5, 64, 96, 40, act, True, 2),  # bneck 直接连接 s0，输入通道改为 64
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 96, act, True, 1),
            # Block(5, 48, 144, 96, act, True, 1),
            # Block(5, 48, 288, 96, act, True, 1),
            # Block(5, 96, 576, 96, act, True, 1),
            # Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.s0(x)
        out = self.bneck(out)  # 直接连接到 bneck

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)


if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 mobilenetv4_large 模型
    model = net = MobileNetV3_Tea_s1s2_4MB(num_classes=6).to(device)
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

