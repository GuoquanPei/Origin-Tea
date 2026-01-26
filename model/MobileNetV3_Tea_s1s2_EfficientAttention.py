import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model.coatnet import conv_3x3_bn, MBConv  # 确保这两个函数已定义

# ======================
# 激活函数和 SE 模块
# ======================

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

# ======================
# Block 模块
# ======================

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

# ======================
# EfficientAttention 模块
# ======================

class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape(n, self.key_channels, h * w)
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape(n, self.value_channels, h * w)

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)

        return reprojected_value + input_

# ======================
# MobileNetV3_Tea_s1s2 with EfficientAttention
# ======================

class MobileNetV3_Tea_s1s2_EfficientAttention(nn.Module):
    def __init__(self, num_classes=6, act=nn.Hardswish):
        super(MobileNetV3_Tea_s1s2_EfficientAttention, self).__init__()

        self.s0 = self._make_layer(conv_3x3_bn, 3, 64, 2, (224 // 2, 224 // 2))

        self.bneck = nn.Sequential(
            Block(5, 64, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        # 插入 EfficientAttention
        self.attention = EfficientAttention(in_channels=96, key_channels=128, head_count=4, value_channels=128)

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
        layers = nn.ModuleList()
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
        out = self.bneck(out)
        out = self.attention(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))
        return self.linear4(out)

# ======================
# 测试代码
# ======================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MobileNetV3_Tea_s1s2_EfficientAttention(num_classes=6).to(device)
    print("Model loaded successfully.")

    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
