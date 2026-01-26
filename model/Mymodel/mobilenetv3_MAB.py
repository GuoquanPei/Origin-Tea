import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 创建权重参数，初始化为1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 创建偏置参数，初始化为0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # 判断data_format是否符合预期
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 设置标准化形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 如果数据格式是"channels_last"，直接使用layer_norm
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果是"channels_first"，手动计算均值和方差
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Gated Spatial Attention Unit (GSAU)，用于空间注意力机制
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        # 第一个1x1卷积层，用于特征扩展
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        # 深度可分离卷积层，用于提取空间信息
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        # 第二个1x1卷积层
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        # 标准化层
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的尺度参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        # 保留输入的副本，作为残差连接
        shortcut = x.clone()

        # 经过标准化和第一个卷积层
        x = self.Conv1(self.norm(x))
        # 将通道分为两部分，a用于门控操作，x继续处理
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)  # 空间注意力机制
        x = self.Conv2(x)  # 再次卷积

        # 加上残差连接并返回
        return x * self.scale + shortcut

# Multi-scale Large Kernel Attention (MLKA)，多尺度大核卷积注意力机制
class MLKA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats必须能被3整除，才能用于MLKA")

        i_feats = 2 * n_feats

        # 标准化层
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        # 可学习的尺度参数
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 三个不同尺度的卷积核分别构建7x7、5x5和3x3的LKA模块
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        # 三个不同尺度的额外卷积层
        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        # 第一个1x1卷积层用于通道扩展
        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        # 最后的1x1卷积层用于恢复通道数量
        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        # 保存输入作为残差连接
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # 将特征分为a和x两部分
        a, x = torch.chunk(x, 2, dim=1)
        # a的三部分通过不同尺度的卷积操作
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        # 合并后的结果再经过最后的卷积层
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

# Multi-scale Attention Blocks (MAB)，多尺度注意力块

# ------------------------MAB------------------------


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
    '''expand + depthwise + pointwise'''

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

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
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


class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        # 大核卷积注意力模块
        self.LKA = MLKA(n_feats)
        # 门控空间注意力单元
        self.LFE = GSAU(n_feats)

    def forward(self, x):
        # 先通过大核卷积注意力
        x = self.LKA(x)
        # 然后通过门控空间注意力
        x = self.LFE(x)
        return x





class MobileNetV3_MAB(nn.Module):
    def __init__(self, num_classes=6, act=nn.Hardswish):
        super(MobileNetV3_MAB, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )


        self.mab = MAB(96)

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
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.mab(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)


if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载修改后的 MobileNetV3_Small 模型
    model = MobileNetV3_MAB(num_classes=5).to(device)
    print("Model loaded successfully.")
    # 模拟输入数据
    input_tensor = torch.randn(2, 3, 224, 224).to(device)
    # 模型推理
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # 检查可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
