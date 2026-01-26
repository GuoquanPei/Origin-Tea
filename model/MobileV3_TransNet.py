import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


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


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hid_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hid_dim),
            nn.ReLU(),
            nn.Linear(ff_hid_dim, embed_size)
        )
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.layernorm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        att_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(att_output))
        ff_output = self.ff(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x


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


class MobileNetV3_Small_Transformer(nn.Module):
    def __init__(self, num_classes=6, act=nn.Hardswish, embed_size=576, num_heads=8, ff_hid_dim=1024, dropout=0.1):
        super(MobileNetV3_Small_Transformer, self).__init__()
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

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Transformer block (after bottleneck)
        self.transformer = TransformerBlock(embed_size, num_heads, ff_hid_dim, dropout)

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

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten(1)

        # Apply transformer mechanism
        out = out.unsqueeze(0)  # Add a batch dimension for transformer
        out = self.transformer(out)

        out = self.drop(self.hs3(self.bn3(self.linear3(out.squeeze(0)))))

        return self.linear4(out)


if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载 mobilenetv3_small_with_transformer 模型
    model = MobileNetV3_Small_Transformer(num_classes=5).to(device)
    print("Model loaded successfully.")
    # 模拟输入数据（batch_size=2, 3通道图像，大小为224x224）
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
