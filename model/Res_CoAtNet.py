import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import resnet50  # 使用 ResNet 结构
from model.coatnet import conv_3x3_bn, MBConv  # 你之前的 CoatNet 部分

class MoCoatResNet(nn.Module):
    def __init__(self, num_classes=6):
        super(MoCoatResNet, self).__init__()

        # 保留 CoatNet 前三层
        self.s0 = self._make_layer(conv_3x3_bn, 3, 64, 2, (224 // 2, 224 // 2))
        self.s1 = self._make_layer(MBConv, 64, 96, 2, (224 // 4, 224 // 4))
        self.s2 = self._make_layer(MBConv, 96, 192, 3, (224 // 8, 224 // 8))

        # 插入通道适配层 (192 -> 512)
        self.channel_adapter = nn.Conv2d(192, 512, kernel_size=1, stride=1, bias=False)

        # 替换 MobileNetV3 部分为 ResNet
        resnet = resnet50(weights="IMAGENET1K_V1")  # 使用新版的权重加载方式
        self.resnet_layer3 = resnet.layer3  # ResNet 的第三层
        self.resnet_layer4 = resnet.layer4  # ResNet 的第四层

        # 替换分类层
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)  # 线性分类层

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
        out = self.s0(x)  # CoatNet 的前三层
        out = self.s1(out)
        out = self.s2(out)

        out = self.channel_adapter(out)  # 适配通道大小
        out = self.resnet_layer3(out)  # ResNet 的 layer3
        out = self.resnet_layer4(out)  # ResNet 的 layer4

        out = self.gap(out).flatten(1)  # 全局平均池化
        out = self.fc(out)  # 分类层

        return out

if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 MoCoatResNet 模型
    model = MoCoatResNet(num_classes=5).to(device)
    print("Model loaded successfully.")

    # 模拟输入数据（batch_size=2, 3通道图像，大小为224x224）
    input_tensor = torch.randn(2, 3, 224, 224).to(device)

    # 模型推理
    with torch.no_grad():  # 关闭梯度计算
        model.eval()  # 设置模型为评估模式
        output = model(input_tensor)

    # 打印输出结果形状
    print(f"Output shape: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}")
