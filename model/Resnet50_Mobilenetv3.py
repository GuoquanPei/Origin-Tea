import torch
import torch.nn as nn
import torchvision.models as models


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ResNet50_MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50_MobileNetV3, self).__init__()
        resnet = models.resnet50(pretrained=False)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-2])  # 移除原本的全连接层
        self.se_module = SqueezeExcitation(2048)  # ResNet-50 的最终通道数
        self.dw_conv = DepthwiseSeparableConv(2048, 1024)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)  # ResNet-50 特征提取
        x = self.se_module(x)  # 加入 Squeeze-and-Excitation 模块
        x = self.dw_conv(x)  # 加入 Depthwise Separable Convolution
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 测试网络
if __name__ == "__main__":
    model = ResNet50_MobileNetV3(num_classes=1000)
    print(model)
    test_input = torch.randn(2, 3, 224, 224)
    output = model(test_input)
    print(output.shape)