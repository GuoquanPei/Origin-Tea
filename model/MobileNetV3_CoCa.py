import torch
import torch.nn as nn
import torchvision.models as models


# 定义 CoCa 风格的 Transformer 块
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=4, ff_mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim)
        )

    def forward(self, x):
        x = self.norm(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x + self.ffn(x)
        return x


# 定义 MobileNetV3-Small + CoCa 结构
class MobileNetV3_CoCa(nn.Module):
    def __init__(self, num_classes=5, dim=256, num_layers=2):
        super().__init__()
        # 加载 MobileNetV3-Small 主干网络（去掉分类头）
        self.mobilenet = models.mobilenet_v3_small(weights=None)
        self.mobilenet.classifier = nn.Identity()  # 移除原始分类层

        # 1x1 卷积降维，减少 Transformer 计算量
        self.conv1x1 = nn.Conv2d(576, dim, kernel_size=1)  # MobileNetV3-Small 最终特征通道数为 576

        # CoCa 风格 Transformer 结构
        self.transformer = nn.Sequential(*[ParallelTransformerBlock(dim=dim) for _ in range(num_layers)])

        # 全局池化 + 分类头
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.mobilenet.features(x)  # MobileNetV3 提取特征 (B, 576, H, W)
        x = self.conv1x1(x)  # 降维到 (B, 256, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 变成 (B, N, C) 送入 Transformer
        x = self.transformer(x)  # Transformer 处理
        x = x.mean(dim=1)  # 计算均值池化
        x = self.fc(x)  # 分类
        return x


# 运行测试
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MobileNetV3_CoCa(num_classes=5).to(device)
    print("Model loaded successfully.")

    input_tensor = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        model.eval()
        output = model(input_tensor)

    print(f"Output shape: {output.shape}")  # 期望输出: torch.Size([2, 5])
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")