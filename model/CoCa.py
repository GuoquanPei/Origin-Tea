import torch
import torch.nn as nn

# 定义 Transformer 块
class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim=512, heads=8, ff_mult=4):
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

# 定义 CoCa 模型
class CoCa(nn.Module):
    def __init__(self, img_size=224, dim=512, num_layers=4, num_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=7, stride=2, padding=3)  # 初步特征提取
        self.norm = nn.LayerNorm(dim)  # 归一化
        self.transformer = nn.Sequential(*[ParallelTransformerBlock(dim=dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局池化
        self.fc = nn.Linear(dim, num_classes)  # 分类层

    def forward(self, x):
        x = self.conv(x)  # 形状: [B, 512, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 变成 (B, N, C)
        x = self.transformer(x)  # Transformer 处理
        x = x.mean(dim=1)  # 计算均值池化
        x = self.fc(x)  # 分类
        return x

# 运行测试
if __name__ == '__main__':
    # 配置设备（CPU 或 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 CoCa 模型
    model = CoCa(num_classes=5).to(device)
    print("Model loaded successfully.")

    # 生成随机输入数据（batch_size=2, 3通道图像，大小为224x224）
    input_tensor = torch.randn(2, 3, 224, 224).to(device)

    # 进行模型推理
    with torch.no_grad():
        model.eval()  # 设置为评估模式
        output = model(input_tensor)

    # 输出结果
    print(f"Output shape: {output.shape}")  # 期望输出: torch.Size([2, 5])

    # 统计模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
