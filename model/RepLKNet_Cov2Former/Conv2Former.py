import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

'''
code for conv2Former
这里要把模型改成双输入的，用语融合不同层之间的特征
'''

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    # 暂时没办法将弄到这个包，先直接使用最原始的方法，后面性能太弱的话再来修改这个
    # if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
    #     sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
    #     #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
    #     #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
    #     # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
    #     # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
    #     from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    #     return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    # else:
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True

def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

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

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.pos = nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio), 3, padding=1, groups=int(dim * mlp_ratio))
        self.fc2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    def __init__(self, dim, K = 11):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, K, padding=int(K/2), groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x

#方向要逆着才有意义
#现在是通道向着浅层，尺寸是向着深层，
class Dual_ConvMod(nn.Module):
    def __init__(self, dim, dim2, K = 11):
        '''

        :param dim: 浅层特征的维度
        :param dim2: 深层特征的维度
        :param K: 大卷积核的尺寸
        '''
        super().__init__()

        self.dim = dim
        self.dim2 = dim2
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        #用于深层特征进行通道转换
        self.linear1 = nn.Sequential(nn.Conv2d(dim2, dim, 1),
                                     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                                     nn.ReLU()
                                     )

        self.DownSample = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=2 ,padding=1),
                                     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                                     nn.ReLU()
                                     )


        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, K, padding=int(K/2), groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        '''
        :param x: 浅层特征
        :param y: 深层特征
        :return: 融合特征
        '''
        B, C, H, W = x.shape  # 获取下层特征的
        y = self.linear1(y)  #将深层特征的通道数进行缩减
        # y = nn.UpsamplingBilinear2d(size=(H, W))(y) #对深层特征进行上采样，要不然两者没办法进行相乘
        if self.dim!=self.dim2:
            x = self.DownSample(x)
        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(y)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, K, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim, K)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()



    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


#双流Conv2Former改好了，明天用这个做个金字塔就行了
class Dual_Block(nn.Module):
    def __init__(self, dim, dim2, K, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = Dual_ConvMod(dim, dim2, K)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.identify = nn.Sequential(nn.Conv2d(dim2, dim, 1),
                                     LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                                     nn.ReLU()
                                     )
        if dim == dim2:
            self.identify = nn.Identity()
    def forward(self, x, y):
        x = self.identify(y) + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x, y))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

#C0：(128,56,56)
#C1：(256,28,28)
#C1：(256,28,28)


# class Dual_FPA(nn.Module):
#     def __init__(self, dims):



if __name__ == '__main__':
    #model = Block(3, 11)
    #model = Dual_ConvMod(3,6)
    model = Dual_Block(3,6, K=11)
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    x = torch.randn(64, 3, 224, 224)
    y = torch.randn(64, 6, 112, 112)
    origin_y = model(x,y)
    print(origin_y.shape)