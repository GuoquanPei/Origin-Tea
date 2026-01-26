"""
InceptionNeXt implementation, paper: https://arxiv.org/abs/2303.16900

Some code is borrowed from timm: https://github.com/huggingface/pytorch-image-models
"""

from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple

def concatenate(inputs, axis):
    h, w = 0, 0
    for i in inputs:
        if i.shape[2] > h:
            h = i.shape[2]
        if i.shape[3] > w:
            w = i.shape[3]
    upsample = []
    for i in inputs:
        upsample.append(nn.UpsamplingBilinear2d(size=(h, w))(i))
    return torch.cat(upsample, axis)

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.mean((2, 3)) # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            
    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        #1、下采样
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            #2、经过InceptionNeXt模块
            x = self.blocks(x)
        return x

#channel_atten
class channel_atten(nn.Module):
    def __init__(self, dims, ratio = 4):
        super().__init__()
        self.dims = dims
        self.inner_dims = int(dims/ratio)
        self.linear1 = nn.Linear(dims, self.inner_dims)
        self.linear2 = nn.Linear(self.inner_dims, dims)


    def forward(self,x):
        _h, _w = x.shape[2:]
        CA = torch.sum(x, [2,3])#nn.AvgPool2d(_h * _w)(x)
              #.view(-1, self.dims))
        CA = self.linear1(CA)
        CA = self.linear2(CA).view((-1, self.dims, 1, 1)).repeat([1, 1, _h, _w])
        return CA * x


# spatial_atten
class spatial_atten(nn.Module):
    def __init__(self, dims, ratio=2):
        super().__init__()
        self.dims = dims
        self.inner_dims = int(dims / ratio)
        self.relu = nn.ReLU()
        self.sigmoid  = nn.Sigmoid()
        self.conv30 = nn.Conv2d(self.dims, self.inner_dims, (1, 9), padding=(0, 4))
        self.bn8 = nn.BatchNorm2d(num_features=self.inner_dims, affine=False)
        self.conv31 = nn.Conv2d(self.inner_dims, 1, (9, 1), padding=(4, 0))
        self.bn9 = nn.BatchNorm2d(num_features=1, affine=False)
        self.conv32 = nn.Conv2d(self.dims, self.inner_dims, (9, 1), padding=(4, 0))
        self.bn10 = nn.BatchNorm2d(num_features=self.inner_dims, affine=False)
        self.conv33 = nn.Conv2d(self.inner_dims, 1, (1, 9), padding=(0, 4))
        self.bn11 = nn.BatchNorm2d(num_features=1, affine=False)

    def forward(self, x1, x2):
        attention1 = self.relu(self.bn8(self.conv30(x1)))  # [-1, 32, h, w]
        attention1 = self.relu(self.bn9(self.conv31(attention1)))  # [-1, 1, h, w]
        attention2 = self.relu(self.bn10(self.conv32(x1)))  # [-1, 32, h, w]
        attention2 = self.relu(self.bn11(self.conv33(attention2)))  # [-1, 1, h, w]
        SA = attention1 + attention2
        SA = self.sigmoid(SA)  # [-1, 1, h, w]
        SA = SA.repeat([1, self.dims, 1, 1])
        return SA * x2  # [-1, 64, h, w]

#将FPA的操作全部弄过来，后面就不会那么混乱
class FPA(nn.Module):
    def __init__(self, dims, with_CA=True, with_SA=True):
        super().__init__()

        self.relu = nn.ReLU()

        self.with_CA = with_CA
        self.with_SA = with_SA
        #FPA组件定义
        self.conv14 = nn.Conv2d(dims[0], dims[0], (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=dims[0], affine=False)
        self.conv15 = nn.Conv2d(dims[0], dims[0], (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=dims[0], affine=False)
        # c12
        self.conv16 = nn.Conv2d(dims[0]*2, int(dims[3]/2), (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=int(dims[3]/2), affine=False)
        # cfe3
        self.conv17 = nn.Conv2d(dims[1], int(dims[3]/24), (1, 1), padding=0)
        self.conv18 = nn.Conv2d(dims[1], int(dims[3]/24), (3, 3), dilation=3, padding=3)
        self.conv19 = nn.Conv2d(dims[1], int(dims[3]/24), (3, 3), dilation=5, padding=5)
        self.conv20 = nn.Conv2d(dims[1], int(dims[3]/24), (3, 3), dilation=7, padding=7)
        self.bn4 = nn.BatchNorm2d(num_features=int(dims[3]/6), affine=False)
        # cfe4
        self.conv21 = nn.Conv2d(dims[2], int(dims[3]/24), (1, 1), padding=0)
        self.conv22 = nn.Conv2d(dims[2], int(dims[3]/24), (3, 3), dilation=3, padding=3)
        self.conv23 = nn.Conv2d(dims[2], int(dims[3]/24), (3, 3), dilation=5, padding=5)
        self.conv24 = nn.Conv2d(dims[2], int(dims[3]/24), (3, 3), dilation=7, padding=7)
        self.bn5 = nn.BatchNorm2d(num_features=int(dims[3]/6), affine=False)
        # cfe5
        self.conv25 = nn.Conv2d(dims[3], int(dims[3]/24), (1, 1), padding=0)
        self.conv26 = nn.Conv2d(dims[3], int(dims[3]/24), (3, 3), dilation=3, padding=3)
        self.conv27 = nn.Conv2d(dims[3], int(dims[3]/24), (3, 3), dilation=5, padding=5)
        self.conv28 = nn.Conv2d(dims[3], int(dims[3]/24), (3, 3), dilation=7, padding=7)
        self.bn6 = nn.BatchNorm2d(num_features=int(dims[3]/6), affine=False)
        # channel wise attention
        self.CA = channel_atten(int(dims[3]/2))

        self.conv29 = nn.Conv2d(int(dims[3]/2), int(dims[3]/2), (1, 1), padding=0)
        self.bn7 = nn.BatchNorm2d(num_features=int(dims[3]/2), affine=False)
        # SpatialAttention
        self.SA = spatial_atten(int(dims[3]/2))

        #最后，要下采样三次才能将输出将为尺寸7*7


    def forward(self, C1, C2, C3, C4, C5, h, w):
        #现在应该来调整这里的通道数了

        C1 = self.conv14(C1)
        C1 = self.relu(self.bn1(C1))
        C2 = self.conv15(C2)
        C2 = self.relu(self.bn2(C2))
        C12 = concatenate([C1, C2], 1)  # C12: [-1, 64+128, h, w]
        C12 = self.conv16(C12)
        C12 = self.relu(self.bn3(C12))  # C12: [-1, 64, h, w]
        C3_cfe = self.relu(
            self.bn4(concatenate([self.conv17(C3), self.conv18(C3), self.conv19(C3), self.conv20(C3)], 1)))
        C4_cfe = self.relu(
            self.bn5(concatenate([self.conv21(C4), self.conv22(C4), self.conv23(C4), self.conv24(C4)], 1)))
        C5_cfe = self.relu(
            self.bn6(concatenate([self.conv25(C5), self.conv26(C5), self.conv27(C5), self.conv28(C5)], 1)))
        C345 = concatenate([C3_cfe, C4_cfe, C5_cfe], 1)  # C345: [-1, 32*4*3, h/4, w/4]
        if self.with_CA:
            C345 = self.CA(C345)
        C345 = self.conv29(C345)
        C345 = self.relu(self.bn7(C345))  # C345: [-1, 64, h/4, w/4]
        C345 = nn.UpsamplingBilinear2d(size=(int(h/4), int(w/4)))(C345)  # C345: [-1, 64, h, w]
        if self.with_SA:
            C12 = self.SA(C345, C12)
        return torch.cat([C12, C345], 1)  # [-1, 128, h, w]   C345和C12各占一半通道

#FPA head 我们只能将FPA和head合为一体，要不然读参数的时候会有问题
class FPA_head(nn.Module):
    def __init__(self, num_classes, dims, with_CA=True, with_SA=True):
        super().__init__()
        self.num_features = dims[-1]

        self.FPA = FPA(dims, with_CA, with_SA)

        #放个1*1搁楞搁楞
        self.head_conv = nn.Sequential(
                        nn.Conv2d(self.num_features, self.num_features, 1),
                        nn.BatchNorm2d(self.num_features),
                        nn.ReLU()
        )

        self.head = MlpHead(self.num_features, num_classes, drop=0)

    def forward(self, C1, C2, C3, C4, C5, h, w):
        x = self.FPA(C1, C2, C3, C4, C5, h, w)
        x =self.head_conv(x)
        x = self.head(x)
        return x

class MetaNeXt(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            token_mixers=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            with_CA=True,
            with_SA=True,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.dims =dims
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []

        # feature resolution stages, each consisting of multiple residual blocks
        prev_chs = dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        #定义的时候，要和原网络结构一样，要不然无法载入预训练权重

        self.head = head_fn(self.num_features, num_classes, drop=drop_rate)
        self.head = FPA_head(9, dims)
        self.apply(self._init_weights)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward(self, x):
        h, w = x.shape[2:] #拿到数据的宽和高

        #--------------stem------------------
        x = self.stem(x)
        C1 = x # 2，96，56，56

        #--------------backbone--------------
        x = self.stages[0](x)
        C2 = x # 2，96，56，56
        x = self.stages[1](x)
        C3 = x # 2，192，28，28
        x = self.stages[2](x)
        C4 = x # 2，384，14，14
        x = self.stages[3](x)
        C5 = x # 2，768，7，7

        #因为载入与训练权重的时候，不经过这个函数，所以我们这里可以按照FPA——head的形式来定义运行，载入预训练权重之后将head换为FPA_head即可
        #x = self.head(x)
        #--------------head------------------
        x = self.head(C1, C2, C3, C4, C5, h, w)

        return x


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    inceptionnext_tiny=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
    ),
    inceptionnext_small=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth',
    ),
    inceptionnext_base=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth',
    ),
    inceptionnext_base_384=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
)


@register_model
def inceptionnext_tiny(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), 
                      token_mixers=InceptionDWConv2d,
                      **kwargs
    )
    model.default_cfg = default_cfgs['inceptionnext_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def inceptionnext_small_FPA(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768),
                      token_mixers=InceptionDWConv2d,
                      **kwargs
    )
    model.default_cfg = default_cfgs['inceptionnext_small']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def inceptionnext_base(pretrained=False, **kwargs):
    model = MetaNeXt(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), 
                      token_mixers=InceptionDWConv2d,
                      **kwargs
    )
    model.default_cfg = default_cfgs['inceptionnext_base']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def inceptionnext_base_384(pretrained=False, **kwargs):
    model = MetaNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], 
                      mlp_ratios=[4, 4, 4, 3],
                      token_mixers=InceptionDWConv2d,
                      **kwargs
    )
    model.default_cfg = default_cfgs['inceptionnext_base_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = inceptionnext_small_FPA()
    model.eval()
    print('------------------- training-time model -------------')
    print(model)
    x = torch.randn(2, 3, 224, 224)
    origin_y = model(x)
    print(origin_y.shape)