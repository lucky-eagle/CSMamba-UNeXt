import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
from SeaAttention import Sea_Attention
from Fusion_mamba import VSSBlock
from functools import partial
from typing import Callable
from CSMamba import SS2D, ChannelAttentionModule, SpatialAttentionModule

__all__ = ['CSMamba-UNeXt']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

def conv1x1(in_planes: int, out_planes: int,
            stride: int = 1) -> nn.Conv2d:  # in_planes: 输入通道数 out_planes: 输出通道数 stride: 步长(虽然定义了但固定为1)
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)  # 使用1×1的卷积核 无偏置项(bias=False)


def shift(dim): #实现特征图移位操作的函数：对输入特征图沿指定维度dim进行不同距离的循环移位；将所有移位后的特征图拼接起来；使用narrow裁剪掉边缘部分
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))] #torch.roll 实现循环移位
    x_cat = torch.cat(x_shift, 1) #dim: 移位的维度 self.pad: 移位的最大距离 H, W: 特征图的高度和宽度
    x_cat = torch.narrow(x_cat, 2, self.pad, H) #torch.cat 沿通道维度拼接
    x_cat = torch.narrow(x_cat, 3, self.pad, W) #torch.narrow 裁剪特征图边界
    return x_cat


class BSConvS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 reduction_ratio=4, use_bn=True, activation=nn.ReLU(inplace=True)):
        super(BSConvS, self).__init__()

        mid_channels = max(1, out_channels // reduction_ratio)

        # 蓝图卷积使用深度可分离卷积
        self.blueprint_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                      groups=in_channels, bias=not use_bn),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            activation,
            # 逐点卷积
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=not use_bn),
            nn.BatchNorm2d(mid_channels) if use_bn else nn.Identity(),
            activation
        )

        # 逐点增强
        self.pointwise = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=not use_bn),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            activation if reduction_ratio > 1 else nn.Identity()
        )

    def forward(self, x):
        x = self.blueprint_conv(x)
        x = self.pointwise(x)
        return x

class CoordAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w

class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, ratio=2, activation=nn.ReLU):
        super(GhostConv, self).__init__()
        init_channels = int(out_channels / ratio)
        new_channels = out_channels - init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(init_channels),
            activation(inplace=True)
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, 3, 1, 1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)

# --- ShiftMLP with CoordAttention GhostConv and LayerNorm ---
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., reduction=32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)

        self.norm1 = nn.LayerNorm(hidden_features)
        self.norm2 = nn.LayerNorm(hidden_features)

        self.coord_att = CoordAttention(hidden_features, hidden_features, reduction=reduction)
        self.ghost_conv = GhostConv(hidden_features, hidden_features, kernel_size=3, padding=1)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x)                         # (B, N, C)
        x = self.norm1(x)

        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, C, H, W)

        x = self.coord_att(x)                   # 替代 Shift
        x = self.ghost_conv(x)                  # 替代 DWConv

        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        x = self.norm2(x)

        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop2(x)

        return x
class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, reduction=32):
        super().__init__()

        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 使用替换后的 shiftmlp（CoordAttention + GhostConv）
        self.mlp = shiftmlp(in_features=dim,
                            hidden_features=mlp_hidden_dim,
                            act_layer=act_layer,
                            drop=drop,
                            reduction=reduction)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # 残差连接 + DropPath
        x = x + self.drop_path(self.mlp(self.norm(x), H, W))
        return x


class DepthwiseSeparableConv(nn.Module):   #一个深度卷积类
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  #卷积核大小为 3x3 步幅为 1，填充为 1；使用 groups=dim 进行深度卷积，每个输入通道独立卷积

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)   #将输入张量从形状 (B, N, C) 转置为 (B, C, N)，然后重塑为 (B, C, H, W)，以适应卷积操作。
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)   #将输出张量展平，使其形状变为 (B, C, H * W)，然后转置为 (B, H * W, C)，以便后续操作。

        return x

class OverlapPatchEmbed(nn.Module):   #主要用于将图像转换为补丁嵌入的类
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768): #patch_size: 每个补丁的大小；stride: 补丁的步幅；embed_dim: 嵌入维度
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)  #尺寸转换：确保 img_size 和 patch_size 转换为二元组格式

        self.img_size = img_size
        self.patch_size = patch_size  #self.img_size 和 self.patch_size 存储图像和补丁的大小
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]   #计算生成的补丁的高度 H 和宽度 W。
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, #初始化一个卷积层 self.proj，用于从输入图像生成嵌入
                              padding=(patch_size[0] // 2, patch_size[1] // 2))  #使用指定的 patch_size 作为卷积核大小，步幅为 stride，并使用适当的填充以保持空间维度
        self.norm = nn.LayerNorm(embed_dim)  #初始化一个层归一化层 self.norm，用于标准化嵌入的输出。

        self.apply(self._init_weights)  #调用 _init_weights 方法，对模型中的所有层进行初始化
                                        #计算示例：原始尺寸: 224x224；块大小: 7x7；步长: 4；输出特征图尺寸: (224-7+2*(7//2))//4 + 1 = 56x56；嵌入序列长度: 56*56 = 3136

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



# EOF
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_rate)
        )
        self.att = SEBlock(out_ch)

    def forward(self, x):
        x = self.conv(x)
        return self.att(x)


class UpsampleLight(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# 额外增强模块：GhostConv、CoordConv、CBAM（简化实现）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_att(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x

class ShallowEnhance(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ghost = GhostConv(in_ch, out_ch, kernel_size=3, padding=1)
        self.coord_conv = nn.Conv2d(out_ch + 2, out_ch, kernel_size=3, padding=1)
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.ghost(x)
        # 添加 CoordConv 坐标通道
        xx = torch.linspace(-1, 1, W, device=x.device).repeat(H, 1).unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        yy = torch.linspace(-1, 1, H, device=x.device).repeat(W, 1).t().unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        x = torch.cat([x, xx, yy], dim=1)
        x = self.coord_conv(x)
        x = self.cbam(x)
        return x

class CSMambaBlock(nn.Module):
    """
    CSMambaBlock: 使用 SS2D + 通道和空间注意力替代原 VSSBlock
    """
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
    ):
        super().__init__()
        # 通道压缩维度
        factor = 2.0
        d_model = int(hidden_dim // factor)

        # down/up 用于通道映射
        self.down = nn.Linear(hidden_dim, d_model)
        self.up = nn.Linear(d_model, hidden_dim)
        # 归一化
        self.ln1 = norm_layer(d_model)
        # 核心 SSM + 注意力
        self.self_attention = SS2D(
            d_model=d_model,
            d_state=d_state,
            dropout=attn_drop_rate,
        )
        # 随机丢弃
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor):
        # x: (B, H, W, C)
        # 1) 通道降维
        x0 = self.down(x)
        # 2) SSM + DropPath
        y = self.drop_path(self.self_attention(self.ln1(x0)))
        # 3) 通道升维 + 残差
        return self.up(x0 + y) + x
#完整CSMamba-UNeXt
class CSMamba-UNeXt(nn.Module):
    def __init__(self, num_classes, img_size=224, embed_dims=[128, 160, 256], drop_path_rate=0.1, drop_rate=0.3):
        super().__init__()

        self.attention1 = Sea_Attention(dim=32, key_dim=8, num_heads=2)
        self.attention2 = Sea_Attention(dim=128, key_dim=16, num_heads=4)

        self.encoder1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Dropout2d(p=drop_rate))
        self.encoder2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout2d(p=drop_rate))
        self.encoder3 = nn.Sequential(nn.Conv2d(32, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(p=drop_rate))

        self.enhance_t1 = ShallowEnhance(16, 16)
        self.enhance_t2 = ShallowEnhance(32, 32)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.block1 = shiftedBlock(embed_dims[1], drop=drop_rate, drop_path=drop_path_rate)
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block2 = shiftedBlock(embed_dims[2], drop=drop_rate, drop_path=drop_path_rate)

        self.dblock1 = shiftedBlock(embed_dims[1], drop=drop_rate, drop_path=drop_path_rate)
        self.dblock2 = shiftedBlock(embed_dims[0], drop=drop_rate, drop_path=drop_path_rate)

        self.up1_trans = UpsampleLight(embed_dims[2])
        self.up1_conv = DWConvBlock(embed_dims[2] + embed_dims[1], embed_dims[1], drop_rate)
        self.fusion1 = CSMambaBlock(embed_dims[1], drop_path=drop_path_rate)

        self.up2_trans = UpsampleLight(embed_dims[1])
        self.up2_conv = DWConvBlock(embed_dims[1] + 128, embed_dims[0], drop_rate)
        self.fusion2 = CSMambaBlock(embed_dims[0], drop_path=drop_path_rate)

        self.up3_trans = UpsampleLight(embed_dims[0])
        self.up3_conv = DWConvBlock(embed_dims[0] + 32, 32, drop_rate)

        self.up4_trans = UpsampleLight(32)
        self.up4_conv = DWConvBlock(32 + 16, 16, drop_rate)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.aux_head = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)

    def forward(self, x):
        B = x.size(0)

        t1 = self.enhance_t1(F.max_pool2d(self.encoder1(x), 2))
        t2 = self.enhance_t2(self.attention1(F.max_pool2d(self.encoder2(t1), 2)))
        t3 = self.attention2(F.max_pool2d(self.encoder3(t2), 2))

        out, H, W = self.patch_embed3(t3)
        out = self.block1(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t4 = out

        out, H, W = self.patch_embed4(out)
        out = self.block2(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        out = self.up1_trans(out)
        out = torch.cat([out, t4], dim=1)
        out = self.up1_conv(out)
        out = self.fusion1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        out = out.flatten(2).transpose(1, 2)
        out = self.dblock1(out, H * 2, W * 2)
        out = out.reshape(B, H * 2, W * 2, -1).permute(0, 3, 1, 2)

        out = self.up2_trans(out)
        out = torch.cat([out, t3], dim=1)
        out = self.up2_conv(out)
        aux_out = self.aux_head(out)
        out = self.fusion2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        out = out.flatten(2).transpose(1, 2)
        out = self.dblock2(out, H * 4, W * 4)
        out = out.reshape(B, H * 4, W * 4, -1).permute(0, 3, 1, 2)

        out = self.up3_trans(out)
        out = torch.cat([out, t2], dim=1)
        out = self.up3_conv(out)

        out = self.up4_trans(out)
        out = torch.cat([out, t1], dim=1)
        out = self.up4_conv(out)

        out = self.final(out)
        out = F.interpolate(out, size=(512, 512), mode='bilinear', align_corners=False)
        aux_out = F.interpolate(aux_out, size=(512, 512), mode='bilinear', align_corners=False)
        return out, aux_out


