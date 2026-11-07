"""Explicit ablation architectures used by the manuscript experiments.

`BaseCSMamba` is the original UNeXt encoder-decoder with exactly two
CSMamba blocks added after the first two decoder skip-add fusions.  It does
not contain SEA Attention, ShallowEnhance, CoordAttention, GhostConv, or the
auxiliary head of the complete model.
"""

import math

import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

from archs import CSMambaBlock, OverlapPatchEmbed, ShallowEnhance, shiftedBlock
from SeaAttention import Sea_Attention


class UNeXtDWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, height: int, width: int):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class UNeXtShiftMLP(nn.Module):
    """The spatial shift MLP from the original UNeXt implementation."""

    def __init__(self, in_features: int, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop: float = 0.0, shift_size: int = 5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = UNeXtDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.shift_size = shift_size
        self.pad = shift_size // 2
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()

    def _shift(self, x, height: int, width: int, spatial_dim: int):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width).contiguous()
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        chunks = torch.chunk(x, self.shift_size, dim=1)
        shifted = [
            torch.roll(chunk, amount, dims=spatial_dim)
            for chunk, amount in zip(chunks, range(-self.pad, self.pad + 1))
        ]
        x = torch.cat(shifted, dim=1)
        x = torch.narrow(x, 2, self.pad, height)
        x = torch.narrow(x, 3, self.pad, width)
        return x.reshape(batch, channels, height * width).transpose(1, 2).contiguous()

    def forward(self, x, height: int, width: int):
        x = self._shift(x, height, width, spatial_dim=2)
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self._shift(x, height, width, spatial_dim=3)
        x = self.fc2(x)
        return self.drop(x)


class UNeXtShiftedBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = UNeXtShiftMLP(dim, hidden_features=dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, height: int, width: int):
        return x + self.drop_path(self.mlp(self.norm(x), height, width))


class BaseCSMamba(nn.Module):
    """Original UNeXt plus CSMamba blocks at two decoder fusion stages."""

    def __init__(self, num_classes: int = 1, img_size: int = 256,
                 embed_dims=None, base_drop_rate: float = 0.0,
                 base_drop_path_rate: float = 0.0,
                 csmamba_drop_path_rate: float = 0.1,
                 attention_dropout: float = 0.0,
                 use_csmamba: bool = True):
        super().__init__()
        embed_dims = embed_dims or [128, 160, 256]

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block1 = UNeXtShiftedBlock(embed_dims[1], base_drop_rate, base_drop_path_rate)
        self.block2 = UNeXtShiftedBlock(embed_dims[2], base_drop_rate, base_drop_path_rate)
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[0], 32, 3, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        if use_csmamba:
            self.fusion1 = CSMambaBlock(
                embed_dims[1], drop_path=csmamba_drop_path_rate,
                attn_drop_rate=attention_dropout,
            )
            self.fusion2 = CSMambaBlock(
                embed_dims[0], drop_path=csmamba_drop_path_rate,
                attn_drop_rate=attention_dropout,
            )
        else:
            self.fusion1 = nn.Identity()
            self.fusion2 = nn.Identity()
        self.dblock1 = UNeXtShiftedBlock(embed_dims[1], base_drop_rate, base_drop_path_rate)
        self.dblock2 = UNeXtShiftedBlock(embed_dims[0], base_drop_rate, base_drop_path_rate)
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        out, height, width = self.patch_embed3(out)
        out = self.block1(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        out = self.block2(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t4)
        out = self.fusion1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock1(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t3)
        out = self.fusion2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock2(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.final(out)


class BaseUNeXt(BaseCSMamba):
    """UNeXt backbone used as Variant 1 in the manuscript ablation study.

    The encoder, decoder, skip-add topology and shifted-MLP blocks are the
    same as :class:`BaseCSMamba`; replacing both decoder CSMamba modules with
    identities removes the only CSMamba-specific operations and parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_csmamba=False, **kwargs)


class BaseShallowSeaCAGhost(nn.Module):
    """Original UNeXt plus ShallowEnhance, SEA, and CA-Ghost blocks.

    The original convolutional encoder/decoder and skip-add topology are
    retained.  CSMamba blocks and the complete model's auxiliary head are
    deliberately absent.
    """

    def __init__(self, num_classes: int = 1, img_size: int = 256,
                 embed_dims=None, module_drop_rate: float = 0.2,
                 module_drop_path_rate: float = 0.1):
        super().__init__()
        embed_dims = embed_dims or [128, 160, 256]

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.enhance_t1 = ShallowEnhance(16, 16)
        self.enhance_t2 = ShallowEnhance(32, 32)
        self.attention1 = Sea_Attention(dim=32, key_dim=8, num_heads=2)
        self.attention2 = Sea_Attention(dim=128, key_dim=16, num_heads=4)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.block2 = shiftedBlock(
            embed_dims[2], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[0], 32, 3, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        self.dblock1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dblock2 = shiftedBlock(
            embed_dims[0], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = self.enhance_t1(out)
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t2 = self.enhance_t2(self.attention1(out))
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t3 = self.attention2(out)

        out, height, width = self.patch_embed3(t3)
        out = self.block1(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        out = self.block2(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t4)
        _, _, height, width = out.shape
        out = self.dblock1(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t3)
        _, _, height, width = out.shape
        out = self.dblock2(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.final(out)


class BaseCSMambaSeaCAGhost(nn.Module):
    """Original UNeXt plus CSMamba, SEA, and CA-Ghost blocks.

    ShallowEnhance and the complete model's auxiliary head are deliberately
    absent.  This makes the variant an explicit test of CSMamba + SEA +
    CA-Ghost under the same encoder-decoder and skip-add topology used by the
    other ablation architectures.
    """

    def __init__(self, num_classes: int = 1, img_size: int = 256,
                 embed_dims=None, module_drop_rate: float = 0.2,
                 module_drop_path_rate: float = 0.1,
                 csmamba_drop_path_rate: float = 0.1,
                 attention_dropout: float = 0.0):
        super().__init__()
        embed_dims = embed_dims or [128, 160, 256]

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.attention1 = Sea_Attention(dim=32, key_dim=8, num_heads=2)
        self.attention2 = Sea_Attention(dim=128, key_dim=16, num_heads=4)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.block2 = shiftedBlock(
            embed_dims[2], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[0], 32, 3, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.fusion1 = CSMambaBlock(
            embed_dims[1], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.fusion2 = CSMambaBlock(
            embed_dims[0], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.dblock1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dblock2 = shiftedBlock(
            embed_dims[0], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t2 = self.attention1(out)
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t3 = self.attention2(out)

        out, height, width = self.patch_embed3(t3)
        out = self.block1(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        out = self.block2(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t4)
        out = self.fusion1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock1(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t3)
        out = self.fusion2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock2(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.final(out)


class BaseCSMambaShallowCAGhost(nn.Module):
    """Original UNeXt plus CSMamba, ShallowEnhance, and CA-Ghost blocks.

    SEA Attention and the complete model's auxiliary head are deliberately
    absent.  This completes the 3-of-4 ablation grid that already contains
    Base+CSMamba, Base+CSMamba+SEA+CA-Ghost, and
    Base+ShallowEnhance+SEA+CA-Ghost.
    """

    def __init__(self, num_classes: int = 1, img_size: int = 256,
                 embed_dims=None, module_drop_rate: float = 0.2,
                 module_drop_path_rate: float = 0.1,
                 csmamba_drop_path_rate: float = 0.1,
                 attention_dropout: float = 0.0):
        super().__init__()
        embed_dims = embed_dims or [128, 160, 256]

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.enhance_t1 = ShallowEnhance(16, 16)
        self.enhance_t2 = ShallowEnhance(32, 32)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.block2 = shiftedBlock(
            embed_dims[2], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[0], 32, 3, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.fusion1 = CSMambaBlock(
            embed_dims[1], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.fusion2 = CSMambaBlock(
            embed_dims[0], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.dblock1 = shiftedBlock(
            embed_dims[1], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dblock2 = shiftedBlock(
            embed_dims[0], mlp_ratio=4.0, drop=module_drop_rate,
            drop_path=module_drop_path_rate,
        )
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = self.enhance_t1(out)
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t2 = self.enhance_t2(out)
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t3 = out

        out, height, width = self.patch_embed3(t3)
        out = self.block1(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        out = self.block2(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t4)
        out = self.fusion1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock1(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t3)
        out = self.fusion2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock2(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.final(out)

class BaseCSMambaShallowSea(nn.Module):
    """Original UNeXt plus CSMamba, ShallowEnhance, and SEA Attention.

    CA-Ghost blocks (the CoordAttention + GhostConv shifted MLP) and the
    complete model's auxiliary head are deliberately absent.  This is the
    fourth member of the 3-of-4 leave-one-out grid: it complements
    Base+CSMamba (only CSMamba), Base+CSMamba+SEA+CA-Ghost (no
    ShallowEnhance), Base+Shallow+SEA+CA-Ghost (no CSMamba) and
    Base+CSMamba+Shallow+CA-Ghost (no SEA).
    """

    def __init__(self, num_classes: int = 1, img_size: int = 256,
                 embed_dims=None, base_drop_rate: float = 0.0,
                 base_drop_path_rate: float = 0.0,
                 csmamba_drop_path_rate: float = 0.1,
                 attention_dropout: float = 0.0):
        super().__init__()
        embed_dims = embed_dims or [128, 160, 256]

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.enhance_t1 = ShallowEnhance(16, 16)
        self.enhance_t2 = ShallowEnhance(32, 32)
        self.attention1 = Sea_Attention(dim=32, key_dim=8, num_heads=2)
        self.attention2 = Sea_Attention(dim=128, key_dim=16, num_heads=4)

        self.patch_embed3 = OverlapPatchEmbed(img_size // 4, 3, 2, 128, embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size // 8, 3, 2, embed_dims[1], embed_dims[2])
        self.block1 = UNeXtShiftedBlock(embed_dims[1], base_drop_rate, base_drop_path_rate)
        self.block2 = UNeXtShiftedBlock(embed_dims[2], base_drop_rate, base_drop_path_rate)
        self.norm3 = nn.LayerNorm(embed_dims[1])
        self.norm4 = nn.LayerNorm(embed_dims[2])

        self.decoder1 = nn.Conv2d(embed_dims[2], embed_dims[1], 3, padding=1)
        self.decoder2 = nn.Conv2d(embed_dims[1], embed_dims[0], 3, padding=1)
        self.decoder3 = nn.Conv2d(embed_dims[0], 32, 3, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, padding=1)
        self.dbn1 = nn.BatchNorm2d(embed_dims[1])
        self.dbn2 = nn.BatchNorm2d(embed_dims[0])
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.fusion1 = CSMambaBlock(
            embed_dims[1], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.fusion2 = CSMambaBlock(
            embed_dims[0], drop_path=csmamba_drop_path_rate,
            attn_drop_rate=attention_dropout,
        )
        self.dblock1 = UNeXtShiftedBlock(embed_dims[1], base_drop_rate, base_drop_path_rate)
        self.dblock2 = UNeXtShiftedBlock(embed_dims[0], base_drop_rate, base_drop_path_rate)
        self.dnorm3 = nn.LayerNorm(embed_dims[1])
        self.dnorm4 = nn.LayerNorm(embed_dims[0])
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        batch = x.shape[0]
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = self.enhance_t1(out)
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(t1)), 2, 2))
        t2 = self.enhance_t2(self.attention1(out))
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(t2)), 2, 2))
        t3 = self.attention2(out)

        out, height, width = self.patch_embed3(t3)
        out = self.block1(out, height, width)
        out = self.norm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        out, height, width = self.patch_embed4(out)
        out = self.block2(out, height, width)
        out = self.norm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t4)
        out = self.fusion1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock1(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm3(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t3)
        out = self.fusion2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _, _, height, width = out.shape
        out = self.dblock2(out.flatten(2).transpose(1, 2), height, width)
        out = self.dnorm4(out)
        out = out.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=2, mode="bilinear"))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode="bilinear"))
        return self.final(out)
