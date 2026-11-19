"""Encoder-decoder depth networks: a torchvision backbone (from scratch) + a shared FPN decoder.

torchvision's ConvNeXt and Swin both expose `.features` as 8 modules:
[stem, stage1, down, stage2, down, stage3, down, stage4], so the 4 stage outputs (strides 4/8/16/32)
are taken at the odd indices. Swin returns channels-last tensors, which we permute.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm

BACKBONES = {
    'convnext_tiny': (tvm.convnext_tiny, False),
    'convnext_base': (tvm.convnext_base, False),
    'swin_t': (tvm.swin_t, True),
    'swin_b': (tvm.swin_b, True),
}
FAMILY = {'convnext_tiny': 'CNN', 'convnext_base': 'CNN', 'swin_t': 'ViT', 'swin_b': 'ViT'}


class Encoder(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        build, self.channels_last = BACKBONES[name]
        self.stages = build(weights=None).features          # from scratch, classifier head dropped
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self.forward(torch.zeros(1, 3, 64, 64))]

    def forward(self, x):
        feats = []
        for i, block in enumerate(self.stages):
            x = block(x)
            if i % 2 == 1:
                feats.append(x.permute(0, 3, 1, 2).contiguous() if self.channels_last else x)
        return feats


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(16, cout), nn.GELU())


class FPNDecoder(nn.Module):
    """Top-down feature pyramid -> 1-channel depth at 1/4 resolution -> upsampled to input size."""

    def __init__(self, in_channels, ch: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, ch, 1) for c in in_channels)
        self.smooth = nn.ModuleList(conv_block(ch, ch) for _ in in_channels)
        self.head = nn.Sequential(conv_block(ch, ch // 2), nn.Conv2d(ch // 2, 1, 3, padding=1))

    def forward(self, feats, out_size):
        x = self.smooth[-1](self.lateral[-1](feats[-1]))
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.smooth[i](x + self.lateral[i](feats[i]))
        x = F.interpolate(self.head(x), size=out_size, mode='bilinear', align_corners=False)
        return F.softplus(x)                                  # depth is strictly positive


class DepthNet(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = backbone
        self.encoder = Encoder(backbone)
        self.decoder = FPNDecoder(self.encoder.channels)

    def forward(self, x):
        return self.decoder(self.encoder(x), x.shape[-2:])


def count_params(model: nn.Module) -> float:
    """Trainable parameters, in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
