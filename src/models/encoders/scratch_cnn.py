"""
ResNet-style CNN trained completely from scratch (the baseline).
Stem: conv7x7/2 + BN + ReLU + maxpool/2   -> stride 4
Stages (2 blocks each): 64 @4, 128 @8, 256 @16, 512 @32
"""
from __future__ import annotations
from typing import Sequence
import torch
from torch import nn

from src.models.encoders.base import BaseEncoder


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()
        self.shortcut = None
        if stride != 1 or cin != cout:  
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class ScratchResNet(BaseEncoder):   
    def __init__(self,
                 widths: Sequence[int] = (64, 128, 256, 512),
                 blocks: Sequence[int] = (2, 2, 2, 2),
                 in_channels: int = 3,
                 zero_init_residual: bool = True):
        super().__init__()
        assert len(widths) == len(blocks) == 4
        self.out_channels = list(widths)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1))                      

        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2                
            stage = [BasicBlock(cin, w, stride)] + [BasicBlock(w, w) for _ in range(n - 1)]
            layers.append(nn.Sequential(*stage))
            cin = w
        self.layer1, self.layer2, self.layer3, self.layer4 = layers

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:       
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)      # stride 4
        f2 = self.layer2(f1)     # stride 8
        f3 = self.layer3(f2)     # stride 16
        f4 = self.layer4(f3)     # stride 32
        return [f1, f2, f3, f4]