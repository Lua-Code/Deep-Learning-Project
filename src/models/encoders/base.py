from __future__ import annotations
import torch
from torch import nn

STRIDES = (4, 8, 16, 32)


class BaseEncoder(nn.Module):   
    out_channels: list[int]          

    def __init__(self) -> None: 
        super().__init__()
        self._frozen = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:     
        raise NotImplementedError

    @property
    def feature_dim(self) -> int:
        return self.out_channels[-1]


    def freeze(self) -> None:
        self._frozen = True
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        self._frozen = False
        for p in self.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):        
        return super().train(mode and not self._frozen)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:   
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)