from __future__ import annotations
import timm
import torch

from src.models.encoders.base import STRIDES, BaseEncoder


class PretrainedEncoder(BaseEncoder):
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.name = name
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True)
        info = self.body.feature_info
        reductions, channels = info.reduction(), info.channels()
        self._keep = [i for i, r in enumerate(reductions) if r in STRIDES]
        kept = [reductions[i] for i in self._keep]
        if kept != list(STRIDES):
            raise ValueError(f"{name}: expected strides {list(STRIDES)}, got {kept}")
        self.out_channels = [channels[i] for i in self._keep]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.body(x)
        return [feats[i] for i in self._keep]