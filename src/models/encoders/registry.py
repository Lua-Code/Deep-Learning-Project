from __future__ import annotations
from typing import Callable
from src.models.encoders.base import BaseEncoder
from src.models.encoders.pretrained import PretrainedEncoder
from src.models.encoders.scratch_cnn import ScratchResNet

_REGISTRY: dict[str, Callable[..., BaseEncoder]] = {}


def register(name: str):
    def deco(fn: Callable[..., BaseEncoder]):
        if name in _REGISTRY:
            raise KeyError(f"encoder '{name}' already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


@register("scratch_resnet")
def _scratch(**kwargs) -> BaseEncoder:
    
    return ScratchResNet(**kwargs)


def build_encoder(name: str, pretrained: bool = False, **kwargs) -> BaseEncoder:    
    if name in _REGISTRY:
        if pretrained:
            raise ValueError(f"'{name}' is a from-scratch encoder; set pretrained: false")
        return _REGISTRY[name](**kwargs)
    return PretrainedEncoder(name, pretrained=pretrained, **kwargs)


def build_encoder_from_cfg(enc_cfg) -> BaseEncoder:
    return build_encoder(enc_cfg.name, bool(enc_cfg.get("pretrained", False)))


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


if __name__ == "__main__":
    # python -m src.models.encoders.registry
    
    import torch
    from src.models.encoders.base import count_parameters

    x = torch.randn(2, 3, 224, 224)
    for name in ("scratch_resnet", "resnet50", "efficientnet_b0"):
        enc = build_encoder(name, pretrained=False)
        feats = enc(x)
        assert [f.shape[-1] for f in feats] == [56, 28, 14, 7], name
        assert [f.shape[1] for f in feats] == enc.out_channels, name
        print(f"{name:16s} channels={enc.out_channels} "f"params={count_parameters(enc) / 1e6:.1f}M")

    enc = build_encoder("scratch_resnet")
    enc.freeze(); enc.train()
    assert not any(p.requires_grad for p in enc.parameters()) and not enc.training
    enc.unfreeze(); enc.train()
    assert all(p.requires_grad for p in enc.parameters()) and enc.training
    print("ok")