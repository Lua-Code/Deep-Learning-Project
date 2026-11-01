from __future__ import annotations
import torch
from torch import nn

from src.models.encoders.base import BaseEncoder
from src.models.encoders.registry import build_encoder_from_cfg
from src.models.encoders.registry import build_encoder, build_encoder_from_cfg


class Classifier(nn.Module):
    def __init__(self, encoder: BaseEncoder, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)[-1]                   
        return self.fc(self.drop(self.pool(f).flatten(1)))


def build_classifier(model_cfg, num_classes: int) -> Classifier:

    encoder = build_encoder_from_cfg(model_cfg.encoder)
    return Classifier(encoder, num_classes, float(model_cfg.head.dropout))

def load_classifier(cfg, checkpoint, num_classes: int, device: str = "cpu") -> Classifier:
    encoder = build_encoder(cfg.model.encoder.name, pretrained=False)
    model = Classifier(encoder, num_classes, float(cfg.model.head.dropout))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


if __name__ == "__main__":
    # python -m src.models.classifier
    from src.models.encoders.registry import build_encoder
    m = Classifier(build_encoder("scratch_resnet"), num_classes=200)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 200)
    print("logits", tuple(out.shape), "ok")