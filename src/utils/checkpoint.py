from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
import torch
from torch import nn


def _atomic_save(obj: Any, path: str | Path) -> None:    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model: nn.Module, optimizer=None, scheduler=None,scaler=None, epoch: int = 0, best_metric: Optional[float] = None,
                    config: Optional[dict] = None, extra: Optional[dict] = None) -> None:
    _atomic_save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model: nn.Module, optimizer=None, scheduler=None, scaler=None, map_location="cpu", strict: bool = True) -> dict:    
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state


def save_backbone(path, encoder: nn.Module, encoder_name: str,extra: Optional[dict] = None) -> None:   
    _atomic_save({
        "state_dict": encoder.state_dict(),
        "encoder_name": encoder_name,
        "out_channels": list(getattr(encoder, "out_channels", [])),
        "extra": extra or {},
    }, path)


def load_backbone(path, encoder: nn.Module, map_location="cpu",strict: bool = True) -> dict:
    blob = torch.load(path, map_location=map_location, weights_only=False)
    saved, current = blob.get("out_channels"), list(getattr(encoder, "out_channels", []))
    if saved and current and saved != current:
        raise ValueError(f"backbone mismatch: checkpoint {saved} vs encoder {current}")
    encoder.load_state_dict(blob["state_dict"], strict=strict)
    return {k: v for k, v in blob.items() if k != "state_dict"}


if __name__ == "__main__":
    #python -m src.utils.checkpoint
    import tempfile
    m = nn.Linear(4, 2)
    opt = torch.optim.AdamW(m.parameters())
    with tempfile.TemporaryDirectory() as d:
        save_checkpoint(f"{d}/last.pt", m, opt, epoch=3, best_metric=0.5)
        m2 = nn.Linear(4, 2)
        st = load_checkpoint(f"{d}/last.pt", m2, torch.optim.AdamW(m2.parameters()))
        assert st["epoch"] == 3 and torch.equal(m.weight, m2.weight)
        m.out_channels = [4]
        save_backbone(f"{d}/bb.pt", m, "toy")
        m2.out_channels = [4]
        print(load_backbone(f"{d}/bb.pt", m2))
    print("ok")