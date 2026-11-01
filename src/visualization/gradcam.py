from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data.transforms import denormalize


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self._feat: Optional[torch.Tensor] = None
        self._handle = model.pool.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self._feat = inputs[0]                       

    def remove(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor, target=None):
        device = next(self.model.parameters()).device
        x = x.to(device).clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            idx = (logits.argmax(1) if target is None
                   else torch.as_tensor(target, device=device))
            score = logits.gather(1, idx[:, None]).sum()
            grads = torch.autograd.grad(score, self._feat)[0]
        acts = self._feat.detach()
        self._feat = None
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        peak = cam.flatten(1).max(1).values.clamp_min(1e-8)
        return (cam / peak.view(-1, 1, 1)).cpu().numpy(), logits.detach()


def overlay(img_hwc: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img_hwc + alpha * heat, 0, 1)


def select_positions(preds: dict, kind: str, n: int = 8, seed: int = 0) -> np.ndarray:
    correct = preds["y_true"] == preds["y_pred"]
    if kind == "correct":
        pool = np.flatnonzero(correct)
        return np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False)
    if kind == "wrong":
        pool = np.flatnonzero(~correct)
        return pool[np.argsort(-preds["confidence"][pool])][:n]
    raise ValueError("kind must be 'correct' or 'wrong'")


def gradcam_figure(model, dataset, preds: dict, class_names: Sequence[str],positions: Sequence[int], save_path: Optional[str] = None,
                   target: str = "pred", suptitle: str = "", show: bool = False) -> None:
    positions = [int(p) for p in positions]
    n = len(positions)
    items = [dataset[p] for p in positions]
    x = torch.stack([it["image"] for it in items])
    tgt = preds["y_pred" if target == "pred" else "y_true"][positions]
    
    cam_engine = GradCAM(model)
    try:
        cams, _ = cam_engine(x, tgt)
    finally:
        cam_engine.remove()

    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.8), squeeze=False)
    for j, (p, it) in enumerate(zip(positions, items)):
        img = denormalize(it["image"]).permute(1, 2, 0).cpu().numpy()
        t, q = int(preds["y_true"][p]), int(preds["y_pred"][p])
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"true: {class_names[t]}\npred: {class_names[q]} "
                             f"({preds['confidence'][p]:.2f})", fontsize=7,
                             color="green" if t == q else "red")
        axes[1, j].imshow(overlay(img, cams[j]))
        axes[1, j].set_title(f"CAM for {target}", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # python -m src.visualization.gradcam
    from src.models.classifier import Classifier
    from src.models.encoders.registry import build_encoder

    model = Classifier(build_encoder("scratch_resnet"), num_classes=10).eval()
    cams, logits = GradCAM(model)(torch.randn(2, 3, 96, 96))
    assert cams.shape == (2, 96, 96) and 0 <= cams.min() and cams.max() <= 1.0 + 1e-6
    print("cam", cams.shape, "ok")