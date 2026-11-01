from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# useful later lol
METRIC_DESCRIPTIONS = {
    "accuracy": "Share of images whose top prediction is the true species. Simple and "
                "interpretable; CUB is nearly balanced, so it is not misleading here.",
    "top5_accuracy": "Share of images whose true species is among the 5 highest-scored "
                     "predictions. Useful with 200 visually similar classes (e.g. warblers).",
    "precision_macro": "For each class: of the images predicted as that class, how many were "
                       "right; averaged over classes. Low = the model over-predicts some classes.",
    "recall_macro": "For each class: of that class's images, how many were found; averaged over "
                    "classes (equals balanced accuracy). Low = some species are often missed.",
    "f1_macro": "Harmonic mean of precision and recall per class, averaged with equal weight per "
                "class. Single number that penalises both kinds of error.",
    "f1_weighted": "Same as macro F1 but weighted by class size; close to macro F1 on CUB.",
}


@torch.no_grad()
def predict_loader(model, loader, device=None, amp: bool = True) -> dict[str, np.ndarray]:

    device = torch.device(device) if device else next(model.parameters()).device
    use_amp = amp and device.type == "cuda"
    model.eval()
    ys, tops, confs, idxs = [], [], [], []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        probs = logits.float().softmax(dim=1)
        conf, top = probs.topk(min(5, probs.size(1)), dim=1)
        ys.append(batch["label"].cpu()); idxs.append(batch["index"].cpu())
        tops.append(top.cpu()); confs.append(conf[:, 0].cpu())
    top5 = torch.cat(tops).numpy()
    return {"y_true": torch.cat(ys).numpy(), "y_pred": top5[:, 0], "top5": top5,
            "confidence": torch.cat(confs).numpy(), "index": torch.cat(idxs).numpy()}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,top5: Optional[np.ndarray] = None) -> dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",zero_division=0)
    wf1 = precision_recall_fscore_support(y_true, y_pred, average="weighted",zero_division=0)[2]
    out = {"accuracy": float((y_true == y_pred).mean()), "precision_macro": float(p),
           "recall_macro": float(r), "f1_macro": float(f1), "f1_weighted": float(wf1)}
    if top5 is not None:
        out["top5_accuracy"] = float((top5 == y_true[:, None]).any(axis=1).mean())
    return out


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:

    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def per_class_report(y_true, y_pred, class_names: Sequence[str]) -> pd.DataFrame:

    labels = np.arange(len(class_names))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels,zero_division=0)
    return (pd.DataFrame({"class": class_names, "precision": p, "recall": r,"f1": f1, "support": s}).sort_values("f1").reset_index(drop=True))


def most_confused(cm: np.ndarray, class_names: Sequence[str], k: int = 10):

    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:k]
    return [(class_names[i], class_names[j], int(off[i, j])) for i, j in zip(*np.unravel_index(order, off.shape)) if off[i, j] > 0]


if __name__ == "__main__":
    # python -m src.evaluation.cls_metrics
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 5, 200)
    yp = np.where(rng.random(200) < 0.7, yt, rng.integers(0, 5, 200))
    t5 = np.stack([yp] + [rng.integers(0, 5, 200) for _ in range(4)], axis=1)
    m = compute_metrics(yt, yp, t5)
    assert m["top5_accuracy"] >= m["accuracy"]
    names = [f"c{i}" for i in range(5)]
    cm = confusion(yt, yp, 5)
    print(m, "\n", per_class_report(yt, yp, names).head(2), "\n", most_confused(cm, names, 3))
    print("ok")