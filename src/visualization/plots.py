from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.transforms import denormalize


def _finish(fig, save_path: Optional[str], show: bool) -> None:    
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_class_distribution(class_df: pd.DataFrame,save_path: Optional[str] = None, show: bool = False) -> None:
    
    order = class_df.sort_values("train_total", ascending=False).reset_index(drop=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.bar(range(len(order)), order["train_total"], width=1.0, label="train (incl. val)")
    a.plot(range(len(order)), order["test"], color="C1", lw=1.2, label="test")
    a.set(xlabel="class (sorted by train count)", ylabel="images",
          title="Images per class")
    a.legend()
    b.hist(class_df["train_total"], bins=15, color="C0", edgecolor="white")
    b.set(xlabel="train images in class", ylabel="number of classes",
          title="Distribution of class sizes")
    _finish(fig, save_path, show)


def plot_image_sizes(scans: pd.DataFrame,save_path: Optional[str] = None,show: bool = False, target: int = 224) -> None:   
    ok = scans[~scans["corrupt"].astype(bool)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.scatter(ok["width"], ok["height"], s=6, alpha=0.2)
    a.axhline(target, color="r", ls="--", lw=1, label=f"{target}px")
    a.axvline(target, color="r", ls="--", lw=1)
    a.set(xlabel="width (px)", ylabel="height (px)", title="Image sizes")
    a.legend()
    b.hist(ok["width"] / ok["height"], bins=40, edgecolor="white")
    b.set(xlabel="width / height", ylabel="images", title="Aspect ratio")
    _finish(fig, save_path, show)


def plot_caption_lengths(lengths: np.ndarray,save_path: Optional[str] = None,show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=np.arange(lengths.min(), lengths.max() + 2) - 0.5,
            edgecolor="white")
    for q, color in ((50, "g"), (95, "orange"), (99, "r")):
        v = np.percentile(lengths, q)
        ax.axvline(v, color=color, ls="--", label=f"p{q} = {v:.0f}")
    ax.set(xlabel="words per caption", ylabel="captions",
           title="Caption length")
    ax.legend()
    _finish(fig, save_path, show)


def plot_top_words(counts: Counter, n: int = 25,save_path: Optional[str] = None, show: bool = False) -> None:
    words, freq = zip(*counts.most_common(n))
    fig, ax = plt.subplots(figsize=(6, 0.28 * n + 1))
    ax.barh(range(n), freq[::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(words[::-1])
    ax.set(xlabel="occurrences", title=f"Top {n} words (train captions)")
    _finish(fig, save_path, show)


def _to_hwc(x: torch.Tensor) -> np.ndarray:   
    return denormalize(x).permute(1, 2, 0).cpu().numpy()


def show_samples(dataset, class_names: Sequence[str], n: int = 12, cols: int = 4,seed: int = 0, save_path: Optional[str] = None,
                 show: bool = False) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), picks):
        item = dataset[int(i)]
        ax.imshow(_to_hwc(item["image"]))
        ax.set_title(class_names[item["label"]], fontsize=8)
    _finish(fig, save_path, show)


def show_augmentations(dataset, transform: Callable, n_images: int = 4,n_aug: int = 5, seed: int = 0,save_path: Optional[str] = None,
                       show: bool = False) -> None:
    assert dataset.transform is None, "pass a dataset built with transform=None"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    picks = rng.choice(len(dataset), size=n_images, replace=False)
    fig, axes = plt.subplots(n_images, n_aug + 1,figsize=(2.6 * (n_aug + 1), 2.6 * n_images))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        pil = dataset[int(i)]["image"]
        axes[r, 0].imshow(pil)
        axes[r, 0].set_title("original", fontsize=8)
        for c in range(1, n_aug + 1):
            axes[r, c].imshow(_to_hwc(transform(pil)))
            axes[r, c].set_title(f"aug {c}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    _finish(fig, save_path, show)

def plot_training_curves(histories: dict, save_path: Optional[str] = None,show: bool = False) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, h) in enumerate(histories.items()):
        c, ep = f"C{i}", [r["epoch"] for r in h]
        a.plot(ep, [r["train_loss"] for r in h], "--", color=c, label=f"{name} train")
        a.plot(ep, [r["val_loss"] for r in h], "-", color=c, label=f"{name} val")
        b.plot(ep, [r["train_acc"] for r in h], "--", color=c, label=f"{name} train")
        b.plot(ep, [r["val_acc"] for r in h], "-", color=c, label=f"{name} val")
    a.set(xlabel="epoch", ylabel="loss", title="Loss")
    b.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    a.legend(fontsize=8)
    b.legend(fontsize=8)
    _finish(fig, save_path, show)


def plot_tuning_comparison(results_by_method: dict, save_path: Optional[str] = None,show: bool = False) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for m, res in results_by_method.items():
        scores = np.nan_to_num([r["score"] for r in res], nan=0.0)
        best = np.maximum.accumulate(scores)
        mins = np.cumsum([r["seconds"] for r in res]) / 60
        a.plot(range(1, len(res) + 1), best, marker="o", label=m)
        b.plot(mins, best, marker="o", label=m)
    a.set(xlabel="trial", ylabel="best val accuracy so far", title="Per trial")
    b.set(xlabel="minutes", ylabel="best val accuracy so far", title="Per wall-clock time")
    a.legend()
    b.legend()
    _finish(fig, save_path, show)

if __name__ == "__main__":
    # python -m src.visualization.plots
    import argparse

    from src.data.caption_utils import count_words, extract_captions
    from src.data.cub_dataset import CUBDataset, load_cub
    from src.data.eda import class_distribution
    from src.data.transforms import build_eval_transform, build_train_transform

    ap = argparse.ArgumentParser(description="Generate EDA figures")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--eda-dir", default="outputs/eda")
    ap.add_argument("--out-dir", default="outputs/figures")
    a = ap.parse_args()
    out = Path(a.out_dir)

    data = load_cub(a.cache_dir)
    plot_class_distribution(class_distribution(data), out / "class_distribution.png")
    plot_image_sizes(pd.read_csv(Path(a.eda_dir) / "image_scan.csv"),
                     out / "image_sizes.png")

    clean = extract_captions(data.train_hf, clean=True)
    lengths = np.array([len(c.split()) for caps in clean for c in caps])
    plot_caption_lengths(lengths, out / "caption_lengths.png")
    plot_top_words(count_words(clean), 25, out / "top_words.png")

    ds_eval = CUBDataset(data.train_hf, data.train_labels, data.train_idx,
                         build_eval_transform())
    show_samples(ds_eval, data.class_names, 12, save_path=out / "samples.png")
    ds_raw = CUBDataset(data.train_hf, data.train_labels, data.train_idx, None)
    show_augmentations(ds_raw, build_train_transform(),
                       save_path=out / "augmentations.png")
    print(f"Figures saved to {out}/")