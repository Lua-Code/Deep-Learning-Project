"""
    python -m src.cli prepare-data
    python -m src.cli train   --task classification --config configs/cls_scratch.yaml [--set train.lr=0.0003 ...]
    python -m src.cli eval    --task classification --checkpoint checkpoints/X/best.pt [--split test]
    python -m src.cli predict --task classification --checkpoint checkpoints/X/best.pt --image bird.jpg [--cam out.png]
    python -m src.cli gradcam --task classification --checkpoint checkpoints/X/best.pt [--n 8]
    python -m src.cli tune    --task classification [--search configs/tuning/cls_search.yaml] [--methods grid random]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

TASKS = ("classification", "colorization", "caption")
HANDLERS: dict[str, dict[str, Callable]] = {c: {} for c in ("train", "eval", "predict", "gradcam", "tune")}


def handler(command: str, task: str):
    def deco(fn: Callable):
        HANDLERS[command][task] = fn
        return fn
    return deco


def _cfg_for_checkpoint(args):
    from src.config import load_config
    path = args.config or Path(args.checkpoint).parent / "config.yaml"
    return load_config(path)


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _split_dataset(data, split: str, transform):
    from src.data.cub_dataset import CUBDataset
    if split == "test":
        return CUBDataset(data.test_hf, data.test_labels, None, transform)
    if split == "val":
        return CUBDataset(data.train_hf, data.train_labels, data.val_idx, transform)
    raise ValueError("split must be 'val' or 'test'")


def _eval_setup(args):
    from src.data.cub_dataset import load_cub, make_loader
    from src.data.transforms import build_eval_transform
    from src.evaluation.cls_metrics import predict_loader
    from src.models.classifier import load_classifier

    cfg, device = _cfg_for_checkpoint(args), _device()
    data = load_cub(cfg.data.cache_dir, cfg.data.val_fraction, cfg.seed)
    model = load_classifier(cfg, args.checkpoint, data.num_classes, device)
    ds = _split_dataset(data, args.split, build_eval_transform(cfg.data.image_size))
    loader = make_loader(ds, cfg.data.batch_size, shuffle=False,num_workers=cfg.data.num_workers)
    return cfg, data, model, ds, predict_loader(model, loader, device)



def prepare_data(args) -> None:
    from src.data.caption_utils import extract_captions, save_captions
    from src.data.cub_dataset import load_cub

    data = load_cub(args.cache_dir, args.val_fraction, args.seed)
    out = Path(args.out_dir)
    save_captions(extract_captions(data.train_hf), out / "captions_train.json")
    save_captions(extract_captions(data.test_hf), out / "captions_test.json")
    (out / "splits.json").write_text(json.dumps({
        "seed": args.seed, "val_fraction": args.val_fraction,
        "train_idx": data.train_idx.tolist(), "val_idx": data.val_idx.tolist(),
        "class_names": data.class_names}))
    print(f"saved captions_train.json, captions_test.json, splits.json to {out}/")


@handler("train", "classification")
def train_cls(args) -> None:
    from src.config import load_config
    from src.training.cls_trainer import run_classification

    result = run_classification(load_config(args.config, args.set))
    print({k: v for k, v in result.items() if k != "history"})


@handler("eval", "classification")
def eval_cls(args) -> None:
    import numpy as np
    from src.evaluation.cls_metrics import (METRIC_DESCRIPTIONS, compute_metrics,confusion, most_confused, per_class_report)

    cfg, data, _, _, preds = _eval_setup(args)
    metrics = compute_metrics(preds["y_true"], preds["y_pred"], preds["top5"])
    cm = confusion(preds["y_true"], preds["y_pred"], data.num_classes)
    report = per_class_report(preds["y_true"], preds["y_pred"], data.class_names)

    out = Path(args.out_dir or f"outputs/eval/{cfg.experiment}_{args.split}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report.to_csv(out / "per_class.csv", index=False)
    np.save(out / "confusion.npy", cm)
    np.savez(out / "predictions.npz", **preds)

    print(f"\n{cfg.experiment} on {args.split} ({len(preds['y_true'])} images)")
    for k, v in metrics.items():
        print(f"  {k:16s} {v:.4f}   {METRIC_DESCRIPTIONS.get(k, '')[:70]}")
    print("\nworst classes by F1:")
    print(report.head(5).to_string(index=False))
    print("\nmost confused (true -> predicted, count):")
    for t, p, c in most_confused(cm, data.class_names, 8):
        print(f"  {t} -> {p}: {c}")
    print(f"\nsaved to {out}/")


@handler("gradcam", "classification")
def gradcam_cls(args) -> None:
    from src.visualization.gradcam import gradcam_figure, select_positions

    cfg, data, model, ds, preds = _eval_setup(args)
    out = Path(args.out_dir or f"outputs/gradcam/{cfg.experiment}_{args.split}")
    for kind in ("correct", "wrong"):
        pos = select_positions(preds, kind, args.n, args.seed)
        gradcam_figure(model, ds, preds, data.class_names, pos,
                       out / f"gradcam_{kind}.png",
                       suptitle=f"{cfg.experiment}: {kind} predictions ({args.split})")
    print(f"saved gradcam_correct.png and gradcam_wrong.png to {out}/")


@handler("predict", "classification")
def predict_cls(args) -> None:
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from src.data.cub_dataset import load_cub
    from src.data.transforms import build_eval_transform, denormalize
    from src.models.classifier import load_classifier
    from src.visualization.gradcam import GradCAM, overlay

    cfg, device = _cfg_for_checkpoint(args), _device()
    data = load_cub(cfg.data.cache_dir, cfg.data.val_fraction, cfg.seed)  
    model = load_classifier(cfg, args.checkpoint, data.num_classes, device)
    img = Image.open(args.image).convert("RGB")
    x = build_eval_transform(cfg.data.image_size)(img).unsqueeze(0)
    with torch.no_grad():
        probs = model(x.to(device)).softmax(1)[0].cpu()
    conf, idx = probs.topk(min(args.topk, len(probs)))
    for c, i in zip(conf, idx):
        print(f"{c.item():6.1%}  {data.class_names[int(i)]}")
    if args.cam:
        cam, _ = GradCAM(model)(x, [int(idx[0])])
        plt.imsave(args.cam, overlay(denormalize(x[0]).permute(1, 2, 0).numpy(), cam[0]))
        print(f"Grad-CAM saved to {args.cam}")


@handler("tune", "classification")
def tune_cls(args) -> None:
    from src.config import load_config
    from src.training.tuning import load_search, run_tuning

    search = load_search(args.search)
    base_cfg = load_config(search["base_config"], args.set)
    out = run_tuning(search, base_cfg, args.methods, args.out_dir)
    print("\n" + out["summary"].drop(columns="best_params").to_string(index=False))
    sets = " ".join(f"'{k}={json.dumps(v)}'" for k, v in out["best_params"].items())
    print(f"\nbest: {out['best_method']} search -> retrain for the full schedule with:\n"
          f"python -m src.cli train --task classification "
          f"--config {search['base_config']} --set {sets}")



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli",description="train / eval / predict")
    sub = p.add_subparsers(dest="command", required=True)

    def with_task(sp):
        sp.add_argument("--task", required=True, choices=TASKS)
        return sp

    sp = sub.add_parser("prepare-data", help="cache cleaned captions and split indices")
    sp.add_argument("--cache-dir", default=None)
    sp.add_argument("--out-dir", default="data/processed")
    sp.add_argument("--val-fraction", type=float, default=0.15)
    sp.add_argument("--seed", type=int, default=42)

    sp = with_task(sub.add_parser("train", help="train a model from a config"))
    sp.add_argument("--config", required=True)
    sp.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",help="config overrides, e.g. train.lr=0.0003")

    sp = with_task(sub.add_parser("eval", help="metrics on val or test"))
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--config", default=None, help="default: config.yaml next to the checkpoint")
    sp.add_argument("--split", choices=("val", "test"), default="test")
    sp.add_argument("--out-dir", default=None)

    sp = with_task(sub.add_parser("predict", help="predict one image"))
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--config", default=None)
    sp.add_argument("--image", required=True)
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--cam", default=None, help="also save a Grad-CAM overlay to this PNG")

    sp = with_task(sub.add_parser("gradcam", help="Grad-CAM on correct and wrong predictions"))
    sp.add_argument("--checkpoint", required=True)
    sp.add_argument("--config", default=None)
    sp.add_argument("--split", choices=("val", "test"), default="test")
    sp.add_argument("--n", type=int, default=8)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out-dir", default=None)

    sp = with_task(sub.add_parser("tune", help="grid + random search, compared"))
    sp.add_argument("--search", default="configs/tuning/cls_search.yaml")
    sp.add_argument("--methods", nargs="+", choices=("grid", "random"),default=["grid", "random"])
    sp.add_argument("--out-dir", default="outputs/tuning")
    sp.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    return p


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-data":
        return prepare_data(args)
    fn = HANDLERS[args.command].get(args.task)
    if fn is None:
        parser.error(f"'{args.command}' for task '{args.task}' is not implemented yet")
    fn(args)


if __name__ == "__main__":
    # python -m src.cli
    main()