from __future__ import annotations
import json
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, save_config
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logging import NullTracker, Tracker, get_logger


def _make_scaler(enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)     


class BaseTrainer(ABC):
    monitor: str = "val_loss"
    mode: str = "min"                

    def __init__(self, cfg: Config, model: nn.Module, train_loader: DataLoader,val_loader: DataLoader, 
                tracker: Optional[Tracker] = None,device: Optional[str] = None):
        self.cfg, self.tcfg = cfg, cfg.train
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.tracker = tracker or NullTracker()
        self.log = get_logger()
        self.ckpt_dir = Path(self.tcfg.checkpoint_dir)
        self.use_amp = bool(self.tcfg.amp) and self.device.type == "cuda"
        self.grad_clip = float(self.tcfg.get("grad_clip", 0.0))

        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler(self.optimizer, len(train_loader))
        self.scaler = _make_scaler(self.use_amp)

        self.history: list[dict] = []
        self.best: Optional[float] = None
        self.best_epoch = 0
        self.bad_epochs = 0


    @abstractmethod
    def build_optimizer(self) -> torch.optim.Optimizer: ...

    @abstractmethod
    def training_step(self, batch: dict):
        """Forward pass (already inside autocast). Return (loss, stats, n)"""

    @abstractmethod
    def validation_step(self, batch: dict):
        """Forward pass, no grad. Return (stats, n)"""

    def on_epoch_start(self, epoch: int) -> None:
        """Hook, called before model.train() (e.g. freeze / unfreeze schedule)"""

    def on_best(self) -> None:
        """Hook, called right after best.pt is written """


    def build_scheduler(self, optimizer, steps_per_epoch: int):

        kind = str(self.tcfg.get("scheduler", "cosine")).lower()
        total = int(self.tcfg.epochs) * steps_per_epoch
        warm = int(self.tcfg.get("warmup_epochs", 0)) * steps_per_epoch
        floor = 0.01

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / max(1, warm)
            if kind == "none":
                return 1.0
            p = min(1.0, (step - warm) / max(1, total - warm))
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


    def _to_device(self, batch: dict) -> dict:
        
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()}

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, enabled=self.use_amp)
    

    def _run_epoch(self, loader: DataLoader, train: bool, desc: str) -> dict:
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        with torch.enable_grad() if train else torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = self._to_device(batch)
                if train:
                    with self._autocast():
                        loss, stats, n = self.training_step(batch)
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                else:
                    with self._autocast():
                        stats, n = self.validation_step(batch)
                for k, v in stats.items():
                    totals[k] += v * n
                seen += n
        return {k: v / max(seen, 1) for k, v in totals.items()}

    def _is_better(self, value: float) -> bool:    
        if self.best is None:
            return True
        return value > self.best if self.mode == "max" else value < self.best

    def _maybe_resume(self) -> int:  
        last = self.ckpt_dir / "last.pt"
        if not (self.tcfg.get("resume", False) and last.exists()):
            return 0
        state = load_checkpoint(last, self.model, self.optimizer, self.scheduler,self.scaler, map_location=self.device)
        extra = state.get("extra", {})
        self.history = extra.get("history", [])
        self.best = state.get("best_metric")
        self.best_epoch = extra.get("best_epoch", 0)
        self.bad_epochs = extra.get("bad_epochs", 0)
        self.log.info(f"resumed from {last} at epoch {state['epoch']}")
        return int(state["epoch"])

    def fit(self) -> dict:   
        t, t0 = self.tcfg, time.time()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, self.ckpt_dir / "config.yaml")
        start = self._maybe_resume()
        patience = int(t.get("early_stopping_patience", 0) or 0)
        epochs = int(t.epochs)

        for epoch in range(start, epochs):
            e0 = time.time()
            self.on_epoch_start(epoch)
            self.model.train()
            tr = self._run_epoch(self.train_loader, True, f"train {epoch + 1}/{epochs}")
            self.model.eval()
            va = self._run_epoch(self.val_loader, False, f"val   {epoch + 1}/{epochs}")

            row = {"epoch": epoch + 1,
                   **{f"train_{k}": v for k, v in tr.items()},
                   **{f"val_{k}": v for k, v in va.items()},
                   "lr": max(g["lr"] for g in self.optimizer.param_groups),
                   "epoch_seconds": time.time() - e0}
            self.history.append(row)

            improved = self._is_better(row[self.monitor])
            if improved:
                self.best, self.best_epoch, self.bad_epochs = row[self.monitor], epoch + 1, 0
            else:
                self.bad_epochs += 1

            self._save(epoch + 1, "last.pt")
            if improved:
                self._save(epoch + 1, "best.pt")
                self.on_best()

            self.tracker.log_metrics({k: v for k, v in row.items() if k != "epoch"},
                                     step=epoch + 1)
            shown = " | ".join(f"{k}={v:.4f}" for k, v in row.items()
                               if k in ("train_loss", "train_acc", "val_loss", "val_acc"))
            self.log.info(f"epoch {epoch + 1}/{epochs} | {shown} | lr={row['lr']:.2e}"
                          f"{' *' if improved else ''}")

            if patience and self.bad_epochs >= patience:
                self.log.info(f"early stopping: no {self.monitor} gain for {patience} epochs")
                break

        hist_path = self.ckpt_dir / "history.json"
        hist_path.write_text(json.dumps(self.history, indent=2))
        self.tracker.log_params({"best_" + self.monitor: self.best,
                                 "best_epoch": self.best_epoch,
                                 "best_checkpoint": str(self.ckpt_dir / "best.pt")})
        self.tracker.log_artifact(str(hist_path))
        self.tracker.log_artifact(str(self.ckpt_dir / "config.yaml"))
        return {"best_metric": self.best, "monitor": self.monitor,
                "best_epoch": self.best_epoch, "epochs_run": len(self.history),
                "train_seconds": time.time() - t0,
                "best_checkpoint": str(self.ckpt_dir / "best.pt"),
                "history": self.history}

    def _save(self, epoch_done: int, name: str) -> None:
        save_checkpoint(self.ckpt_dir / name, self.model, self.optimizer, self.scheduler,
                        self.scaler, epoch=epoch_done, best_metric=self.best,
                        config=self.cfg.to_dict(),
                        extra={"history": self.history, "best_epoch": self.best_epoch,
                               "bad_epochs": self.bad_epochs})