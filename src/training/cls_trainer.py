from __future__ import annotations
from typing import Optional
import torch
from torch import nn

from src.config import Config
from src.data.cub_dataset import CUBData, CUBDataset, load_cub, make_loader
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.classifier import Classifier, build_classifier
from src.training.base_trainer import BaseTrainer
from src.utils.checkpoint import save_backbone
from src.utils.logging import Tracker, make_tracker
from src.utils.seed import set_seed


class ClassificationTrainer(BaseTrainer):
    
    monitor, mode = "val_acc", "max"

    def __init__(self, cfg, model: Classifier, train_loader, val_loader, tracker=None,device=None):
        super().__init__(cfg, model, train_loader, val_loader, tracker, device)
        self.train_criterion = nn.CrossEntropyLoss(
            label_smoothing=float(self.tcfg.get("label_smoothing", 0.0)))
        self.val_criterion = nn.CrossEntropyLoss()

    def build_optimizer(self):
        t = self.tcfg
        lr, wd = float(t.lr), float(t.weight_decay)
        mult = float(t.get("backbone_lr_mult", 1.0))
        groups = []
        for module, group_lr in ((self.model.encoder, lr * mult),
                                 (self.model.fc, lr)):
            decay = [p for p in module.parameters() if p.ndim > 1]
            no_decay = [p for p in module.parameters() if p.ndim <= 1]
            groups.append({"params": decay, "lr": group_lr, "weight_decay": wd})
            groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
        name = str(t.get("optimizer", "adamw")).lower()
        if name == "adamw":
            return torch.optim.AdamW(groups)
        if name == "sgd":
            return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
        raise ValueError(f"unknown optimizer '{name}' (adamw | sgd)")


    def on_epoch_start(self, epoch: int) -> None:
        freeze_epochs = int(self.tcfg.get("freeze_backbone_epochs", 0))
        enc = self.model.encoder
        if epoch < freeze_epochs:
            enc.freeze()
        elif enc._frozen:
            enc.unfreeze()
            self.log.info(f"epoch {epoch + 1}: backbone unfrozen")

    def training_step(self, batch):  
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.train_criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"loss": loss.item(), "acc": acc}, y.size(0)

    def validation_step(self, batch):    
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.val_criterion(logits.float(), y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "acc": acc}, y.size(0)

    def on_best(self) -> None:    
        save_backbone(self.ckpt_dir / "backbone_best.pt", self.model.encoder,self.cfg.model.encoder.name, extra={"val_acc": self.best})


def run_classification(cfg: Config, data: Optional[CUBData] = None,tracker: Optional[Tracker] = None) -> dict:
    set_seed(cfg.seed)
    d, aug = cfg.data, cfg.data.aug
    data = data or load_cub(d.cache_dir, d.val_fraction, cfg.seed)

    train_tf = build_train_transform(d.image_size, tuple(aug.crop_scale), aug.rotation,aug.brightness, aug.contrast, aug.saturation)
    eval_tf = build_eval_transform(d.image_size)
    train_ds = CUBDataset(data.train_hf, data.train_labels, data.train_idx, train_tf)
    val_ds = CUBDataset(data.train_hf, data.train_labels, data.val_idx, eval_tf)
    train_loader = make_loader(train_ds, d.batch_size, shuffle=True,num_workers=d.num_workers, drop_last=True, seed=cfg.seed)
    val_loader = make_loader(val_ds, d.batch_size, shuffle=False, num_workers=d.num_workers)

    model = build_classifier(cfg.model, data.num_classes)
    own_tracker = tracker is None
    if own_tracker:
        tr = cfg.tracking
        tracker = make_tracker(tr.backend, tr.project, cfg.experiment, cfg.to_dict(),tr.get("tracking_uri"))
    try:
        return ClassificationTrainer(cfg, model, train_loader, val_loader, tracker).fit()
    finally:
        if own_tracker:
            tracker.finish()


if __name__ == "__main__":
    # python -m src.training.cls_trainer
    import tempfile
    from pathlib import Path
    from torch.utils.data import DataLoader, Dataset

    class Fake(Dataset):
        def __init__(self, n=64, c=5):
            g = torch.Generator().manual_seed(0)
            self.y = torch.randint(0, c, (n,), generator=g)
            self.x = torch.randn(n, 3, 64, 64, generator=g) + self.y.view(-1, 1, 1, 1) * 0.5
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return {"image": self.x[i], "label": int(self.y[i]), "index": i}

    with tempfile.TemporaryDirectory() as tmp:
        def make_cfg(epochs, resume, freeze):
            return Config({"experiment": "smoke", "seed": 0,
                "model": {"encoder": {"name": "scratch_resnet", "pretrained": False},
                          "head": {"dropout": 0.1}},
                "train": {"epochs": epochs, "optimizer": "adamw", "lr": 1e-3,
                          "weight_decay": 0.01, "label_smoothing": 0.1,
                          "scheduler": "cosine", "warmup_epochs": 1, "amp": False,
                          "early_stopping_patience": 5, "freeze_backbone_epochs": freeze,
                          "backbone_lr_mult": 1.0, "checkpoint_dir": tmp, "resume": resume}})

        def trainer(cfg):
            ld = lambda: DataLoader(Fake(), batch_size=16, shuffle=True, drop_last=True)
            return ClassificationTrainer(cfg, build_classifier(cfg.model, 5), ld(), ld())

        r = trainer(make_cfg(3, False, 0)).fit()
        assert r["epochs_run"] == 3
        r = trainer(make_cfg(5, True, 0)).fit()          
        assert r["epochs_run"] == 5, r["epochs_run"]
        for f in ("last.pt", "best.pt", "backbone_best.pt", "history.json", "config.yaml"):
            assert (Path(tmp) / f).exists(), f

        t = trainer(make_cfg(2, False, 1))
        t.on_epoch_start(0)
        assert not any(p.requires_grad for p in t.model.encoder.parameters())
        t.on_epoch_start(1)
        assert all(p.requires_grad for p in t.model.encoder.parameters())
    print("ok")