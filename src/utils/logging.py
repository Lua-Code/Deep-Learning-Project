from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


def get_logger(name: str = "p2l", level: int = logging.INFO) -> logging.Logger:   
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


class Tracker(ABC):
    @abstractmethod
    def log_params(self, params: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()


class NullTracker(Tracker):    
    def log_params(self, params): pass
    def log_metrics(self, metrics, step): pass
    def log_artifact(self, path): pass
    def finish(self): pass


class MLflowTracker(Tracker):    
    def __init__(self, project: str, run_name: str, tracking_uri: Optional[str] = None):
        import mlflow
        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(project)
        mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        self.mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics, step):
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path):
        self.mlflow.log_artifact(str(path))

    def finish(self):
        self.mlflow.end_run()


class WandbTracker(Tracker):    
    def __init__(self, project: str, run_name: str, params: Optional[Mapping] = None):
        import wandb
        self.wandb = wandb
        wandb.init(project=project, name=run_name, config=dict(params or {}))

    def log_params(self, params):
        self.wandb.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics, step):
        self.wandb.log(dict(metrics), step=step)

    def log_artifact(self, path):
        self.wandb.save(str(path), policy="now")

    def finish(self):
        self.wandb.finish()


def make_tracker(backend: str, project: str, run_name: str,params: Optional[Mapping[str, Any]] = None,
                 tracking_uri: Optional[str] = None) -> Tracker:
    flat = flatten(params or {})
    backend = (backend or "none").lower()
    if backend == "mlflow":
        t: Tracker = MLflowTracker(project, run_name, tracking_uri)
    elif backend == "wandb":
        t = WandbTracker(project, run_name, flat)
    elif backend == "none":
        t = NullTracker()
    else:
        raise ValueError(f"unknown tracking backend '{backend}' (mlflow | wandb | none)")
    t.log_params(flat)
    return t


if __name__ == "__main__":
    #python -m src.utils.logging
    with make_tracker("none", "demo", "run0", {"train": {"lr": 0.1}}) as t:
        t.log_metrics({"val_acc": 0.5}, step=1)
    get_logger().info("ok")