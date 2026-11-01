from __future__ import annotations
import gc
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Optional, Sequence
import pandas as pd
import yaml

from src.config import Config
from src.data.cub_dataset import CUBData, load_cub
from src.training.cls_trainer import run_classification
from src.utils.logging import get_logger

METHODS = ("grid", "random")



def load_search(path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def grid_trials(space: dict) -> list[dict]:
    names = list(space)
    return [dict(zip(names, combo))
            for combo in itertools.product(*(space[n]["grid"] for n in names))]


def _sample(spec: dict, rng: random.Random) -> Any:
    kind = spec["dist"]
    if kind == "loguniform":
        v = math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))
        return float(f"{v:.3g}")                     # readable values in logs/reports
    if kind == "uniform":
        return float(f"{rng.uniform(spec['low'], spec['high']):.3g}")
    if kind == "choice":
        return rng.choice(spec["values"])
    raise ValueError(f"unknown distribution '{kind}' (loguniform | uniform | choice)")


def random_trials(space: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [{name: _sample(s["random"], rng) for name, s in space.items()}
            for _ in range(n)]



def _load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def run_search(method: str, trials: Sequence[dict], base_cfg: Config, data: CUBData,trial_epochs: int, backend: str, out_dir: Path) -> list[dict]:
    log = get_logger()
    out = out_dir / method
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = _load_results(res_path)
    done = {r["trial"] for r in results}

    for i, params in enumerate(trials):
        if i in done:
            continue
        trial_dir = out / f"trial_{i:02d}"
        cfg = base_cfg.override({
            **params,
            "experiment": f"tune_{method}_{i:02d}",
            "train.epochs": trial_epochs,
            "train.checkpoint_dir": str(trial_dir),
            "train.resume": False,
            "tracking.backend": backend,
        })
        log.info(f"[{method}] trial {i + 1}/{len(trials)}: {params}")
        t0 = time.time()
        try:
            r = run_classification(cfg, data=data)
            score, best_epoch, status = float(r["best_metric"]), r["best_epoch"], "ok"
        except Exception as e:                    
            log.error(f"[{method}] trial {i} failed: {type(e).__name__}: {e}")
            score, best_epoch, status = float("nan"), 0, f"failed: {type(e).__name__}"
        finally:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            for f in trial_dir.glob("*.pt"):
                f.unlink()
        results.append({"trial": i, "params": params, "score": score,
                        "best_epoch": best_epoch, "seconds": time.time() - t0,
                        "status": status})
        res_path.write_text(json.dumps(results, indent=2))
        log.info(f"[{method}] trial {i + 1} -> val_acc={score:.4f}")
    return results



def summarize(results_by_method: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for method, res in results_by_method.items():
        ok = [r for r in res if r["status"] == "ok"]
        if not ok:
            continue
        res = sorted(res, key=lambda r: r["trial"])
        best = max(ok, key=lambda r: r["score"])
        cum, time_to_best = 0.0, 0.0
        for r in res:
            cum += r["seconds"]
            if r is best:
                time_to_best = cum
        scores = [r["score"] for r in ok]
        rows.append({
            "method": method, "trials": len(res), "failed": len(res) - len(ok),
            "best_val_acc": best["score"], "mean_val_acc": sum(scores) / len(scores),
            "worst_val_acc": min(scores), "trial_of_best": best["trial"] + 1,
            "time_to_best_min": round(time_to_best / 60, 1),
            "total_time_min": round(cum / 60, 1),
            "best_params": json.dumps(best["params"]),
        })
    return pd.DataFrame(rows)


def run_tuning(search: dict, base_cfg: Config, methods: Sequence[str] = METHODS,out_dir: str = "outputs/tuning",
               backend: Optional[str] = None) -> dict:
    log = get_logger()
    out = Path(out_dir)
    space = search["space"]
    grid = grid_trials(space)
    plans = {"grid": grid,"random": random_trials(space, len(grid), int(search.get("seed", 42)))}
    data = load_cub(base_cfg.data.cache_dir, base_cfg.data.val_fraction, base_cfg.seed)
    backend = backend or base_cfg.tracking.backend
    log.info(f"budget: {len(grid)} trials per method, {search['trial_epochs']} epochs each")

    for m in methods:
        run_search(m, plans[m], base_cfg, data, int(search["trial_epochs"]), backend, out)

    results = {m: _load_results(out / m / "results.json") for m in METHODS if (out / m / "results.json").exists()}
    summary = summarize(results)
    summary.to_csv(out / "comparison.csv", index=False)
    top = summary.sort_values("best_val_acc", ascending=False).iloc[0]
    best_params = json.loads(top["best_params"])
    (out / "best_params.json").write_text(json.dumps({"method": top["method"], "params": best_params}, indent=2))
    return {"results": results, "summary": summary,"best_method": top["method"], "best_params": best_params}


if __name__ == "__main__":
    # python -m src.training.tuning
    space = load_search("configs/tuning/cls_search.yaml")["space"]
    g = grid_trials(space)
    r1, r2 = random_trials(space, len(g), 42), random_trials(space, len(g), 42)
    assert len(g) == len(r1) == 12 and r1 == r2
    fake = {m: [{"trial": i, "params": p, "score": 0.5 + 0.01 * i, "best_epoch": 3,
                 "seconds": 60.0, "status": "ok"} for i, p in enumerate(t)]
            for m, t in (("grid", g), ("random", r1))}
    print(summarize(fake).drop(columns="best_params").to_string(index=False))
    print("ok")