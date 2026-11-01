from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import yaml


class Config:

    def __init__(self, data: Mapping[str, Any]):
        d = {k: Config(v) if isinstance(v, Mapping) else v for k, v in data.items()}
        object.__setattr__(self, "_d", d) #this reference the default setattr method not the custom one :O

    def __getattr__(self, name: str):
        d = object.__getattribute__(self, "_d")
        try:
            return d[name]
        except KeyError:
            raise AttributeError(f"config has no field '{name}'") from None #remove the context chaining and shows the attribute error bas

    def __setattr__(self, name: str, value: Any) -> None:    
        self[name] = value

    def __getitem__(self, key: str):     
        return self._d[key]

    def __setitem__(self, key: str, value: Any) -> None:     
        self._d[key] = Config(value) if isinstance(value, Mapping) else value

    def __contains__(self, key: str) -> bool:     
        return key in self._d

    def get(self, key: str, default: Any = None):       
        return self._d.get(key, default)

    def to_dict(self) -> dict:  #conversion layer for pyYAML    
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self._d.items()}

    def __repr__(self) -> str:       
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def override(self, overrides: Mapping[str, Any]) -> "Config":
       
        out = Config(copy.deepcopy(self.to_dict()))
        for key, value in overrides.items():
            *parents, last = key.split(".")
            node = out
            for p in parents:
                if p not in node or not isinstance(node[p], Config):
                    raise KeyError(f"unknown config path: {key}")
                node = node[p]
            if last not in node:
                raise KeyError(f"unknown config key: {key}")
            node[last] = value
        return out


def parse_overrides(items: Sequence[str]) -> dict:
    out = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got '{item}'")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_config(path: Union[str, Path],overrides: Optional[Union[Sequence[str], Mapping[str, Any]]] = None) -> Config:
   
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config(raw)
    if overrides:
        if not isinstance(overrides, Mapping):
            overrides = parse_overrides(overrides)
        cfg = cfg.override(overrides)
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(repr(cfg), encoding="utf-8")


if __name__ == "__main__":
    # python -m src.config
    cfg = load_config("configs/cls_scratch.yaml", ["train.lr=0.0003", "data.aug.rotation=0"])
    assert cfg.train.lr == 0.0003 and cfg.data.aug.rotation == 0
    assert cfg.data.aug.crop_scale == [0.5, 1.0]
    try:
        cfg.override({"train.lrr": 1})
    except KeyError as e:
        print("typo caught:", e)
    print(cfg.experiment, cfg.model.encoder.name, "ok")