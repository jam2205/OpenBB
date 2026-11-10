"""Adapter for AcyLa/multi-asset-predictive-model.

The repo's artifact format is detected at load time (transformers / sklearn-joblib / torch /
ONNX) because it could not be inspected when this was written. Run `inspect()` once in Colab,
then set FEATURE_COLUMNS / the call in `predict()` to match the model card's input contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Fill in from the model card once inspected. None -> pass the engineered frame as-is.
FEATURE_COLUMNS: list[str] | None = None


class Predictive:
    def __init__(self, repo_id: str):
        from huggingface_hub import snapshot_download

        self.path = Path(snapshot_download(repo_id))
        self.kind, self.model = self._load()

    def inspect(self) -> dict:
        files = sorted(str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file())
        info = {"kind": self.kind, "files": files}
        readme = self.path / "README.md"
        if readme.exists():
            info["readme_head"] = readme.read_text()[:3000]
        cfg = self.path / "config.json"
        if cfg.exists():
            info["config"] = json.loads(cfg.read_text())
        return info

    def _load(self):
        p = self.path
        if (p / "config.json").exists() and (any(p.glob("*.safetensors")) or any(p.glob("*.bin"))):
            from transformers import AutoModel

            return "transformers", AutoModel.from_pretrained(p, trust_remote_code=False)
        for pat in ("*.joblib", "*.pkl"):
            hit = next(p.rglob(pat), None)
            if hit:
                import joblib

                # Only load pickles from repos you trust: unpickling executes code.
                return "sklearn", joblib.load(hit)
        hit = next(p.rglob("*.onnx"), None)
        if hit:
            import onnxruntime as ort

            return "onnx", ort.InferenceSession(str(hit))
        hit = next(p.rglob("*.pt"), None) or next(p.rglob("*.pth"), None)
        if hit:
            import torch

            return "torch", torch.load(hit, map_location="cpu", weights_only=False)
        log.warning("No recognised model artifact in %s", p)
        return "unknown", None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X = features[FEATURE_COLUMNS] if FEATURE_COLUMNS else features
        if self.kind == "sklearn":
            y = self.model.predict(X)
        elif self.kind == "onnx":
            name = self.model.get_inputs()[0].name
            y = self.model.run(None, {name: X.to_numpy("float32")})[0]
        else:
            raise NotImplementedError(
                f"Wire predict() for kind={self.kind!r}; see inspect() output and the model card."
            )
        y = pd.DataFrame(y, index=X.index)
        y.columns = [f"pred_{c}" for c in y.columns]
        return y


def build_features(rets: pd.DataFrame, driver_tone: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generic feature block: lagged returns, rolling vol/momentum, plus NLP tone per driver.

    Replace or extend to match the model's documented features.
    """
    f = {}
    for c in rets:
        f[f"{c}_r1"] = rets[c]
        f[f"{c}_mom5"] = rets[c].rolling(5).sum()
        f[f"{c}_vol20"] = rets[c].rolling(20).std()
    feats = pd.DataFrame(f).dropna()
    if driver_tone is not None and not driver_tone.empty:
        for _, row in driver_tone.iterrows():
            feats[f"nlp_{row['driver']}_tone"] = row["net_tone"]
    return feats
