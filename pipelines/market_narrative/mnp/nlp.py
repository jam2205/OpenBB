"""FinBERT tone + ModernBERT embeddings for driver attribution."""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd


def _device(pref: str) -> int | str:
    import torch

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


class NLP:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = _device(cfg.get("device", "auto"))

    @cached_property
    def _sentiment(self):
        from transformers import pipeline

        return pipeline(
            "text-classification",
            model=self.cfg["sentiment"],
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )

    @cached_property
    def _embedder(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg["embedder"])
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(self.cfg["embedder"], torch_dtype=dtype).to(self.device).eval()
        return tok, model

    def tone(self, texts: list[str], batch_size: int = 32) -> pd.DataFrame:
        """Label + signed score in [-1, 1] (Positive=+p, Negative=-p, Neutral=0)."""
        if not texts:
            return pd.DataFrame(columns=["label", "score", "signed"])
        res = self._sentiment(texts, batch_size=batch_size)
        df = pd.DataFrame(res)
        df["label"] = df["label"].str.lower()
        sign = df["label"].map({"positive": 1, "negative": -1}).fillna(0)
        df["signed"] = sign * df["score"]
        return df

    def embed(self, texts: list[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        """Mean-pooled, L2-normalised ModernBERT embeddings.

        ModernBERT-large is a base encoder (not sentence-tuned): similarities are useful for
        ranking/attribution, not as calibrated scores. Fine-tune later for sharper matching.
        """
        import torch

        tok, model = self._embedder
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i : i + batch_size], padding=True, truncation=True,
                          max_length=max_length, return_tensors="pt").to(self.device)
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                vecs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, model.config.hidden_size))

    def attribute_drivers(self, news: pd.DataFrame, drivers: dict[str, str]) -> pd.DataFrame:
        """Score each headline (FinBERT) and assign its closest driver (ModernBERT)."""
        if news.empty:
            return news
        texts = news["text"].tolist()
        tone = self.tone(texts)
        names = list(drivers)
        sims = self.embed(texts) @ self.embed(list(drivers.values())).T
        out = news.copy().reset_index(drop=True)
        out[["tone", "tone_score", "signed"]] = tone[["label", "score", "signed"]].values
        out["driver"] = [names[i] for i in sims.argmax(1)]
        out["driver_sim"] = sims.max(1)
        return out

    @staticmethod
    def driver_summary(scored: pd.DataFrame) -> pd.DataFrame:
        if scored.empty:
            return pd.DataFrame(columns=["driver", "count", "net_tone", "share"])
        g = scored.groupby("driver").agg(count=("signed", "size"), net_tone=("signed", "mean"))
        g["share"] = g["count"] / g["count"].sum()
        return g.sort_values("count", ascending=False).reset_index()
