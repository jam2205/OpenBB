"""NLP layer.

- Tone: Modern-FinBERT-large (ModernBERT fine-tuned for financial sentiment), with
  FinBERT-tone as an optional second opinion; a local fine-tuned head overrides both.
- Drivers: fine-tuned head if present, else pre-trained ModernBERT zero-shot classifier,
  else embedding similarity.
- Long documents (transcripts, MD&A): chunked to `doc_chunk_tokens` and length-weighted.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd


def _device(pref: str) -> str:
    import torch

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def modernbert_kwargs(device: str) -> dict:
    """bf16 + Flash Attention 2 on GPUs that support them (Ampere+), as the model card recommends."""
    import torch

    kw: dict = {}
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        kw["torch_dtype"] = torch.bfloat16 if major >= 8 else torch.float16
        try:
            import flash_attn  # noqa: F401

            if major >= 8:
                kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    return kw


def _exists(path: str | None) -> bool:
    return bool(path) and (Path(path) / "config.json").exists()


class NLP:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = _device(cfg.get("device", "auto"))
        self.tone_head = cfg.get("tone_head") if _exists(cfg.get("tone_head")) else None
        self.driver_head = cfg.get("driver_head") if _exists(cfg.get("driver_head")) else None

    # ---------- loaders ----------
    def _clf(self, model_id: str, max_length: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        kw = modernbert_kwargs(self.device) if "modern" in str(model_id).lower() or Path(str(model_id)).exists() else {}
        model = AutoModelForSequenceClassification.from_pretrained(model_id, **kw)
        tok = AutoTokenizer.from_pretrained(model_id)
        return pipeline("text-classification", model=model, tokenizer=tok,
                        device=0 if self.device == "cuda" else -1, truncation=True, max_length=max_length)

    @cached_property
    def _tone(self):
        return self._clf(self.tone_head or self.cfg["sentiment"], self.cfg.get("max_length", 8192))

    @cached_property
    def _tone_check(self):
        return self._clf(self.cfg["sentiment_check"], 512)

    @cached_property
    def _drivers(self):
        return self._clf(self.driver_head, 512)

    @cached_property
    def _zero_shot(self):
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=self.cfg["zero_shot"],
                        device=0 if self.device == "cuda" else -1, **{"model_kwargs": modernbert_kwargs(self.device)})

    @cached_property
    def _embedder(self):
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg["embedder"])
        model = AutoModel.from_pretrained(self.cfg["embedder"], **modernbert_kwargs(self.device))
        return tok, model.to(self.device).eval()

    # ---------- tone ----------
    def tone(self, texts: list[str], batch_size: int = 32, check: bool = False) -> pd.DataFrame:
        """Label + signed score in [-1, 1] (positive=+p, negative=-p, neutral=0)."""
        if not texts:
            return pd.DataFrame(columns=["label", "score", "signed"])
        clf = self._tone_check if check else self._tone
        df = pd.DataFrame(clf(texts, batch_size=batch_size))
        df["label"] = df["label"].str.lower()
        df["signed"] = df["label"].map({"positive": 1, "negative": -1}).fillna(0) * df["score"]
        return df

    # ---------- embeddings ----------
    def embed(self, texts: list[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        import torch

        tok, model = self._embedder
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [self.cfg.get("embed_prefix", "") + t for t in texts[i : i + batch_size]]
                enc = tok(batch, padding=True, truncation=True, max_length=max_length,
                          return_tensors="pt", return_token_type_ids=False).to(self.device)
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                vecs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.vstack(vecs) if vecs else np.zeros((0, model.config.hidden_size))

    # ---------- drivers ----------
    def attribute_drivers(self, news: pd.DataFrame, drivers: dict[str, str], threshold: float = 0.35) -> pd.DataFrame:
        if news.empty:
            return news
        texts = news["text"].tolist()
        tone = self.tone(texts)
        out = news.copy().reset_index(drop=True)
        out[["tone", "tone_score", "signed"]] = tone[["label", "score", "signed"]].values
        if self.cfg.get("sentiment_check"):
            chk = self.tone(texts, check=True)
            out["tone_check"] = chk["label"].values
            out["tone_agree"] = out["tone"] == out["tone_check"]
        if self.driver_head:
            pred = pd.DataFrame(self._drivers(texts, batch_size=32))
            out["driver"], out["driver_sim"] = pred["label"].values, pred["score"].values
            out["driver_method"] = "fine-tuned"
        elif self.cfg.get("zero_shot"):
            label2name = {v: k for k, v in drivers.items()}
            res = self._zero_shot(texts, candidate_labels=list(drivers.values()),
                                  hypothesis_template="This news is about {}.", multi_label=False, batch_size=16)
            res = res if isinstance(res, list) else [res]
            out["driver"] = [label2name[r["labels"][0]] if r["scores"][0] >= threshold else "other" for r in res]
            out["driver_sim"] = [r["scores"][0] for r in res]
            out["driver_method"] = "zero-shot"
        else:
            names = list(drivers)
            sims = self.embed(texts) @ self.embed(list(drivers.values())).T
            out["driver"] = [names[i] for i in sims.argmax(1)]
            out["driver_sim"] = sims.max(1)
            out["driver_method"] = "similarity"
        return out

    @staticmethod
    def driver_summary(scored: pd.DataFrame) -> pd.DataFrame:
        if scored.empty:
            return pd.DataFrame(columns=["driver", "count", "net_tone", "share"])
        agg = {"count": ("signed", "size"), "net_tone": ("signed", "mean")}
        if "tone_agree" in scored:
            agg["model_agree"] = ("tone_agree", "mean")  # low agreement = ambiguous news, weight it less
        g = scored.groupby("driver").agg(**agg)
        g["share"] = g["count"] / g["count"].sum()
        return g.sort_values("count", ascending=False).reset_index()

    # ---------- long documents ----------
    def chunk(self, text: str, max_tokens: int) -> list[str]:
        """Split on paragraph boundaries into chunks of at most max_tokens (ModernBERT tokenizer)."""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.cfg["embedder"])
        chunks, cur, n = [], [], 0
        for para in [p for p in text.split("\n") if p.strip()]:
            k = len(tok.encode(para, add_special_tokens=False))
            if cur and n + k > max_tokens - 2:
                chunks.append("\n".join(cur))
                cur, n = [], 0
            cur.append(para)
            n += k
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def document_tone(self, text: str) -> dict:
        """Tone of a long document, length-weighted across chunks.

        Uses 8k-token chunks with a fine-tuned ModernBERT tone head, 512 with FinBERT.
        Also returns the most negative and most positive chunk openings as evidence.
        """
        if not text or not text.strip():
            return {"net_tone": np.nan, "chunks": 0, "method": "n/a"}
        limit = self.cfg.get("doc_chunk_tokens", 1024)
        chunks = self.chunk(text, limit)
        t = self.tone(chunks, batch_size=8)
        w = np.array([len(c) for c in chunks], dtype=float)
        t["text"] = chunks
        return {
            "net_tone": float(np.average(t["signed"], weights=w)),
            "pct_negative": float((t["label"] == "negative").mean()),
            "chunks": len(chunks),
            "method": f"{Path(str(self.tone_head or self.cfg['sentiment'])).name} @{limit}tok",
            "most_negative": t.nsmallest(1, "signed")["text"].iloc[0][:300],
            "most_positive": t.nlargest(1, "signed")["text"].iloc[0][:300],
        }
