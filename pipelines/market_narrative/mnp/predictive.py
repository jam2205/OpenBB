"""Adapter for AcyLa/multi-asset-predictive-model.

What the repo is: per-ETF LightGBM (+ optional LSTM) regressors predicting forward returns at
1/5/21 trading days from ~194 technical features. It ships its own feature code
(`features.py`), so we call its `AssetPredictor` and feed it OpenBB OHLCV.

Covered: SPY QQQ GLD SLV USO EEM EFA EWJ EWG EWU FXI INDA EWZ KWEB. Anything else is skipped.

Reading its published metrics (asset_results/*.json):
  * Sharpe is annualised with sqrt(252) over *overlapping* h-day returns, which inflates it by
    ~sqrt(h). We report sharpe/sqrt(h) as `sharpe_adj`.
  * Directional accuracy has no buy-and-hold baseline and comes from one 70/15/15 split with
    no purge gap, so treat it as an upper bound.
  * Signals are only labelled "usable" when test dir. accuracy >= SKILL_MIN; GLD/SLV/USO/INDA/KWEB
    fail that, so they show as "no skill" instead of BUY/SELL.

Security: the model files are pickles (joblib / torch.load); loading them runs code from that
repo. Only point this at a repo revision you trust; pin `revision` once you've reviewed it.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SKILL_MIN = 0.55
COVERED = {"SPY", "QQQ", "GLD", "SLV", "USO", "EEM", "EFA", "EWJ", "EWG", "EWU", "FXI", "INDA", "EWZ", "KWEB"}


@contextlib.contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class AcyLaModel:
    def __init__(self, repo_id: str, revision: str | None = None, use_lstm: bool = False):
        from huggingface_hub import snapshot_download

        self.path = Path(snapshot_download(repo_id, revision=revision))
        self.use_lstm = use_lstm
        if str(self.path) not in sys.path:
            sys.path.insert(0, str(self.path))
        self.cache = Path(os.environ.get("MNP_CACHE", Path.home() / ".cache" / "mnp_acyla"))
        self.cache.mkdir(parents=True, exist_ok=True)
        with _cwd(self.cache):  # its data_loader creates a cache dir in the cwd at import
            import config as acyla_cfg  # noqa: PLC0415  (module from the downloaded repo)
            import predict as acyla_predict  # noqa: PLC0415

        acyla_cfg.cfg.model_dir = str(self.path / "asset_models")
        self._predict_mod = acyla_predict

    def skill(self, ticker: str) -> pd.DataFrame:
        f = self.path / "asset_results" / f"{ticker}_results.json"
        if not f.exists():
            return pd.DataFrame()
        df = pd.DataFrame(json.loads(f.read_text())["quick_eval"]).set_index("horizon")
        df["sharpe_adj"] = df["sharpe"] / np.sqrt(df.index.to_numpy())
        return df[["dir_acc", "sharpe_adj"]]

    @staticmethod
    def to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """OpenBB lower-case columns -> the Open/High/Low/Close/Volume frame the repo expects."""
        cols = {c: c.capitalize() for c in ("open", "high", "low", "close", "volume")}
        out = df.rename(columns=cols)[list(cols.values())].astype(float).dropna()
        out.index = pd.to_datetime(out.index)
        return out

    def predict(self, ticker: str, ohlcv: pd.DataFrame | None = None) -> pd.DataFrame:
        """One row per horizon: prediction, raw signal, test-set skill, and the signal we'd show."""
        if ticker not in COVERED:
            return pd.DataFrame()
        with _cwd(self.cache), contextlib.redirect_stdout(open(os.devnull, "w")):  # noqa: SIM115
            res = self._predict_mod.AssetPredictor(ticker).get_multi_horizon(
                df=self.to_ohlcv(ohlcv) if ohlcv is not None else None, use_lstm=self.use_lstm)
        sk = self.skill(ticker)
        rows = []
        for h in (1, 5, 21):
            r = res.get(h, {})
            if "signal" not in r:
                continue
            acc = sk["dir_acc"].get(h, np.nan) if not sk.empty else np.nan
            usable = bool(acc >= SKILL_MIN)
            rows.append({
                "ticker": ticker, "horizon_d": h,
                "pred_return": r["ensemble_prediction"], "model_signal": r["signal"],
                "test_dir_acc": acc, "sharpe_adj": sk["sharpe_adj"].get(h, np.nan) if not sk.empty else np.nan,
                "shown_signal": r["signal"] if usable else "no skill",
            })
        return pd.DataFrame(rows)

    def predict_many(self, ohlcv_by_ticker: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
        frames = []
        for t, df in ohlcv_by_ticker.items():
            try:
                frames.append(self.predict(t, df))
            except Exception as e:  # noqa: BLE001
                log.warning("AcyLa predict failed for %s: %s", t, e)
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
