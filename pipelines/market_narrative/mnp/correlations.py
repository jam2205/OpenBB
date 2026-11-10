"""Cross-asset returns, rolling correlations, and correlation-regime shifts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame, yield_cols: list[str] | None = None) -> pd.DataFrame:
    """Log returns for prices; first differences for yield series (bp moves, not % of a %)."""
    prices = prices.sort_index().ffill()
    yield_cols = [c for c in (yield_cols or []) if c in prices]
    r = np.log(prices.drop(columns=yield_cols)).diff()
    for c in yield_cols:
        r[c] = prices[c].diff()
    return r.dropna(how="all")


def snapshot(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    last = prices.ffill().iloc[-1]
    return pd.DataFrame({
        "last": last,
        "1w": prices.ffill().pct_change(5, fill_method=None).iloc[-1],
        "1m": prices.ffill().pct_change(21, fill_method=None).iloc[-1],
        "vol_20d_ann": rets.tail(20).std() * np.sqrt(252),
    })


def corr_shift(rets: pd.DataFrame, anchor: str, window: int) -> pd.DataFrame:
    """Recent vs. longer-run correlation of every asset to the anchor; flags regime breaks."""
    if anchor not in rets:
        return pd.DataFrame()
    recent = rets.tail(window).corr()[anchor]
    base = rets.corr()[anchor]
    df = pd.DataFrame({"corr_recent": recent, "corr_full": base}).drop(index=anchor)
    df["shift"] = df["corr_recent"] - df["corr_full"]
    df["flag"] = df["shift"].abs() > 0.3
    return df.sort_values("shift", key=abs, ascending=False)
