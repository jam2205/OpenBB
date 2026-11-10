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


def snapshot(prices: pd.DataFrame, rets: pd.DataFrame, yield_cols: list[str] | None = None) -> pd.DataFrame:
    """Level, 1w / 1m change and 20d annualised vol. Yield rows show changes in bp, vol in bp/yr."""
    px = prices.ffill()
    df = pd.DataFrame({
        "last": px.iloc[-1],
        "chg_1w": px.pct_change(5, fill_method=None).iloc[-1],
        "chg_1m": px.pct_change(21, fill_method=None).iloc[-1],
        "vol_20d_ann": rets.tail(20).std() * np.sqrt(252),
    })
    df["unit"] = "%"
    for c in [c for c in (yield_cols or []) if c in px]:
        df.loc[c, ["chg_1w", "chg_1m"]] = [px[c].diff(5).iloc[-1] * 100, px[c].diff(21).iloc[-1] * 100]
        df.loc[c, "vol_20d_ann"] *= 100
        df.loc[c, "unit"] = "bp"
    pct = df["unit"] == "%"
    df.loc[pct, ["chg_1w", "chg_1m", "vol_20d_ann"]] *= 100
    return df.round(2)


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


def pair_corr(rets: pd.DataFrame, a: str | None, b: str | None, window: int) -> tuple[float, float] | None:
    """(recent, full-sample) correlation of two return series; None if either is missing."""
    if not a or not b or a not in rets or b not in rets:
        return None
    j = rets[[a, b]].dropna()
    if len(j) <= window:
        return None
    return float(j.tail(window).corr().iloc[0, 1]), float(j.corr().iloc[0, 1])


def up_share(close: pd.Series, horizon: int) -> float:
    """Share of h-day periods that ended higher: what "always predict up" scores on direction."""
    r = close.dropna().pct_change(horizon).dropna()
    return float((r > 0).mean()) if len(r) else float("nan")
