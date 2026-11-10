"""Data access: OpenBB Platform for everything it covers, MinIO for FX pairs it doesn't."""

from __future__ import annotations

import io
import logging
import os
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger(__name__)


def _obb():
    from openbb import obb  # imported lazily so the module loads without openbb installed

    obb.user.preferences.output_type = "dataframe"
    return obb


def _safe(fn, *a, **kw) -> pd.DataFrame:
    """Call an OpenBB endpoint; return empty frame on failure (missing key, provider outage)."""
    try:
        df = fn(*a, **kw)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        log.warning("OpenBB call %s failed: %s", getattr(fn, "__name__", fn), e)
        return pd.DataFrame()


def closes(symbols: list[str], start: date, provider: str, kind: str = "equity") -> pd.DataFrame:
    """Wide frame of close prices, one column per symbol."""
    obb = _obb()
    fn = {
        "equity": obb.equity.price.historical,
        "currency": obb.currency.price.historical,
        "index": obb.index.price.historical,
    }[kind]
    out = {}
    for s in symbols:
        df = _safe(fn, symbol=s, start_date=start, provider=provider)
        if not df.empty and "close" in df:
            out[s] = df["close"]
    wide = pd.DataFrame(out)
    wide.index = pd.to_datetime(wide.index)
    return wide


def ohlcv(symbol: str, start: date, provider: str) -> pd.DataFrame:
    """Full OHLCV for one symbol (lower-case columns, date index)."""
    df = _safe(_obb().equity.price.historical, symbol=symbol, start_date=start, provider=provider)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


def minio_closes(pairs: list[str], cfg: dict) -> pd.DataFrame:
    """Load FX closes for pairs not covered by OpenBB from MinIO (parquet or csv)."""
    if not pairs:
        return pd.DataFrame()
    from minio import Minio

    client = Minio(
        os.environ[cfg["endpoint_env"]],
        access_key=os.environ[cfg["access_key_env"]],
        secret_key=os.environ[cfg["secret_key_env"]],
        secure=cfg.get("secure", True),
    )
    out = {}
    for pair in pairs:
        key = cfg["object_pattern"].format(pair=pair)
        try:
            resp = client.get_object(cfg["bucket"], key)
            raw = io.BytesIO(resp.read())
            resp.close()
            resp.release_conn()
            df = pd.read_parquet(raw) if key.endswith(".parquet") else pd.read_csv(raw)
            df.columns = [c.lower() for c in df.columns]
            out[pair] = df.set_index(pd.to_datetime(df["date"]))["close"]
        except Exception as e:  # noqa: BLE001
            log.warning("MinIO load failed for %s (%s): %s", pair, key, e)
    return pd.DataFrame(out)


def news(symbols: list[str], provider: str, limit: int = 50) -> pd.DataFrame:
    obb = _obb()
    frames = [_safe(obb.news.company, symbol=s, provider=provider, limit=limit) for s in symbols]
    frames.append(_safe(obb.news.world, provider="benzinga", limit=limit))  # optional; empty w/o key
    df = pd.concat([f for f in frames if not f.empty], axis=0) if any(not f.empty for f in frames) else pd.DataFrame()
    if df.empty:
        return df
    df = df.reset_index().rename(columns={"index": "date"})
    text_col = "title" if "title" in df else df.columns[1]
    df["text"] = df[text_col].fillna("") + ". " + df.get("text", pd.Series("", index=df.index)).fillna("").astype(str).str[:500]
    return df.drop_duplicates(subset=[text_col])


def earnings_calendar(start: date, end: date, provider: str) -> pd.DataFrame:
    return _safe(_obb().equity.calendar.earnings, start_date=start, end_date=end, provider=provider)


def econ_calendar(start: date, end: date, provider: str) -> pd.DataFrame:
    return _safe(_obb().economy.calendar, start_date=start, end_date=end, provider=provider)


def fundamentals(symbol: str, provider: str = "yfinance") -> dict[str, pd.DataFrame]:
    obb = _obb()
    return {
        "metrics": _safe(obb.equity.fundamental.metrics, symbol=symbol, provider=provider),
        "income": _safe(obb.equity.fundamental.income, symbol=symbol, provider=provider, limit=4),
        "estimates": _safe(obb.equity.estimates.consensus, symbol=symbol, provider=provider),
    }


def documents(symbol: str, provider: str = "fmp") -> dict[str, str]:
    """Latest earnings-call transcript and MD&A text (empty strings if unavailable)."""
    obb = _obb()
    out = {}
    for name, fn in (("transcript", obb.equity.fundamental.transcript),
                     ("mdna", obb.equity.fundamental.management_discussion_analysis)):
        df = _safe(fn, symbol=symbol, provider=provider)
        out[name] = str(df["content"].iloc[-1]) if not df.empty and "content" in df else ""
    return out


def imf_search(query: str) -> pd.DataFrame:
    """Find IMF indicator symbols (no key). E.g. imf_search("exchange rate")."""
    return _safe(_obb().economy.available_indicators, provider="imf", query=query)


def imf_indicators(symbols: list[str], countries: list[str], start: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return _safe(_obb().economy.indicators, provider="imf", symbol=",".join(symbols),
                 country=",".join(countries), start_date=start)


def chokepoints(names: list[str], start: date) -> pd.DataFrame:
    """IMF PortWatch daily transit volumes through shipping chokepoints (oil / metals / geopolitics)."""
    if not names:
        return pd.DataFrame()
    return _safe(_obb().economy.shipping.chokepoint_volume, provider="imf",
                 chokepoint=",".join(names), start_date=start)


def window(today: date | None = None, ahead: int = 7) -> tuple[date, date]:
    today = today or date.today()
    return today, today + timedelta(days=ahead)
