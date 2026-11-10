"""Bridge to your own private data store (for example a lakehouse snapshot on object storage).

The pipeline never opens its own connection and holds no addresses, bucket names or dataset
names. Your notebook's connector cell provides `read(name, ...)` and `tables()`; you then tell
the pipeline which of your datasets plays which role:

    from mnp import medallion
    medallion.register(read, tables, datasets={
        "calendar": "<your economic-calendar dataset>",
        "carry":    "<your FX carry dataset>",          # optional
        "candles":  "<your hourly FX candles dataset>",  # optional
    })

Expected columns:
  calendar  date, title, currency, impact [, source, forecast, previous, event, family, event_p]
  carry     pair, date, carry [, carry_z*, *momentum*]
  candles   time, pair, close

Calendar handling:
  * rows whose `source` does not start with "forex" are dropped (observation dates, not releases)
  * impact HOLIDAY and family "calendar" are dropped
  * several rows for one decision (rate, statement, votes) count once
  * a label with `event_p` below 0.7 is ignored; the title is kept
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)
_read = None
_tables = None
_datasets: dict[str, str] = {}


def register(read, tables=None, datasets: dict[str, str] | None = None) -> None:
    """Hand the pipeline your notebook's `read(name, ...)`, optional `tables()`, and a
    role -> dataset-name map (roles: calendar, carry, candles)."""
    global _read, _tables, _datasets
    _read, _tables, _datasets = read, tables, dict(datasets or {})
    log.info("private data reader registered; roles: %s", sorted(_datasets))


def role(name: str) -> str | None:
    return _datasets.get(name)


def available() -> bool:
    return _read is not None


def tables() -> list[str]:
    return list(_tables()) if _tables else []


def dataset(name: str | None, **kw) -> pd.DataFrame:
    if _read is None or not name:
        return pd.DataFrame()
    try:
        return _read(name, **kw)
    except Exception as e:  # noqa: BLE001
        log.warning("private dataset unavailable: %s: %s", type(e).__name__, str(e)[:200])
        return pd.DataFrame()


def calendar(start: date, end: date, currencies: list[str] | None = None,
             impacts: tuple[str, ...] = ("high",), min_label_p: float = 0.7) -> pd.DataFrame:
    """Scheduled releases in [start, end]: date, event (title), currency, impact, event_code,
    family, forecast, previous. Empty if the reader is not registered."""
    df = dataset(role("calendar"))
    if df.empty or not {"date", "title", "currency"} <= set(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["date"] >= lo) & (df["date"] < hi)]
    if "source" in df:
        df = df[df["source"].astype(str).str.lower().str.startswith("forex")]
    if "impact" in df:
        df = df[df["impact"].astype(str).str.lower().isin([i.lower() for i in impacts])]
    if "family" in df:
        df = df[df["family"].astype(str) != "calendar"]
    if currencies:
        df = df[df["currency"].isin(currencies)]
    if df.empty:
        return pd.DataFrame()
    code = df["event"] if "event" in df else pd.Series(None, index=df.index, dtype=object)
    fam = df["family"] if "family" in df else pd.Series(None, index=df.index, dtype=object)
    if "event_p" in df:
        unsure = (df["event_p"].fillna(0) < min_label_p).to_numpy(dtype=bool)
        code, fam = code.mask(unsure), fam.mask(unsure)
    out = pd.DataFrame({
        "date": df["date"], "currency": df["currency"], "event": df["title"].astype(str),
        "impact": df.get("impact"), "event_code": code, "family": fam,
        "forecast": df.get("forecast"), "previous": df.get("previous"),
    })
    # one decision = several rows (rate, statement, votes): keep one per currency/timestamp/code
    labelled = out["event_code"].notna().to_numpy(dtype=bool)
    out = pd.concat([out[labelled].drop_duplicates(["date", "currency", "event_code"]), out[~labelled]])
    return out.sort_values(["date", "currency"]).reset_index(drop=True)


def fx_daily_closes(pairs: list[str], start: date) -> pd.DataFrame:
    """Daily closes for FX pairs from the hourly candles role (last close of each UTC day)."""
    if not pairs or _read is None:
        return pd.DataFrame()
    df = dataset(role("candles"), columns=["time", "pair", "close"], filters=[("pair", "in", list(pairs))])
    if df.empty:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    wide = df.pivot_table(index="time", columns="pair", values="close", aggfunc="last").sort_index()
    daily = wide.resample("1D").last().dropna(how="all")
    daily.index = daily.index.tz_localize(None)
    return daily


def carry(pairs: list[str] | None = None) -> pd.DataFrame:
    """Latest policy-rate carry row per pair (base minus quote), with its z-scores."""
    df = dataset(role("carry"))
    if df.empty or "pair" not in df or "date" not in df:
        return pd.DataFrame()
    df = df.sort_values("date").groupby("pair").tail(1)
    if pairs:
        df = df[df["pair"].isin(pairs)]
    keep = [c for c in df.columns if c in ("pair", "date", "carry") or c.startswith("carry_z") or "momentum" in c]
    return df[keep].set_index("pair").sort_index()
