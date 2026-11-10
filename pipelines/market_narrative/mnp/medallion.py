"""Bridge to your own private data store (for example a lakehouse snapshot on object storage).

The pipeline never opens its own connection and holds no addresses, bucket names or dataset
names. Your notebook's connector cell provides `read(name, ...)` and `tables()`; you then tell
the pipeline which of your datasets plays which role:

    from mnp import medallion
    medallion.register(read, tables)             # roles are matched to your dataset names by pattern
    medallion.register(read, tables, datasets={  # ... or name them yourself (overrides the match)
        "calendar": "<your economic-calendar dataset>",
        "carry":    "<your FX carry dataset>",
        "candles":  "<your hourly FX candles dataset>",
        "yields":   "<your government-bond yields dataset>",
        "positioning": "<your per-pair COT positioning dataset>",
    })

Expected columns:
  calendar     date (or release_ts_utc), title, currency, impact
               [, source, actual, forecast, previous, event, family, event_p]
  carry        pair, date, carry [, carry_z*, *momentum*]
  candles      time, pair, close
  yields       date, <cc>_10y [, <cc>_2y]     one row per day, cc = us, de, jp, gb, ch ...
  positioning  pair, report_date, institutional_bias [, bias_direction, is_current, report_age_days]
  range_forecast  pair, anchor_time, window_end, forecast_range [, adr20]   (pair briefs)
  regime       pair, state, timestamp [, confidence, vocabulary]            (pair briefs)
  candles      also open, high, low for the pair briefs' day and week profile

Nothing is downloaded at registration: a dataset is read the first time a section needs it.

Calendar handling:
  * rows whose `source` does not start with "forex" are dropped (observation dates, not releases)
  * impact HOLIDAY and family "calendar" are dropped
  * several rows for one decision (rate, statement, votes) count once
  * a label with `event_p` below 0.7 is ignored; the title is kept
"""

from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)
_read = None
_tables = None
_datasets: dict[str, str] = {}


# role -> name patterns, most specific first. The shortest matching dataset name wins.
_ROLE_PATTERNS = {
    "calendar": [r"^calendar.*event", r"^econ.*calendar", r"calendar"],
    "carry": [r"^carry", r"carry"],
    "candles": [r"^candles?_?1h$", r"^candles?.*1h"],
    "yields": [r"bond.*yield", r"^yields?"],
    "positioning": [r"cot.*pair", r"pair.*(cot|positioning)"],
    "range_forecast": [r"^range.*forecast", r"range.*forecast"],
    "regime": [r"markov.*regime", r"^regime.*log"],
}


def match_roles(names: list[str]) -> dict[str, str]:
    """Pick a dataset for each role from a list of dataset names, by name pattern only."""
    out = {}
    for role_, patterns in _ROLE_PATTERNS.items():
        for pat in patterns:
            hits = sorted((n for n in names if re.search(pat, n, re.I)), key=lambda n: (len(n), n))
            if hits:
                out[role_] = hits[0]
                break
    return out


def register(read, tables=None, datasets: dict[str, str] | None = None) -> dict[str, str]:
    """Hand the pipeline your notebook's `read(name, ...)` and `tables()`.

    Roles (calendar, carry, candles, yields, positioning) are matched to your dataset names by
    pattern; `datasets` overrides any of them, and a role set to "" or None is switched off.
    Returns the role -> dataset map in use.
    """
    global _read, _tables, _datasets
    _read, _tables = read, tables
    found = {}
    if tables is not None:
        try:
            found = match_roles(list(tables()))
        except Exception as e:  # noqa: BLE001
            log.warning("could not list your datasets to match roles: %s: %s", type(e).__name__, str(e)[:160])
    _datasets = {k: v for k, v in {**found, **(datasets or {})}.items() if v}
    log.info("private data reader registered; roles: %s", sorted(_datasets))
    return dict(_datasets)


def roles() -> dict[str, str]:
    return dict(_datasets)


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
    family, forecast, previous. Empty if the reader is not registered.
    """
    df = dataset(role("calendar"))
    if not df.empty and "date" not in df and "release_ts_utc" in df:
        df = df.rename(columns={"release_ts_utc": "date"})
    if df.empty or not {"date", "title", "currency"} <= set(df.columns):
        if not df.empty:
            log.warning("your calendar dataset lacks date / title / currency columns, so it is not used")
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df["date"] >= lo) & (df["date"] < hi)]
    if "source" in df and df["source"].astype(str).str.lower().str.startswith("forex").any():
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
        "forecast": df.get("forecast"), "previous": df.get("previous"), "actual": df.get("actual"),
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


def global_yields(asof: date | None = None, base: str = "us", max_age_days: int = 10) -> pd.DataFrame:
    """10Y government yields by country from the `yields` role: level, 1-week and 1-month change
    in bp, spread to the base country, and each country's own 2s10s where a 2Y column exists.

    Countries whose last value is more than `max_age_days` older than the newest row are left
    out, so a series that stopped updating cannot pass as current. `attrs["asof"]` is the date
    of the newest row.
    """
    df = dataset(role("yields"))
    if df.empty or "date" not in df:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date").sort_index()
    if asof is not None:
        df = df[df.index <= pd.Timestamp(asof) + pd.Timedelta(days=1)]
    tens = {m.group(1): c for c in df.columns if (m := re.fullmatch(r"([a-z]{2})_10y", str(c)))}
    if df.empty or not tens:
        return pd.DataFrame()
    newest = df.index[-1]

    def at(s: pd.Series, days: int):
        past = s[s.index <= s.index[-1] - pd.Timedelta(days=days)]
        return past.iloc[-1] if len(past) else float("nan")

    rows = {}
    for cc, col in tens.items():
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty or (newest - s.index[-1]).days > max_age_days:
            continue
        row = {"10y_%": s.iloc[-1], "chg_1w_bp": (s.iloc[-1] - at(s, 7)) * 100, "chg_1m_bp": (s.iloc[-1] - at(s, 30)) * 100}
        two = pd.to_numeric(df[f"{cc}_2y"], errors="coerce").dropna() if f"{cc}_2y" in df else pd.Series(dtype=float)
        row["2s10s_bp"] = (s.iloc[-1] - two.iloc[-1]) * 100 if len(two) and (newest - two.index[-1]).days <= max_age_days else float("nan")
        row["as_of"] = s.index[-1].date().isoformat()
        rows[cc.upper()] = row
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).T
    num = ["10y_%", "chg_1w_bp", "chg_1m_bp", "2s10s_bp"]
    t[num] = t[num].astype(float)
    if base.upper() in t.index:
        t.insert(3, f"vs_{base.upper()}_bp", (t["10y_%"] - t.loc[base.upper(), "10y_%"]) * 100)
        t = pd.concat([t.loc[[base.upper()]], t.drop(index=base.upper()).sort_values("10y_%", ascending=False)])
    t = t.round({"10y_%": 2, "chg_1w_bp": 0, "chg_1m_bp": 0, "2s10s_bp": 0, f"vs_{base.upper()}_bp": 0})
    if t["2s10s_bp"].isna().all():
        t = t.drop(columns="2s10s_bp")
    t.attrs["asof"] = newest.date()
    if asof is not None and (pd.Timestamp(asof) - newest).days > max_age_days:
        log.warning("your yields dataset ends %s: the global yields table is %d days old",
                    newest.date(), (pd.Timestamp(asof) - newest).days)
    return t


def positioning(pairs: list[str] | None = None, top: int = 6) -> pd.DataFrame:
    """Latest COT-derived positioning per FX pair from the `positioning` role. Shows `pairs`
    when given and present, otherwise the `top` pairs with the largest absolute bias.
    """
    df = dataset(role("positioning"))
    if df.empty or not {"pair", "report_date"} <= set(df.columns):
        return pd.DataFrame()
    df = df.sort_values("report_date").groupby("pair").tail(1)
    if "is_complete" in df:
        df = df[df["is_complete"].astype(bool)]
    bias = next((c for c in ("institutional_bias", "bias", "net_bias") if c in df), None)
    sel = df[df["pair"].isin(pairs)] if pairs else df.iloc[0:0]
    if sel.empty and bias:
        sel = df.reindex(df[bias].abs().sort_values(ascending=False).index).head(top)
    if sel.empty:
        return pd.DataFrame()
    keep = [c for c in ("pair", bias, "bias_direction", "base_index", "quote_index", "report_age_days", "report_date") if c and c in sel]
    t = sel[keep].copy()
    t["report_date"] = pd.to_datetime(t["report_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    for c in t.select_dtypes("float").columns:
        t[c] = t[c].round(2)
    return t.set_index("pair")


def _filtered(name: str | None, pair: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One pair's rows. Readers differ in what they accept, so try a pushed-down filter first
    and fall back to reading the dataset and filtering here."""
    if _read is None or not name:
        return pd.DataFrame()
    for kw in ({"columns": columns, "filters": [("pair", "==", pair)]}, {}):
        try:
            df = _read(name, **{k: v for k, v in kw.items() if v is not None})
        except TypeError:
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("private dataset unavailable: %s: %s", type(e).__name__, str(e)[:200])
            return pd.DataFrame()
        if isinstance(df, pd.DataFrame) and "pair" in df:
            return df[df["pair"] == pair]
    return pd.DataFrame()


def hourly(pair: str, start: date | None = None) -> pd.DataFrame:
    """Hourly bars for one pair from the candles role: time (UTC), open, high, low, close."""
    df = _filtered(role("candles"), pair)
    if df.empty or not {"time", "high", "low", "close"} <= set(df.columns):
        return pd.DataFrame()
    df = df[[c for c in ("time", "open", "high", "low", "close") if c in df]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if start is not None:
        df = df[df["time"] >= pd.Timestamp(start, tz="UTC")]
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def range_forecast(pair: str, now: pd.Timestamp) -> dict:
    """The stored 24h range forecast whose window contains `now`, if there is one."""
    df = _filtered(role("range_forecast"), pair)
    if df.empty or not {"anchor_time", "window_end", "forecast_range"} <= set(df.columns):
        return {}
    a, e = pd.to_datetime(df["anchor_time"], utc=True), pd.to_datetime(df["window_end"], utc=True)
    hit = df[(a <= now) & (now < e)]
    if hit.empty:
        return {}
    r = hit.iloc[-1]
    return {"forecast_range": float(r["forecast_range"]), "anchor_time": pd.Timestamp(r["anchor_time"]),
            "adr20": float(r["adr20"]) if "adr20" in r and pd.notna(r["adr20"]) else None}


def regime(pair: str, now: pd.Timestamp, max_age_h: float = 48) -> pd.DataFrame:
    """Latest regime label per vocabulary (two label sets share the log; they are never pooled)."""
    df = _filtered(role("regime"), pair)
    if df.empty or not {"state", "timestamp"} <= set(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"] <= now].sort_values("timestamp")
    if "vocabulary" not in df:
        df["vocabulary"] = "unlabelled"
    last = df.groupby(df["vocabulary"].fillna("unlabelled")).tail(1)
    out = pd.DataFrame({"state": last["state"].to_numpy(), "confidence": last.get("confidence", pd.Series(index=last.index, dtype=float)).to_numpy(),
                        "age_h": ((now - last["timestamp"]).dt.total_seconds() / 3600).round(1).to_numpy()},
                       index=pd.Index(last["vocabulary"].fillna("unlabelled").to_numpy(), name="vocabulary"))
    out["stale"] = out["age_h"] > max_age_h
    return out
