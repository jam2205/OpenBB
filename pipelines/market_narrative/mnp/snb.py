"""Swiss National Bank data portal (https://data.snb.ch): public, no key.

API (verified live 2026-11-10):
    GET https://data.snb.ch/api/cube/<cube>/data/csv/en[?fromDate=YYYY-MM-DD&dimSel=D0(A,B)]
    GET https://data.snb.ch/api/cube/<cube>/dimensions/en        -> series codes and labels

Cubes used, with their update rhythm:
    snbgwdzid     daily      SNB policy rate, SARON, sight-deposit rates
    snbgwdchfsgw  daily      sight deposits at the SNB (CHF mn): the usual intervention proxy
    rendoeid      daily      Confederation bonds: yield to maturity + residual maturity per ISIN
    devwkieffid   daily      CHF effective exchange-rate indices (nominal / real)
    snbfxtr       quarterly  SNB foreign-exchange transactions (CHF mn; + = bought foreign currency)
    snbcurrc      quarterly  currency allocation of the SNB's reserves
    snboffzisa    monthly    official rates: SNB, Fed, ECB, BoE, BoJ

Do NOT use `rendoblid` / `rendopar` for current yields: both stopped at 2025-07-31.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
BASE = "https://data.snb.ch/api/cube"
# Oldest acceptable "latest observation" per rhythm, in days, before a series is flagged stale.
MAX_AGE = {"daily": 10, "weekly_batch": 21, "monthly": 75, "quarterly": 200}  # weekly_batch: daily data released once a week


def _to_date(s: pd.Series) -> pd.Series:
    """SNB dates come as YYYY-MM-DD, YYYY-MM, YYYY-Qn or YYYY. Periods map to their last day."""
    s = s.astype(str)
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    m = out.isna() & s.str.fullmatch(r"\d{4}-\d{2}")
    out[m] = pd.PeriodIndex(s[m], freq="M").to_timestamp(how="end").normalize()
    q = out.isna() & s.str.fullmatch(r"\d{4}-Q[1-4]")
    out[q] = pd.PeriodIndex(s[q].str.replace("-", ""), freq="Q").to_timestamp(how="end").normalize()
    y = out.isna() & s.str.fullmatch(r"\d{4}")
    out[y] = pd.to_datetime(s[y] + "-12-31")
    return out


def cube(cube_id: str, from_date: date | None = None, dim_sel: str | None = None,
         timeout: int = 40, retries: int = 2) -> pd.DataFrame:
    """One cube as a tidy frame: date, one column per dimension (lower-case), value."""
    import requests

    params = {}
    if from_date:
        params["fromDate"] = from_date.isoformat()
    if dim_sel:
        params["dimSel"] = dim_sel
    err = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{BASE}/{cube_id}/data/csv/en", params=params, timeout=timeout)
            if r.status_code == 200:
                break
            err = f"HTTP {r.status_code}"
            if r.status_code < 500 and r.status_code != 429:
                raise ValueError(err)
        except ValueError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"SNB cube {cube_id}: {err}")
    text = r.content.decode("utf-8-sig")
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"Date"')), None)
    if start is None:
        raise ValueError(f"SNB cube {cube_id}: no data header in response")
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=";", dtype=str)
    df.columns = [c.lower() for c in df.columns]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["date"] = _to_date(df["date"])
    df.attrs["published"] = next((ln.split(";")[1].strip('"') for ln in lines[:start] if "PublishingDate" in ln), None)
    return df.dropna(subset=["date"]).reset_index(drop=True)


def dimensions(cube_id: str) -> dict[str, str]:
    """Series code -> label for a cube (use it to explore other cubes)."""
    import requests

    data = requests.get(f"{BASE}/{cube_id}/dimensions/en", timeout=40).json()
    out: dict[str, str] = {}

    def walk(items):
        for it in items:
            out[it.get("id")] = it.get("name")
            walk(it.get("dimensionItems", []))

    for dim in data.get("dimensions", []):
        walk(dim.get("dimensionItems", []))
    return out


def _wide(df: pd.DataFrame, col: str = "d0") -> pd.DataFrame:
    return df.pivot_table(index="date", columns=col, values="value", aggfunc="last").sort_index()


def _stale(last: pd.Timestamp, rhythm: str, asof: date) -> bool:
    return (pd.Timestamp(asof) - last).days > MAX_AGE[rhythm]


def _asof(wide: pd.DataFrame, days: int) -> pd.Series:
    """Row as of `days` before the last observation (NaN row if history is too short)."""
    past = wide[wide.index <= wide.index[-1] - pd.Timedelta(days=days)]
    return past.iloc[-1] if not past.empty else pd.Series(np.nan, index=wide.columns)


def policy(asof: date) -> pd.DataFrame:
    """SNB policy rate, SARON and the sight-deposit rates, with 1m / 3m changes (pp)."""
    w = _wide(cube("snbgwdzid", asof - timedelta(days=200), "D0(LZ,SARON,ZIGBL,ZIG)")).ffill()
    names = {"LZ": "SNB policy rate", "SARON": "SARON", "ZIGBL": "Sight deposits up to threshold",
             "ZIG": "Sight deposits above threshold"}
    t = pd.DataFrame({"last_%": w.iloc[-1], "chg_1m_pp": w.iloc[-1] - _asof(w, 30), "chg_3m_pp": w.iloc[-1] - _asof(w, 91)})
    t.index = [names.get(i, i) for i in t.index]
    t.attrs.update(asof=w.index[-1].date(), stale=_stale(w.index[-1], "daily", asof))
    return t.round(3)


def sight_deposits(asof: date) -> pd.DataFrame:
    """Sight deposits at the SNB in CHF bn with 1w / 4w / 13w changes.

    A rise of several CHF bn in a week, with no other liquidity operation, is the market's usual
    read for SNB franc selling (it is a proxy: repo and bill operations move it too).
    """
    w = _wide(cube("snbgwdchfsgw", asof - timedelta(days=200))) / 1000.0
    w = w.dropna(how="all").ffill()
    names = {"TG": "Total sight deposits", "GI": "Domestic banks", "UEB": "Other sight deposits"}
    t = pd.DataFrame({"last_chf_bn": w.iloc[-1], "chg_1w": w.iloc[-1] - _asof(w, 7),
                      "chg_4w": w.iloc[-1] - _asof(w, 28), "chg_13w": w.iloc[-1] - _asof(w, 91)})
    t = t.reindex([c for c in ("TG", "GI", "UEB") if c in t.index])
    t.index = [names.get(i, i) for i in t.index]
    t.attrs.update(asof=w.index[-1].date(), stale=_stale(w.index[-1], "daily", asof))
    return t.round(2)


def fx_transactions(n: int = 6) -> pd.DataFrame:
    """SNB FX transactions per quarter, CHF bn. Positive = foreign currency bought (franc sold).
    Published with a lag of about one quarter.
    """
    df = cube("snbfxtr")
    df = df[df["d0"] == "T0"].sort_values("date").tail(n)
    t = pd.DataFrame({"quarter": df["date"].dt.to_period("Q").astype(str), "fx_purchases_chf_bn": (df["value"] / 1000).round(2)})
    return t.set_index("quarter")


def confederation_curve(asof: date, tenors: tuple[float, ...] = (2, 5, 10, 20, 30)) -> pd.DataFrame:
    """Confederation yield curve interpolated from individual bonds (yield vs residual maturity)."""
    df = cube("rendoeid", asof - timedelta(days=120))
    w = df.pivot_table(index=["date", "d0"], columns="d1", values="value").dropna(subset=["R1", "LZ"]).reset_index()
    days = sorted(w["date"].unique())
    if not days:
        return pd.DataFrame()

    def curve(day) -> pd.Series:
        g = w[w["date"] == day].sort_values("LZ")
        if len(g) < 3:
            return pd.Series(np.nan, index=tenors)
        # no extrapolation: tenors outside the range of outstanding bonds stay NaN
        return pd.Series([np.interp(t, g["LZ"], g["R1"]) if g["LZ"].min() <= t <= g["LZ"].max() else np.nan
                          for t in tenors], index=tenors)

    last = pd.Timestamp(days[-1])
    prior = [d for d in days if pd.Timestamp(d) <= last - pd.Timedelta(days=28)]
    now, then = curve(days[-1]), (curve(prior[-1]) if prior else pd.Series(np.nan, index=tenors))
    t = pd.DataFrame({"yield_%": now, "chg_1m_bp": (now - then) * 100})
    t.index = [f"{int(x)}Y" for x in tenors]
    t.attrs.update(asof=last.date(), stale=_stale(last, "weekly_batch", asof), bonds=int((w["date"] == days[-1]).sum()))
    return t.round(3).dropna(subset=["yield_%"])


def effective_fx(asof: date) -> pd.DataFrame:
    """CHF effective exchange-rate indices (up = stronger franc) with 1w / 1m / 3m % changes."""
    df = cube("devwkieffid", asof - timedelta(days=200))
    df = df[df["d2"] == "I"]
    df["series"] = df["d0"] + "_" + df["d1"]
    w = _wide(df, "series").ffill()
    names = {"N_G": "Nominal, overall", "N_E": "Nominal, vs euro area", "K_G": "Real (CPI), overall", "K_E": "Real (CPI), vs euro area"}
    w = w[[c for c in names if c in w]]
    t = pd.DataFrame({"index": w.iloc[-1], "chg_1w_%": (w.iloc[-1] / _asof(w, 7) - 1) * 100,
                      "chg_1m_%": (w.iloc[-1] / _asof(w, 30) - 1) * 100, "chg_3m_%": (w.iloc[-1] / _asof(w, 91) - 1) * 100})
    t.index = [names[i] for i in t.index]
    t.attrs.update(asof=w.index[-1].date(), stale=_stale(w.index[-1], "weekly_batch", asof))
    return t.round(2)


def reserves_allocation() -> pd.DataFrame:
    """Currency shares of the SNB's reserves, latest quarter vs a year earlier (percentage points)."""
    df = cube("snbcurrc")
    codes = {"ICHF0": "USD", "ICHF1": "EUR", "ICHF2": "JPY", "ICHF3": "GBP", "ICHF4": "CAD", "ICHF5": "Other", "ICHF6": "Total"}
    w = _wide(df[df["d0"].isin(codes)]).rename(columns=codes).dropna(subset=["Total"])
    share = w.drop(columns="Total").div(w["Total"], axis=0) * 100
    t = pd.DataFrame({"share_%": share.iloc[-1], "chg_1y_pp": share.iloc[-1] - share.iloc[-5] if len(share) > 4 else np.nan})
    t.index.name = None
    t.attrs.update(asof=str(w.index[-1].to_period("Q")), total_chf_bn=round(float(w["Total"].iloc[-1]) / 1000, 1))
    return t.round(1)


def global_policy_rates() -> pd.DataFrame:
    """Official rates published by the SNB for CH, US, euro area, UK and Japan (monthly, lagged)."""
    w = _wide(cube("snboffzisa", date.today() - timedelta(days=500))).ffill()
    last, prev = w.iloc[-1], _asof(w, 365)
    def mid(s, lo, hi): return (s.get(lo, np.nan) + s.get(hi, np.nan)) / 2
    rows = {"SNB policy rate": (last.get("LZ"), prev.get("LZ")), "Fed target (mid)": (mid(last, "UG1", "OG1"), mid(prev, "UG1", "OG1")),
            "ECB deposit facility": (last.get("EF"), prev.get("EF")), "BoE Bank rate": (last.get("L0"), prev.get("L0")),
            "BoJ call rate": (last.get("L1"), prev.get("L1"))}
    t = pd.DataFrame(rows, index=["rate_%", "year_ago"]).T
    t["chg_1y_pp"] = t["rate_%"] - t["year_ago"]
    t["vs_SNB_pp"] = t["rate_%"] - t.loc["SNB policy rate", "rate_%"]
    t.attrs.update(asof=str(w.index[-1].to_period("M")), stale=_stale(w.index[-1], "monthly", date.today()))
    return t.drop(columns="year_ago").round(2)


def block(asof: date) -> dict:
    """Everything for the report's Switzerland section. Each part is independent: a failure in
    one is logged and leaves that table empty.
    """
    parts = {"policy": lambda: policy(asof), "sight": lambda: sight_deposits(asof), "fx_tx": fx_transactions,
             "curve": lambda: confederation_curve(asof), "eff_fx": lambda: effective_fx(asof),
             "reserves": reserves_allocation, "global_rates": global_policy_rates}
    out, notes = {}, []
    for name, fn in parts.items():
        try:
            out[name] = fn()
            a = out[name].attrs
            if a.get("stale"):
                notes.append(f"{name}: latest observation {a.get('asof')} is older than expected")
        except Exception as e:  # noqa: BLE001
            log.warning("SNB %s unavailable: %s: %s", name, type(e).__name__, str(e)[:160])
            out[name] = pd.DataFrame()
    out["asof"] = {k: v.attrs.get("asof") for k, v in out.items() if isinstance(v, pd.DataFrame) and v.attrs.get("asof")}
    out["notes"] = notes
    out["reserves_total"] = out["reserves"].attrs.get("total_chf_bn") if not out["reserves"].empty else None
    return out
