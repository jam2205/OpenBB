"""Per-pair brief: where the pair stands in its day and week, what is scheduled for its two
currencies, and the fundamentals behind it. One markdown brief and one machine-readable row
per pair, built from your own data store (medallion roles) only.

    from mnp import pairs
    paths = pairs.run()                    # every pair in config.yaml -> pairs.list
    b = pairs.brief("EURUSD")              # the same content as a dict

The brief never calls a direction. The measured part is the range (how much of it is used, how
often each extreme held from a state like this one); the calendar, yields, carry, positioning
and regime label are shown beside it as context for a setup you already have.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import medallion, profile, report

log = logging.getLogger(__name__)
_YIELD_CC = {"USD": "US", "EUR": "DE", "JPY": "JP", "GBP": "GB", "CHF": "CH", "AUD": "AU", "CAD": "CA", "NZD": "NZ"}


def _template(p: dict, events_today: pd.DataFrame) -> str:
    """Which "cookie cutter" this pair gets right now. First match wins."""
    if p.get("hours_left", 1) <= 0:
        return "day_closed"             # the last hour is in: the extremes are final by definition
    if not events_today.empty:
        return "event_ahead"            # a high-impact release before the roll: ranges break more often
    if p.get("used_of_expected", 0) >= 1.0:
        return "range_spent"            # the expected range is already travelled
    if p.get("hours_elapsed", 0) <= 9:
        return "early_session"          # Asia only: most days set at least one extreme later
    return "range_open"


def _yield_gap(base: str, quote: str, yields: pd.DataFrame) -> dict:
    b, q = _YIELD_CC.get(base), _YIELD_CC.get(quote)
    if yields.empty or b not in yields.index or q not in yields.index:
        return {}
    gap = (yields.loc[b, "10y_%"] - yields.loc[q, "10y_%"]) * 100
    out = {"gap_bp": float(gap)}
    for c in ("chg_1w_bp", "chg_1m_bp"):
        if c in yields:
            out[c.replace("chg", "gap_chg")] = float(yields.loc[b, c] - yields.loc[q, c])
    return out


def _rest_hourly(pair: str, start: pd.Timestamp) -> pd.DataFrame:
    """Recent hourly bars for one pair from a public REST source (yfinance through OpenBB)."""
    from . import sources

    df = sources._safe(sources._obb().currency.price.historical, symbol=pair, interval="1h",
                       start_date=start.date(), provider="yfinance")
    if df.empty or not {"high", "low", "close"} <= set(df.columns):
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"date": "time", "index": "time"})
    return df[[c for c in ("time", "open", "high", "low", "close") if c in df]]


def _aligned(main: pd.DataFrame, extra: pd.DataFrame, tol: float = 0.0015, min_overlap: int = 12) -> pd.DataFrame:
    """`extra` with its timestamps proven against `main`, or an empty frame.

    Another feed may stamp bars in another timezone or at the bar's close. Each candidate
    reading of its clock is tried, and one is accepted only if, on the hours both feeds have,
    the closes agree (median gap under 0.15%) and no other reading agrees better. Without that
    proof the bars are not used: a shifted bar would move the day's high and low."""
    raw = pd.to_datetime(extra["time"], errors="coerce")
    best, best_gap = pd.DataFrame(), np.inf
    for tz in ("UTC", "America/New_York", "Europe/London"):
        for shift_h in (0, -1):
            try:
                t = raw.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") if raw.dt.tz is None else raw
            except (TypeError, ValueError):
                continue
            t = t.dt.tz_convert("UTC").dt.floor("h") + pd.Timedelta(hours=shift_h)
            cand = extra.assign(time=t).dropna(subset=["time", "high", "low", "close"]).drop_duplicates("time", keep="last")
            j = cand.merge(main[["time", "close"]], on="time", suffixes=("", "_main"))
            if len(j) < min_overlap:
                continue
            gap = float((j["close"] / j["close_main"] - 1).abs().median())
            if gap < best_gap:
                best, best_gap = cand, gap
            if raw.dt.tz is not None and shift_h == -1:
                break
        if raw.dt.tz is not None:
            break
    return best if best_gap < tol else pd.DataFrame()


def top_up(pair: str, bars: pd.DataFrame, max_age_h: float = 3.0, sources_order: tuple[str, ...] = ("store", "rest")) -> tuple[pd.DataFrame, str | None]:
    """If the pair's bars have fallen behind, append newer hourly bars from another source.

    Order: your other hourly candle datasets ("store"), then a public REST feed ("rest"). Bars
    are appended only after `_aligned` has proven the other feed's clock against the main one,
    and only hours that have closed. Returns (bars, note); note names the source and how many
    bars it supplied, or says why nothing could be added. History is never replaced."""
    now = pd.Timestamp.now(tz="UTC")
    last = bars["time"].iloc[-1]
    if (now - last).total_seconds() / 3600 - 1 <= max_age_h or now.weekday() >= 5:
        return bars, None
    tried = []
    for src in sources_order:
        cands = medallion.other_hourly(pair, last) if src == "store" else [("yfinance (REST)", _rest_hourly(pair, last - pd.Timedelta(days=5)))]
        for name, extra in cands:
            if extra.empty:
                tried.append(f"{name}: nothing returned")
                continue
            ok = _aligned(bars.tail(24 * 30), extra)
            new = ok[(ok["time"] > last) & (ok["time"] + pd.Timedelta(hours=1) <= now)] if not ok.empty else ok
            if new.empty:
                tried.append(f"{name}: {'no newer closed bars' if not ok.empty else 'could not be matched to the main feed'}")
                continue
            if "open" not in new:
                new = new.assign(open=new["close"].shift(1).fillna(bars["close"].iloc[-1]))
            out = pd.concat([bars, new[[c for c in bars.columns if c in new]]], ignore_index=True).sort_values("time").reset_index(drop=True)
            return out, f"main candles stopped at {last:%a %d %b %H:%M} UTC; the last {len(new)} hourly bars come from {name}"
    return bars, f"main candles stopped at {last:%a %d %b %H:%M} UTC and no other source could fill the gap ({'; '.join(tried) or 'none available'})"


def brief(pair: str, now: pd.Timestamp | None = None, cfg: dict | None = None, shared: dict | None = None) -> dict:
    """Everything the pair template needs. `shared` carries tables that are the same for every
    pair (yields, calendar) so a run over many pairs reads them once."""
    pair = pair.upper().replace("/", "")
    base, quote = pair[:3], pair[3:]
    pcfg = (cfg or {}).get("pairs", {})
    bars = medallion.hourly(pair, date.today() - timedelta(days=int(pcfg.get("history_days", 3650))))
    if bars.empty:
        return {"pair": pair, "error": "no hourly bars for this pair in your candles dataset"}
    note = None
    if now is None and pcfg.get("top_up", True):   # a replay (`now` given) reads history only
        try:
            bars, note = top_up(pair, bars, float(pcfg.get("max_age_h", 3)), tuple(pcfg.get("top_up_sources", ["store", "rest"])))
        except Exception as e:  # noqa: BLE001  a failed top-up leaves the brief as it was, flagged stale
            note = f"top-up failed: {type(e).__name__}: {str(e)[:120]}"
        if note:
            log.warning("%s: %s", pair, note)
    now = pd.Timestamp(now).tz_convert("UTC") if now is not None else bars["time"].iloc[-1] + pd.Timedelta(hours=1)
    fc = medallion.range_forecast(pair, now)
    p = profile.day_profile(bars, now, fc.get("forecast_range"))
    if not p:
        return {"pair": pair, "error": "too little history for a day profile (needs about 120 complete days)"}
    w = profile.week_profile(bars, now)
    age_h = (pd.Timestamp.now(tz="UTC") - p["asof"]).total_seconds() / 3600

    shared = shared if shared is not None else {}
    if "yields" not in shared:
        shared["yields"] = medallion.global_yields(now.date())
    if "calendar" not in shared:
        c = (cfg or {}).get("calendar", {})
        shared["calendar"] = medallion.calendar(now.date(), now.date() + timedelta(days=7), None, tuple(c.get("impacts", ["high"])))
    cal = shared["calendar"]
    cal = cal[cal["currency"].isin([base, quote]) & (cal["date"] >= now)] if not cal.empty else cal
    roll = p["asof"] + pd.Timedelta(hours=p["hours_left"])
    today = cal[cal["date"] < roll] if not cal.empty else cal
    cal_tbl = pd.DataFrame()
    if not cal.empty:
        cal_tbl = pd.DataFrame({"when (UTC)": cal["date"].dt.strftime("%a %d %b %H:%M"), "currency": cal["currency"], "event": cal["event"],
                                "forecast": cal.get("forecast"), "previous": cal.get("previous")}).head(12).fillna("").astype(str).reset_index(drop=True)
    carry = medallion.carry([pair])
    cot = medallion.positioning([pair])
    cot = cot if not cot.empty and pair in cot.index else pd.DataFrame()
    reg = medallion.regime(pair, now)
    lv, hv = profile.verdict(p["low_hold"]), profile.verdict(p["high_hold"])
    if p["hours_left"] <= 0:            # a 100% "hold rate" here is arithmetic, not information
        lv = hv = "final (day closed)"
        for sd in (p["low_hold"], p["high_hold"]):
            sd["tilt"] = sd["vs_null"] = sd["vs_drift"] = "n/a (day closed)"
    return {
        "pair": pair, "base": base, "quote": quote, "bars_note": note, "asof": p["asof"], "data_age_h": round(age_h, 1), "stale": age_h > float(pcfg.get("max_age_h", 3)),
        "template": _template(p, today), "day": p, "week": w, "low_verdict": lv, "high_verdict": hv, "forecast": fc,
        "events_today": len(today), "calendar": cal_tbl, "yield_gap": _yield_gap(base, quote, shared["yields"]),
        "carry": carry, "positioning": cot, "regime": reg,
    }


def row(b: dict) -> dict:
    """One flat, JSON-safe record per pair: the numbers an agent or a job would filter on."""
    if "error" in b:
        return {"pair": b["pair"], "error": b["error"]}
    p, lo, hi, w = b["day"], b["day"]["low_hold"], b["day"]["high_hold"], b.get("week") or {}
    out = {
        "pair": b["pair"], "ts": b["asof"].isoformat(), "source": "mnp_pair_brief", "template": b["template"], "data_age_h": b["data_age_h"], "bars_note": b.get("bars_note"),
        "stale": bool(b["stale"]), "hours_into_day": p["hours_elapsed"], "last": p["last"], "day_low": p["low"], "day_high": p["high"],
        "expected_range": p["expected_range"], "expected_from": p["expected_from"], "used_of_expected": p["used_of_expected"],
        "p_low_in": lo["p"], "p_low_in_baseline": lo["baseline"], "p_low_in_random_walk": lo.get("null"), "low_vs_random_walk": lo["vs_null"],
        "p_low_in_drifting_walk": lo.get("drift"), "low_tilt": lo["tilt"],
        "low_n_days": lo["n_days"], "low_n_weeks": lo["n_weeks"], "low_verdict": b["low_verdict"],
        "p_high_in": hi["p"], "p_high_in_baseline": hi["baseline"], "p_high_in_random_walk": hi.get("null"), "high_vs_random_walk": hi["vs_null"],
        "p_high_in_drifting_walk": hi.get("drift"), "high_tilt": hi["tilt"],
        "high_n_days": hi["n_days"], "high_n_weeks": hi["n_weeks"], "high_verdict": b["high_verdict"],
        "high_if_low_holds": p["high_if_low_holds"], "low_if_high_holds": p["low_if_high_holds"], "events_before_roll": b["events_today"],
        "week_low": (w.get("low") or {}).get("level"), "p_week_low_in": (w.get("low") or {}).get("p"),
        "week_high": (w.get("high") or {}).get("level"), "p_week_high_in": (w.get("high") or {}).get("p"),
        "yield_gap_bp": b["yield_gap"].get("gap_bp"), "yield_gap_chg_1w_bp": b["yield_gap"].get("gap_chg_1w_bp"),
        "regime": "; ".join(f"{v}: {s}" for v, s in b["regime"]["state"].items()) if not b["regime"].empty else None,
        "method": "conditional frequency from the pair's own hourly history; descriptive, not a tested claim",
    }
    return {k: (None if isinstance(v, float) and np.isnan(v) else (round(v, 6) if isinstance(v, float) else v)) for k, v in out.items()}


def run(pair_list: list[str] | None = None, out_dir: str | Path = "reports/pairs", cfg: dict | None = None,
        now: pd.Timestamp | None = None) -> dict:
    """Write one brief per pair (markdown + HTML) and one `pairs_latest.json` with a row per
    pair. Returns {"briefs": {pair: path}, "rows": path, "table": DataFrame}."""
    from .run import load_config

    if not medallion.available():
        raise RuntimeError("pair briefs read your own data store, and no reader is connected. Run your connector cell "
                           "(the one that prints '5/5 connected'), then run this cell again.")
    cfg = cfg or load_config()
    pair_list = pair_list or cfg.get("pairs", {}).get("list") or []
    out, shared, briefs, rows = Path(out_dir), {}, {}, []
    out.mkdir(parents=True, exist_ok=True)
    for pr in pair_list:
        try:
            b = brief(pr, now, cfg, shared)
        except Exception as e:  # noqa: BLE001  one pair must not stop the others
            log.warning("pair brief %s skipped: %s: %s", pr, type(e).__name__, str(e)[:200])
            b = {"pair": pr.upper(), "error": f"{type(e).__name__}: {str(e)[:120]}"}
        if "error" not in b:
            try:
                briefs[b["pair"]] = report.render_named({"b": b, "p": b["day"], "w": b["week"]}, "pair.md.j2", out, f"{b['pair']}_{b['asof']:%Y-%m-%d_%H%M}")
            except Exception as e:  # noqa: BLE001
                b = {"pair": b["pair"], "error": f"could not write the brief: {type(e).__name__}: {str(e)[:160]}"}
        if "error" in b:
            log.warning("pair brief %s: %s", b["pair"], b["error"])
        rows.append(row(b))
    rows_path = out / "pairs_latest.json"
    rows_path.write_text(json.dumps({"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": rows}, indent=1, default=str))
    return {"briefs": briefs, "rows": rows_path, "table": pd.DataFrame(rows)}
