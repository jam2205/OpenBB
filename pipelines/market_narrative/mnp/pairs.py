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


def brief(pair: str, now: pd.Timestamp | None = None, cfg: dict | None = None, shared: dict | None = None) -> dict:
    """Everything the pair template needs. `shared` carries tables that are the same for every
    pair (yields, calendar) so a run over many pairs reads them once."""
    pair = pair.upper().replace("/", "")
    base, quote = pair[:3], pair[3:]
    pcfg = (cfg or {}).get("pairs", {})
    bars = medallion.hourly(pair, date.today() - timedelta(days=int(pcfg.get("history_days", 3650))))
    if bars.empty:
        return {"pair": pair, "error": "no hourly bars for this pair in your candles dataset"}
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
    return {
        "pair": pair, "base": base, "quote": quote, "asof": p["asof"], "data_age_h": round(age_h, 1), "stale": age_h > float(pcfg.get("max_age_h", 3)),
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
        "pair": b["pair"], "ts": b["asof"].isoformat(), "source": "mnp_pair_brief", "template": b["template"], "data_age_h": b["data_age_h"],
        "stale": bool(b["stale"]), "hours_into_day": p["hours_elapsed"], "last": p["last"], "day_low": p["low"], "day_high": p["high"],
        "expected_range": p["expected_range"], "expected_from": p["expected_from"], "used_of_expected": p["used_of_expected"],
        "p_low_in": lo["p"], "p_low_in_baseline": lo["baseline"], "p_low_in_random_walk": lo.get("null"), "low_vs_random_walk": lo["vs_null"],
        "low_n_days": lo["n_days"], "low_n_weeks": lo["n_weeks"], "low_verdict": b["low_verdict"],
        "p_high_in": hi["p"], "p_high_in_baseline": hi["baseline"], "p_high_in_random_walk": hi.get("null"), "high_vs_random_walk": hi["vs_null"],
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
        rows.append(row(b))
        if "error" in b:
            log.warning("pair brief %s: %s", b["pair"], b["error"])
            continue
        briefs[b["pair"]] = report.render_named({"b": b, "p": b["day"], "w": b["week"]}, "pair.md.j2", out, f"{b['pair']}_{b['asof']:%Y-%m-%d_%H%M}")
    rows_path = out / "pairs_latest.json"
    rows_path.write_text(json.dumps({"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "rows": rows}, indent=1, default=str))
    return {"briefs": briefs, "rows": rows_path, "table": pd.DataFrame(rows)}
