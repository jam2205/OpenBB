"""Decide which week-type template applies: central bank, bond/macro, earnings, or default."""

from __future__ import annotations

import pandas as pd


def _event_col(df: pd.DataFrame) -> str | None:
    for c in ("event", "name", "title", "description"):
        if c in df.columns:
            return c
    return None


def detect(cfg: dict, econ: pd.DataFrame, earnings: pd.DataFrame) -> dict:
    """Return {'name', 'template', 'events': [...], 'reporters': [...]} for the upcoming window."""
    focus = set(cfg["universe"]["equities"])
    reporters: list[str] = []
    if not earnings.empty and "symbol" in earnings:
        reporters = sorted(set(earnings["symbol"]) & focus)

    col = _event_col(econ) if not econ.empty else None
    events_all = econ[col].astype(str).tolist() if col else []

    matches = []
    for rule in cfg["regimes"]:
        kws = rule.get("econ_keywords")
        if kws:
            hits = [e for e in events_all if any(k.lower() in e.lower() for k in kws)]
            if hits:
                matches.append({**rule, "events": sorted(set(hits))[:15]})
        elif "min_focus_reporters" in rule:
            if len(reporters) >= rule["min_focus_reporters"]:
                matches.append({**rule, "events": []})
        else:  # default
            matches.append({**rule, "events": []})
    if not matches:
        raise ValueError("config.regimes must end with a default rule")
    primary = matches[0]
    others = [m for m in matches[1:] if "econ_keywords" in m or "min_focus_reporters" in m]
    # Carry the other active week types and all their catalysts so nothing is hidden by priority.
    events = list(primary["events"])
    for m in others:
        events += [e for e in m["events"] if e not in events]
    return {**primary, "events": events[:20], "reporters": reporters, "also": [m["name"] for m in others]}
