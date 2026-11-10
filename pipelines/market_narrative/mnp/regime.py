"""Decide which week-type template applies: central bank, bond/macro, earnings, or default."""

from __future__ import annotations

import pandas as pd


def _event_col(df: pd.DataFrame) -> str | None:
    for c in ("event", "name", "title", "description"):
        if c in df.columns:
            return c
    return None


def detect(cfg: dict, econ: pd.DataFrame, earnings: pd.DataFrame, window: tuple | None = None) -> dict:
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
        codes, fams = rule.get("event_codes"), rule.get("families")
        if kws or rule.get("dates") or codes or fams:
            hits, labelled = [], set()
            # Labelled calendar rows (your own calendar): match on event code / family.
            if not econ.empty and {"event_code", "family"} <= set(econ.columns):
                labelled = set(econ.loc[econ["event_code"].notna() | econ["family"].notna(), col].astype(str))
                if codes or fams:
                    m = econ[econ["event_code"].isin(codes or []) | econ["family"].isin(fams or [])]
                    for r in m.itertuples():
                        day = pd.Timestamp(r.date).strftime("%a %d") if pd.notna(r.date) else ""
                        hits.append(f"{getattr(r, 'currency', '') or ''} {r.event} ({day})".strip())
            # Keyword matching is for unlabelled sources only (FRED, paid calendars).
            hits += [e for e in events_all if e not in labelled and any(k.lower() in e.lower() for k in (kws or []))]
            # Explicit dated events (e.g. central-bank decisions): {"2026-12-09": "FOMC decision"}
            if window and rule.get("dates"):
                for d, label in rule["dates"].items():
                    if window[0] <= pd.Timestamp(str(d)).date() <= window[1]:
                        hits.append(f"{label} ({d})")
            if hits:
                matches.append({**rule, "events": list(dict.fromkeys(hits))[:15]})
        elif "min_focus_reporters" in rule:
            if len(reporters) >= rule["min_focus_reporters"]:
                matches.append({**rule, "events": []})
        else:  # default
            matches.append({**rule, "events": []})
    if not matches:
        raise ValueError("config.regimes must end with a default rule")
    primary = matches[0]
    others = [m for m in matches[1:] if any(k in m for k in ("econ_keywords", "dates", "event_codes", "families", "min_focus_reporters"))]
    # Carry the other active week types and all their catalysts so nothing is hidden by priority.
    events = list(primary["events"])
    for m in others:
        events += [e for e in m["events"] if e not in events]
    return {**primary, "events": events[:20], "reporters": reporters, "also": [m["name"] for m in others]}
