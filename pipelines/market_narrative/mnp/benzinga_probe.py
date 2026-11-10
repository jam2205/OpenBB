"""Check what your Benzinga key delivers before you train on it.

Run this BEFORE building a training set. It answers:
  1. Which endpoints does the key serve (news, calendars, ratings, guidance ...)?
  2. How far back does news go, and how many articles come back per window?
  3. Do articles carry a body, or only a headline and teaser?
  4. Which channels / tags are attached? (They can serve as driver labels for fine-tuning,
     which is far better than keyword-matched weak labels.)
  5. How well are FX, commodities, rates and your focus tickers covered?
  6. Does the economic calendar carry `actual` and `consensus`, and for which countries?
     (That is what makes a release *surprise* computable.)

    from mnp.benzinga_probe import probe
    report = probe()          # about 45 requests; prints a summary and returns the tables

The key is read from BENZINGA_API_KEY, sent as the `token` URL parameter (the only form the API
accepts), and never printed. First run against a live key on 2026-11-10: news, the removed feed
and every calendar answered; the channels list and logo search returned HTTP 400 and transcripts
503, so those three paths or parameters are still unconfirmed.
"""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from datetime import date, timedelta

import pandas as pd

BASE = "https://api.benzinga.com"
_SECRET = re.compile(r"(?i)token=[^&\s'\"]+")
_TAG = re.compile(r"<[^>]+>")


def _call(path: str, key: str, **params):
    """-> (status, payload). status: ok | empty | plan | missing | rate-limited | error:<detail>"""
    import requests

    try:
        r = requests.get(f"{BASE}{path}", params={**params, "token": key}, headers={"accept": "application/json"}, timeout=40)
    except Exception as e:  # noqa: BLE001
        return "error: " + _SECRET.sub("token=***", f"{type(e).__name__}: {e}")[:80], None
    if r.status_code in (401, 402, 403):
        return "plan", None
    if r.status_code == 404:
        return "missing", None
    if r.status_code == 429:
        return "rate-limited", None
    if r.status_code != 200:
        return f"error: http {r.status_code}", None
    try:
        data = r.json()
    except Exception:  # noqa: BLE001
        return "error: not json", None
    return ("ok" if data else "empty"), data


def _rows(data, key: str | None = None) -> list:
    """Benzinga wraps calendar results as {"<name>": [...]}; news is a bare list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if key and isinstance(data.get(key), list):
            return data[key]
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data] if data else []
    return []


def _names(items) -> list[str]:
    return [str(i.get("name")) for i in (items or []) if isinstance(i, dict) and i.get("name")]


def probe(tickers: tuple[str, ...] = ("NVDA", "AAPL", "SPY", "QQQ", "GLD"), api_key: str | None = None,
          pause: float = 0.3, sample_pages: int = 5) -> dict:
    key = api_key or os.environ.get("BENZINGA_API_KEY")
    if not key:
        raise RuntimeError("BENZINGA_API_KEY is not set (run the keys cell first)")
    today = date.today()
    d = lambda n: (today - timedelta(days=n)).isoformat()  # noqa: E731
    out: dict = {}

    def cal(name, **extra):
        return (f"/api/v2.1/calendar/{name}", {"parameters[date_from]": d(30), "parameters[date_to]": (today + timedelta(days=14)).isoformat(),
                                               "pagesize": 50, **extra}, name)

    # ---- 1. endpoint availability -------------------------------------------------------
    endpoints = [
        ("news", "News (latest)", "/api/v2/news", {"pageSize": 5, "displayOutput": "full"}, None),
        ("news", "News by ticker", "/api/v2/news", {"pageSize": 5, "tickers": tickers[0]}, None),
        ("news", "News channels list", "/api/v2/news/channels", {}, None),
        ("news", "News removed/updated feed", "/api/v2/news-removed", {"pageSize": 5}, None),
        ("calendar", "Economics", *cal("economics")[:2], "economics"),
        ("calendar", "Earnings", *cal("earnings")[:2], "earnings"),
        ("calendar", "Analyst ratings", *cal("ratings")[:2], "ratings"),
        ("calendar", "Guidance", *cal("guidance")[:2], "guidance"),
        ("calendar", "Dividends", *cal("dividends")[:2], "dividends"),
        ("calendar", "Conference calls", *cal("conference-calls")[:2], "conference"),
        ("calendar", "M&A", *cal("ma")[:2], "ma"),
        ("calendar", "IPOs", *cal("ipos")[:2], "ipos"),
        ("calendar", "Splits", *cal("splits")[:2], "splits"),
        ("calendar", "Offerings", *cal("offerings")[:2], "offerings"),
        ("calendar", "FDA", *cal("fda")[:2], "fda"),
        ("analyst", "Analyst insights", "/api/v1/analyst/insights", {"symbols": tickers[0], "pageSize": 5}, "analyst-insights"),
        ("analyst", "Bulls say / bears say", "/api/v1/bulls_bears_say", {"symbols": tickers[0]}, "bulls_say_bears_say"),
        ("analyst", "Consensus ratings", "/api/v1/consensus-ratings", {"parameters[tickers]": tickers[0]}, None),
        ("signals", "Unusual options activity", "/api/v1/signal/option_activity", {"parameters[date_from]": d(5), "pagesize": 5}, "option_activity"),
        ("signals", "Government trades", "/api/v1/gov/usa/congress/trades", {"pagesize": 5}, None),
        ("signals", "Insider transactions", "/api/v1/sec/insider_transactions/filings", {"pagesize": 5}, None),
        ("market", "Delayed quote", "/api/v2/quoteDelayed", {"symbols": ",".join(tickers[:3])}, None),
        ("market", "Bars", "/api/v2/bars", {"symbols": tickers[0], "from": d(10), "interval": "1D"}, None),
        ("market", "Movers", "/api/v1/market/movers", {}, None),
        ("company", "Fundamentals", "/api/v2.1/fundamentals", {"symbols": tickers[0]}, None),
        ("company", "Transcripts", "/api/v1/earnings-call-transcripts", {"tickers": tickers[0], "pagesize": 2}, None),
        ("company", "Logos", "/api/v2/logos/search", {"search_keys": tickers[0], "search_keys_type": "symbol"}, None),
    ]
    rows = []
    for use, label, path, params, wrap in endpoints:
        status, data = _call(path, key, **params)
        items = _rows(data, wrap) if data else []
        fields = ", ".join(list(items[0])[:8]) if items and isinstance(items[0], dict) else ""
        rows.append({"use": use, "endpoint": label, "path": path, "status": "empty" if status == "ok" and not items else status,
                     "rows": len(items), "fields": fields})
        time.sleep(pause)
    out["endpoints"] = pd.DataFrame(rows)

    # ---- 2. news history depth ----------------------------------------------------------
    depth = []
    for label, days in (("last 7 days", 7), ("1 year ago", 365), ("3 years ago", 1095), ("5 years ago", 1825), ("10 years ago", 3650)):
        lo, hi = (d(7), today.isoformat()) if days == 7 else (d(days + 7), d(days))
        status, data = _call("/api/v2/news", key, pageSize=100, dateFrom=lo, dateTo=hi, displayOutput="headline")
        items = _rows(data)
        depth.append({"window": label, "from": lo, "to": hi, "status": status,
                      "articles (max 100 per page)": len(items), "first title": (items[0].get("title", "")[:60] if items else "")})
        time.sleep(pause)
    out["history"] = pd.DataFrame(depth)

    # ---- 3 + 4. content and labels on a sample of recent articles ------------------------
    sample = []
    for page in range(sample_pages):
        status, data = _call("/api/v2/news", key, pageSize=100, page=page, displayOutput="full", dateFrom=d(7))
        items = _rows(data)
        if not items:
            break
        sample += items
        time.sleep(pause)
    chan, tags, stocks, body_len, teaser_len = Counter(), Counter(), Counter(), [], []
    for a in sample:
        chan.update(_names(a.get("channels")))
        tags.update(_names(a.get("tags")))
        stocks.update(_names(a.get("stocks")))
        body_len.append(len(_TAG.sub("", a.get("body") or "")))
        teaser_len.append(len(_TAG.sub("", a.get("teaser") or "")))
    n = len(sample)
    out["content"] = pd.DataFrame({"value": {
        "articles sampled (last 7 days)": n,
        "with a body": f"{sum(b > 0 for b in body_len)} ({100 * sum(b > 0 for b in body_len) // max(n, 1)}%)",
        "median body length (chars)": int(pd.Series(body_len).median()) if body_len else 0,
        "with a teaser": f"{sum(t > 0 for t in teaser_len)} ({100 * sum(t > 0 for t in teaser_len) // max(n, 1)}%)",
        "with at least one channel": f"{sum(bool(a.get('channels')) for a in sample)} ({100 * sum(bool(a.get('channels')) for a in sample) // max(n, 1)}%)",
        "with at least one ticker": f"{sum(bool(a.get('stocks')) for a in sample)} ({100 * sum(bool(a.get('stocks')) for a in sample) // max(n, 1)}%)",
        "distinct channels / tags / tickers": f"{len(chan)} / {len(tags)} / {len(stocks)}",
        "fields on an article": ", ".join(list(sample[0])) if sample else "",
    }})
    out["channels"] = pd.DataFrame(chan.most_common(40), columns=["channel", "articles"])
    out["tags"] = pd.DataFrame(tags.most_common(25), columns=["tag", "articles"])

    # ---- 5. coverage of what the report cares about -------------------------------------
    cover = []
    queries = [(f"ticker {t}", {"tickers": t}) for t in tickers]
    queries += [(f"channel {c}", {"channels": c}) for c in ("Forex", "Commodities", "Economics", "Federal Reserve", "Bonds",
                                                            "Treasuries", "Earnings", "Guidance", "Analyst Ratings", "Macro Economic Events",
                                                            "Global", "Futures", "Cryptocurrency")]
    queries += [(f"topic {t}", {"topics": t}) for t in ("gold", "yen", "Swiss franc", "inflation", "tariffs")]
    for label, q in queries:
        # sort is explicit: without it a date-filtered query came back oldest-first, so "newest" showed
        # the start of the window (seen on the first live run, 2026-11-10).
        status, data = _call("/api/v2/news", key, pageSize=100, dateFrom=d(30), displayOutput="headline", sort="created:desc", **q)
        items = _rows(data)
        cover.append({"query": label, "status": status, "articles, last 30d (max 100)": len(items),
                      "newest": (items[0].get("created", "")[:16] if items else ""), "example": (items[0].get("title", "")[:70] if items else "")})
        time.sleep(pause)
    out["coverage"] = pd.DataFrame(cover)

    # ---- 6. economic calendar: can a surprise be computed? ------------------------------
    status, data = _call("/api/v2.1/calendar/economics", key, **{"parameters[date_from]": d(30), "parameters[date_to]": today.isoformat(), "pagesize": 1000})
    econ = pd.DataFrame(_rows(data, "economics"))
    if not econ.empty:
        def has(c):
            if c not in econ:
                return pd.Series(False, index=econ.index)
            return econ[c].notna() & ~econ[c].astype(str).str.strip().str.lower().isin(["", "none", "nan", "<na>", "null"])

        both = has("actual") & has("consensus")
        by = pd.DataFrame({"events": econ.groupby("country").size() if "country" in econ else pd.Series(dtype=int)})
        if "country" in econ:
            by["with actual"] = has("actual").groupby(econ["country"]).sum()
            by["with actual and consensus"] = both.groupby(econ["country"]).sum()
        out["econ_by_country"] = by.sort_values("events", ascending=False).head(25)
        out["econ_summary"] = pd.DataFrame({"value": {
            "events, last 30 days": len(econ), "with actual": int(has("actual").sum()), "with consensus": int(has("consensus").sum()),
            "with both (surprise computable)": f"{int(both.sum())} ({100 * int(both.sum()) // max(len(econ), 1)}%)",
            "fields": ", ".join(list(econ.columns)[:20]),
            "importance values": ", ".join(sorted(map(str, econ["importance"].dropna().unique()))[:8]) if "importance" in econ else "",
        }})
    else:
        out["econ_summary"] = pd.DataFrame({"value": {"economics calendar": status}})
        out["econ_by_country"] = pd.DataFrame()

    # ---- printed summary ----------------------------------------------------------------
    with pd.option_context("display.max_rows", 200, "display.max_colwidth", 90, "display.width", 250):
        for title, name in (("1. ENDPOINTS", "endpoints"), ("2. NEWS HISTORY DEPTH", "history"), ("3. ARTICLE CONTENT", "content"),
                            ("4a. CHANNELS (candidate training labels)", "channels"), ("4b. TAGS", "tags"), ("5. COVERAGE", "coverage"),
                            ("6a. ECONOMIC CALENDAR", "econ_summary"), ("6b. ECONOMIC CALENDAR BY COUNTRY", "econ_by_country")):
            print(f"\n===== {title} =====")
            print(out[name].to_string() if not out[name].empty else "(nothing returned)")
    return out
