"""Find out which FMP endpoints your plan actually serves.

FMP gates endpoints by plan and answers 402 for the ones you don't have, so the only reliable
way to pick "the best sources" is to ask. This calls each endpoint once with your key and
reports: ok (rows returned), empty, plan (402/403: not on your plan), missing (404), or error.

    from mnp.fmp_probe import probe
    probe()            # about 45 requests; the free plan allows 250 a day

The key is read from FMP_API_KEY and is never printed.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, timedelta

import pandas as pd

BASE = "https://financialmodelingprep.com/stable"
_SECRET = re.compile(r"(?i)apikey=[^&\s'\"]+")


def _endpoints(sym: str, fx: str) -> list[tuple[str, str, str, dict]]:
    today = date.today()
    d0, d1, back = today.isoformat(), (today + timedelta(days=14)).isoformat(), (today - timedelta(days=30)).isoformat()
    q = {"symbol": sym}
    return [  # (use in the pipeline, label, path, params)
        ("calendar", "Earnings calendar", "earnings-calendar", {"from": d0, "to": d1}),
        ("calendar", "Economic calendar", "economic-calendar", {"from": d0, "to": d1}),
        ("calendar", "Dividends calendar", "dividends-calendar", {"from": d0, "to": d1}),
        ("calendar", "IPO calendar", "ipos-calendar", {"from": d0, "to": d1}),
        ("fundamentals", "Company profile", "profile", q),
        ("fundamentals", "Income statement (quarterly)", "income-statement", {**q, "period": "quarter", "limit": 5}),
        ("fundamentals", "Balance sheet", "balance-sheet-statement", {**q, "limit": 2}),
        ("fundamentals", "Cash flow", "cash-flow-statement", {**q, "limit": 2}),
        ("fundamentals", "Key metrics (TTM)", "key-metrics-ttm", q),
        ("fundamentals", "Ratios (TTM)", "ratios-ttm", q),
        ("fundamentals", "Financial scores (Altman Z, Piotroski)", "financial-scores", q),
        ("fundamentals", "Revenue by product segment", "revenue-product-segmentation", q),
        ("fundamentals", "Revenue by geography", "revenue-geographic-segmentation", q),
        ("fundamentals", "DCF valuation", "discounted-cash-flow", q),
        ("earnings", "Earnings history (actual vs estimate)", "earnings", {**q, "limit": 8}),
        ("earnings", "Analyst estimates", "analyst-estimates", {**q, "period": "annual", "limit": 4}),
        ("earnings", "Price target consensus", "price-target-consensus", q),
        ("earnings", "Price target summary", "price-target-summary", q),
        ("earnings", "Analyst grades (latest)", "grades", {**q, "limit": 10}),
        ("earnings", "Grades consensus", "grades-consensus", q),
        ("earnings", "Ratings snapshot", "ratings-snapshot", q),
        ("earnings", "Transcript dates", "earning-call-transcript-dates", q),
        ("earnings", "Transcript text", "earning-call-transcript", {**q, "year": today.year, "quarter": 2}),
        ("news", "Stock news (by symbol)", "news/stock", {"symbols": sym, "limit": 20}),
        ("news", "Stock news (latest)", "news/stock-latest", {"limit": 20}),
        ("news", "General news", "news/general-latest", {"limit": 20}),
        ("news", "Forex news", "news/forex-latest", {"limit": 20}),
        ("news", "Forex news (by pair)", "news/forex", {"symbols": fx, "limit": 20}),
        ("news", "Press releases", "news/press-releases", {"symbols": sym, "limit": 10}),
        ("news", "FMP articles", "fmp-articles", {"limit": 10}),
        ("prices", "Quote", "quote", q),
        ("prices", "Daily history (equity)", "historical-price-eod/light", {**q, "from": back, "to": d0}),
        ("prices", "Daily history (FX)", "historical-price-eod/light", {"symbol": fx, "from": back, "to": d0}),
        ("prices", "Daily history (gold)", "historical-price-eod/light", {"symbol": "GCUSD", "from": back, "to": d0}),
        ("prices", "Forex list", "forex-list", {}),
        ("prices", "Commodities list", "commodities-list", {}),
        ("macro", "Treasury rates", "treasury-rates", {"from": back, "to": d0}),
        ("macro", "Economic indicator (GDP)", "economic-indicators", {"name": "GDP"}),
        ("macro", "Market risk premium", "market-risk-premium", {}),
        ("macro", "Sector performance", "sector-performance-snapshot", {"date": (today - timedelta(days=3)).isoformat()}),
        ("macro", "Sector P/E", "sector-pe-snapshot", {"date": (today - timedelta(days=3)).isoformat()}),
        ("positioning", "COT report", "commitment-of-traders-report", {"symbol": "GC", "from": back, "to": d0}),
        ("positioning", "COT analysis", "commitment-of-traders-analysis", {"symbol": "GC", "from": back, "to": d0}),
        ("positioning", "Insider trades (latest)", "insider-trading/latest", {"limit": 10}),
        ("positioning", "Institutional ownership (latest)", "institutional-ownership/latest", {"limit": 10}),
        ("filings", "SEC filings by symbol", "sec-filings-search/symbol", {**q, "from": back, "to": d0}),
    ]


def probe(symbol: str = "NVDA", fx: str = "EURUSD", api_key: str | None = None, pause: float = 0.25) -> pd.DataFrame:
    import requests

    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set (run the keys cell first)")
    rows = []
    for use, label, path, params in _endpoints(symbol, fx):
        status, n, fields = "error", 0, ""
        try:
            r = requests.get(f"{BASE}/{path}", params={**params, "apikey": api_key}, timeout=30)
            if r.status_code in (402, 403):
                status = "plan"
            elif r.status_code == 404:
                status = "missing"
            elif r.status_code == 429:
                status = "rate-limited"
            elif r.status_code == 200:
                data = r.json()
                if isinstance(data, dict) and any(k in data for k in ("Error Message", "error", "message")):
                    msg = str(next(v for k, v in data.items() if k in ("Error Message", "error", "message")))
                    status = "plan" if re.search(r"(?i)subscription|upgrade|premium|not available under", msg) else "error"
                    fields = _SECRET.sub("apikey=***", msg)[:70]
                else:
                    items = data if isinstance(data, list) else [data]
                    n = len(items)
                    status = "ok" if n and items[0] else "empty"
                    if n and isinstance(items[0], dict):
                        fields = ", ".join(list(items[0])[:7])
            else:
                status = f"http {r.status_code}"
        except Exception as e:  # noqa: BLE001
            fields = _SECRET.sub("apikey=***", f"{type(e).__name__}: {e}")[:70]
        rows.append({"use": use, "endpoint": label, "path": path, "status": status, "rows": n, "fields / note": fields})
        time.sleep(pause)
    df = pd.DataFrame(rows)
    counts = df["status"].value_counts().to_dict()
    print("FMP probe:", ", ".join(f"{v} {k}" for k, v in counts.items()))
    return df
