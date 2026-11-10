"""Find out which FMP endpoints your plan actually serves.

FMP gates endpoints by plan and answers 402 for the ones you don't have, so the only reliable
way to pick "the best sources" is to ask. This calls each endpoint once with your key and
reports: ok (rows returned), empty, plan (402/403: not on your plan), missing (404), or error.

    from mnp.fmp_probe import probe
    probe()                 # 46 core endpoints
    probe(extended=True)    # about 85 more across the rest of the catalogue

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


def _extended(sym: str, fx: str) -> list[tuple[str, str, str, dict]]:
    """A second sweep across the rest of the catalogue."""
    today = date.today()
    d0, back, yr = today.isoformat(), (today - timedelta(days=30)).isoformat(), (today - timedelta(days=365)).isoformat()
    last_wd = today - timedelta(days=max(0, today.weekday() - 4)) - timedelta(days=1 if today.weekday() < 5 else 0)
    while last_wd.weekday() >= 5:
        last_wd -= timedelta(days=1)
    wd = last_wd.isoformat()
    q = {"symbol": sym}
    econ = [("macro", f"Economic indicator ({n})", "economic-indicators", {"name": n, "from": yr, "to": d0})
            for n in ("CPI", "inflationRate", "federalFunds", "unemploymentRate", "realGDP", "consumerSentiment",
                      "retailSales", "totalNonfarmPayroll", "initialClaims", "industrialProductionTotalIndex",
                      "30YearFixedRateMortgageAverage", "durableGoods")]
    return econ + [
        ("fundamentals", "Key metrics (annual)", "key-metrics", {**q, "limit": 5}),
        ("fundamentals", "Ratios (annual)", "ratios", {**q, "limit": 5}),
        ("fundamentals", "Financial growth", "financial-growth", {**q, "limit": 5}),
        ("fundamentals", "Income statement growth", "income-statement-growth", {**q, "limit": 5}),
        ("fundamentals", "Enterprise values", "enterprise-values", {**q, "limit": 5}),
        ("fundamentals", "Owner earnings", "owner-earnings", {**q, "limit": 5}),
        ("fundamentals", "Levered DCF", "levered-discounted-cash-flow", q),
        ("fundamentals", "Income statement TTM", "income-statement-ttm", {**q, "limit": 1}),
        ("fundamentals", "As-reported income", "income-statement-as-reported", {**q, "limit": 2}),
        ("fundamentals", "Statement dates", "financial-reports-dates", q),
        ("fundamentals", "Peers", "stock-peers", q),
        ("fundamentals", "Shares float", "shares-float", q),
        ("fundamentals", "Market cap", "market-capitalization", q),
        ("fundamentals", "Market cap history", "historical-market-capitalization", {**q, "from": back, "to": d0}),
        ("fundamentals", "Employee count", "employee-count", {**q, "limit": 5}),
        ("fundamentals", "Executives", "key-executives", q),
        ("fundamentals", "Dividends (company)", "dividends", {**q, "limit": 5}),
        ("fundamentals", "Splits (company)", "splits", {**q, "limit": 5}),
        ("fundamentals", "ESG ratings", "esg-ratings", q),
        ("fundamentals", "ESG disclosures", "esg-disclosures", q),
        ("earnings", "Grades history", "grades-historical", {**q, "limit": 5}),
        ("earnings", "Ratings history", "ratings-historical", {**q, "limit": 5}),
        ("earnings", "Price target news", "price-target-news", {**q, "limit": 5}),
        ("earnings", "Grades news", "grades-news", {**q, "limit": 5}),
        ("earnings", "Earnings calendar (confirmed)", "earnings-calendar-confirmed", {"from": d0, "to": (today + timedelta(days=14)).isoformat()}),
        ("prices", "Price change summary", "stock-price-change", q),
        ("prices", "Daily history (full OHLC)", "historical-price-eod/full", {**q, "from": back, "to": d0}),
        ("prices", "Daily, dividend adjusted", "historical-price-eod/dividend-adjusted", {**q, "from": back, "to": d0}),
        ("prices", "Intraday 1 hour", "historical-chart/1hour", {**q, "from": back, "to": d0}),
        ("prices", "Intraday 5 min", "historical-chart/5min", {**q, "from": d0, "to": d0}),
        ("prices", "Intraday 1 hour (FX)", "historical-chart/1hour", {"symbol": fx, "from": back, "to": d0}),
        ("prices", "Aftermarket quote", "aftermarket-quote", q),
        ("prices", "Batch quote", "batch-quote", {"symbols": f"{sym},AAPL,SPY"}),
        ("prices", "Batch FX quotes", "batch-forex-quotes", {}),
        ("prices", "Batch commodity quotes", "batch-commodity-quotes", {}),
        ("prices", "Batch index quotes", "batch-index-quotes", {}),
        ("prices", "Index list", "index-list", {}),
        ("prices", "Index history (S&P 500)", "historical-price-eod/light", {"symbol": "^GSPC", "from": back, "to": d0}),
        ("prices", "Index history (VIX)", "historical-price-eod/light", {"symbol": "^VIX", "from": back, "to": d0}),
        ("prices", "Crypto list", "cryptocurrency-list", {}),
        ("prices", "Crypto history (BTC)", "historical-price-eod/light", {"symbol": "BTCUSD", "from": back, "to": d0}),
        ("technical", "RSI", "technical-indicators/rsi", {**q, "periodLength": 14, "timeframe": "1day", "from": back, "to": d0}),
        ("technical", "SMA", "technical-indicators/sma", {**q, "periodLength": 50, "timeframe": "1day", "from": back, "to": d0}),
        ("technical", "EMA", "technical-indicators/ema", {**q, "periodLength": 20, "timeframe": "1day", "from": back, "to": d0}),
        ("technical", "ADX", "technical-indicators/adx", {**q, "periodLength": 14, "timeframe": "1day", "from": back, "to": d0}),
        ("macro", "Industry performance", "industry-performance-snapshot", {"date": wd}),
        ("macro", "Industry P/E", "industry-pe-snapshot", {"date": wd}),
        ("macro", "Sector performance history", "historical-sector-performance", {"sector": "Technology", "from": back, "to": d0}),
        ("macro", "Sector P/E history", "historical-sector-pe", {"sector": "Technology", "from": back, "to": d0}),
        ("macro", "Biggest gainers", "biggest-gainers", {}),
        ("macro", "Biggest losers", "biggest-losers", {}),
        ("macro", "Most active", "most-actives", {}),
        ("macro", "Market hours", "exchange-market-hours", {"exchange": "NASDAQ"}),
        ("macro", "Exchange holidays", "holidays-by-exchange", {"exchange": "NASDAQ"}),
        ("etf", "ETF holdings", "etf/holdings", {"symbol": "SPY"}),
        ("etf", "ETF info", "etf/info", {"symbol": "SPY"}),
        ("etf", "ETF sector weights", "etf/sector-weightings", {"symbol": "SPY"}),
        ("etf", "ETF country weights", "etf/country-weightings", {"symbol": "EEM"}),
        ("etf", "ETF asset exposure", "etf/asset-exposure", q),
        ("index", "S&P 500 constituents", "sp500-constituent", {}),
        ("index", "Nasdaq constituents", "nasdaq-constituent", {}),
        ("index", "Dow constituents", "dowjones-constituent", {}),
        ("positioning", "COT list", "commitment-of-traders-list", {}),
        ("positioning", "Senate trades (latest)", "senate-latest", {"limit": 10}),
        ("positioning", "House trades (latest)", "house-latest", {"limit": 10}),
        ("positioning", "13F positions summary", "institutional-ownership/symbol-positions-summary", {**q, "year": today.year, "quarter": 2}),
        ("positioning", "Beneficial ownership", "acquisition-of-beneficial-ownership", q),
        ("filings", "Latest 8-K", "sec-filings-8k", {"from": back, "to": d0, "limit": 10}),
        ("filings", "Latest filings", "sec-filings-financials", {"from": back, "to": d0, "limit": 10}),
        ("filings", "M&A latest", "mergers-acquisitions-latest", {"limit": 10}),
        ("search", "Symbol search", "search-symbol", {"query": sym}),
        ("search", "Screener", "company-screener", {"marketCapMoreThan": 500000000000, "limit": 5}),
    ]


def probe(symbol: str = "NVDA", fx: str = "EURUSD", api_key: str | None = None, pause: float = 0.25,
          extended: bool = False) -> pd.DataFrame:
    import requests

    api_key = api_key or os.environ.get("FMP_API_KEY")
    if not api_key:
        raise RuntimeError("FMP_API_KEY is not set (run the keys cell first)")
    rows = []
    for use, label, path, params in (_extended(symbol, fx) if extended else _endpoints(symbol, fx)):
        status, n, fields = "error", 0, ""
        try:
            # "proxy-injected": a network proxy adds the key as a header, so none is sent here
            auth = {} if api_key == "proxy-injected" else {"apikey": api_key}
            r = requests.get(f"{BASE}/{path}", params={**params, **auth}, timeout=30)
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
