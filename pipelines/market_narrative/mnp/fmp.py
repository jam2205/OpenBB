"""Financial Modeling Prep, direct REST (stable API).

Built only on endpoints verified to answer on a Starter-level plan (probe run 2026-11-10; re-run
`mnp.fmp_probe.probe()` if your plan changes). Limits seen on that plan and respected here:
`limit` <= 5 on earnings and statements, annual periods only for estimates and segments, FX majors
but not crosses. Not on that plan, so not used: news, economic calendar, transcripts, COT,
insider search, institutional ownership, intraday bars, technical indicators, ETF holdings.

Responses are cached on disk per calendar day (~/.cache/mnp_fmp), so re-running a report costs no
API calls. If FMP reports that the request allowance is used up, the module stops calling it for
the rest of the run and says so once, instead of retrying.

Key: env FMP_API_KEY (the notebook's keys cell sets it). It is sent as a query parameter and is
masked in every log line. The value "proxy-injected" means a network proxy adds it as a header.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
BASE = "https://financialmodelingprep.com/stable"
_SECRET = re.compile(r"(?i)apikey=[^&\s'\"]+")
_plan_blocked: set[str] = set()


class PlanError(RuntimeError):
    """The endpoint or parameter is not included in the user's FMP plan."""


class RateLimited(RuntimeError):
    """FMP reported that the plan's request allowance is used up."""


_limited = False          # set once per process: after a limit response, stop calling for this run
CACHE_DIR = Path(os.environ.get("MNP_FMP_CACHE", Path.home() / ".cache" / "mnp_fmp"))


def _cache_file(path: str, params: dict) -> Path:
    """One file per (endpoint, parameters, calendar day). Fundamentals do not change intraday,
    so a second run on the same day costs no API calls."""
    raw = json.dumps([path, sorted((k, str(v)) for k, v in params.items())])
    return CACHE_DIR / date.today().isoformat() / (hashlib.sha1(raw.encode()).hexdigest()[:20] + ".json")


def reset_limit() -> None:
    """Call after waiting out a rate limit to let this process try FMP again."""
    global _limited
    _limited = False


def _key() -> str | None:
    key = os.environ.get("FMP_API_KEY")
    if not key:
        try:
            from openbb import obb

            k = obb.user.credentials.fmp_api_key
            key = k.get_secret_value() if hasattr(k, "get_secret_value") else k
        except Exception:  # noqa: BLE001
            key = None
    return key or None


def available() -> bool:
    return _key() is not None


def get(path: str, retries: int = 1, cache: bool = True, **params) -> list[dict]:
    """One FMP call -> list of rows, cached per day.

    Raises PlanError (402/403: not on the plan), RateLimited (allowance used up; no further calls
    are made in this process until reset_limit()), or RuntimeError.
    """
    import requests

    global _limited
    key = _key()
    if not key:
        raise RuntimeError("FMP_API_KEY is not set")
    if path in _plan_blocked:
        raise PlanError(path)
    cf = _cache_file(path, params)
    if cache and cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:  # noqa: BLE001
            pass
    if _limited:
        raise RateLimited("FMP request allowance used up")
    auth = {} if key == "proxy-injected" else {"apikey": key}
    err = ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{BASE}/{path}", params={**params, **auth}, timeout=30)
        except Exception as e:  # noqa: BLE001
            err = _SECRET.sub("apikey=***", f"{type(e).__name__}: {e}")
            time.sleep(1.5 * (attempt + 1))
            continue
        body = r.text[:300]
        if r.status_code == 429 or "Limit Reach" in body:
            _limited = True
            raise RateLimited("FMP request allowance used up")
        if r.status_code in (402, 403):
            if "Parameter" not in body[:60]:   # whole endpoint is off-plan: don't ask again
                _plan_blocked.add(path)
            raise PlanError(f"{path}: not on this FMP plan")
        if r.status_code >= 500:
            err = f"HTTP {r.status_code}"
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"{path}: HTTP {r.status_code}")
        data = r.json()
        if isinstance(data, dict) and "Error Message" in data:
            raise PlanError(f"{path}: {str(data['Error Message'])[:80]}")
        rows = data if isinstance(data, list) else [data]
        if cache:   # empty answers are cached too: they are answers, and cost a call to repeat
            try:
                cf.parent.mkdir(parents=True, exist_ok=True)
                cf.write_text(json.dumps(rows))
            except Exception:  # noqa: BLE001
                pass
        return rows
    raise RuntimeError(f"{path}: {err}")


_warned_limit = False


def frame(path: str, **params) -> pd.DataFrame:
    """Like get(), but returns an empty frame (and one log line) instead of raising."""
    global _warned_limit
    try:
        return pd.DataFrame(get(path, **params))
    except RateLimited:
        if not _warned_limit:
            log.warning("FMP request allowance is used up: FMP sections are skipped for the rest of this run "
                        "(already-cached data is still used). Try again later.")
            _warned_limit = True
    except PlanError as e:
        log.info("FMP %s", e)
    except Exception as e:  # noqa: BLE001
        log.warning("FMP %s unavailable: %s", path, str(e)[:160])
    return pd.DataFrame()


def _bn(x) -> float:
    return np.nan if x is None or pd.isna(x) else float(x) / 1e9


# ---------------------------------------------------------------- per-company tables
def earnings_track(symbol: str) -> tuple[dict, pd.DataFrame]:
    """(next report, history). History has EPS and revenue surprise per past quarter."""
    df = frame("earnings", symbol=symbol, limit=5)
    if df.empty:
        return {}, pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    upcoming = df[df["epsActual"].isna() & (df["date"] >= pd.Timestamp(date.today()))]
    nxt = {}
    if not upcoming.empty:
        u = upcoming.iloc[0]
        nxt = {"date": u["date"].date(), "eps_est": u.get("epsEstimated"), "revenue_est_bn": _bn(u.get("revenueEstimated"))}
    past = df[df["epsActual"].notna()].copy()
    hist = pd.DataFrame({
        "eps_actual": past["epsActual"], "eps_est": past["epsEstimated"],
        "eps_surprise_%": (past["epsActual"] / past["epsEstimated"] - 1) * 100,
        "revenue_bn": past["revenueActual"].map(_bn), "revenue_est_bn": past["revenueEstimated"].map(_bn),
        "rev_surprise_%": (past["revenueActual"] / past["revenueEstimated"] - 1) * 100,
    })
    hist.index = past["date"].dt.date.astype(str)
    hist.index.name = "reported"
    if nxt and not hist.empty:
        nxt["eps_est_vs_last_actual_%"] = (nxt["eps_est"] / hist["eps_actual"].iloc[-1] - 1) * 100 if nxt.get("eps_est") else np.nan
        nxt["beat_rate"] = f"{int((hist['eps_surprise_%'] > 0).sum())}/{len(hist)} EPS beats, avg surprise {hist['eps_surprise_%'].mean():+.1f}%"
    return nxt, hist.round(2)


def street_view(symbol: str, price: float | None = None, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary, recent rating changes). Summary: ratings split, price targets, target trend."""
    rows: dict[str, object] = {}
    gc = frame("grades-consensus", symbol=symbol)
    if not gc.empty:
        g = gc.iloc[0]
        n = sum(int(g.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
        rows["Consensus rating"] = g.get("consensus")
        rows["Buy / hold / sell"] = f"{int(g.get('strongBuy', 0)) + int(g.get('buy', 0))} / {int(g.get('hold', 0))} / {int(g.get('sell', 0)) + int(g.get('strongSell', 0))}"
        rows["% buy"] = round(100 * (int(g.get("strongBuy", 0)) + int(g.get("buy", 0))) / n, 1) if n else np.nan
    pt = frame("price-target-consensus", symbol=symbol)
    if not pt.empty:
        p = pt.iloc[0]
        rows["Target consensus / median"] = f"{p.get('targetConsensus')} / {p.get('targetMedian')}"
        rows["Target low - high"] = f"{p.get('targetLow')} - {p.get('targetHigh')}"
        if price:
            rows["Upside to consensus %"] = round((float(p["targetConsensus"]) / price - 1) * 100, 1)
    ps = frame("price-target-summary", symbol=symbol)
    if not ps.empty:
        s = ps.iloc[0]
        rows["Avg target: last month / quarter / year"] = (f"{s.get('lastMonthAvgPriceTarget')} ({s.get('lastMonthCount')}) / "
                                                           f"{s.get('lastQuarterAvgPriceTarget')} ({s.get('lastQuarterCount')}) / "
                                                           f"{s.get('lastYearAvgPriceTarget')} ({s.get('lastYearCount')})")
        lm, lq = s.get("lastMonthAvgPriceTarget"), s.get("lastQuarterAvgPriceTarget")
        if lm and lq and int(s.get("lastMonthCount") or 0) >= 3:
            rows["Target drift, month vs quarter %"] = round((float(lm) / float(lq) - 1) * 100, 1)
    changes = pd.DataFrame()
    gr = frame("grades", symbol=symbol, limit=100)
    if not gr.empty:
        gr["date"] = pd.to_datetime(gr["date"])
        recent = gr[gr["date"] >= pd.Timestamp(date.today() - timedelta(days=days))]
        counts = recent["action"].value_counts()
        rows[f"Rating actions, last {days}d"] = (f"{int(counts.get('upgrade', 0))} up / {int(counts.get('downgrade', 0))} down / "
                                                 f"{int(counts.get('maintain', 0))} maintained")
        ch = recent[recent["action"].isin(["upgrade", "downgrade"])].head(8)
        if not ch.empty:
            changes = pd.DataFrame({"firm": ch["gradingCompany"], "action": ch["action"], "from": ch["previousGrade"], "to": ch["newGrade"]})
            changes.index = ch["date"].dt.date.astype(str)
            changes.index.name = "date"
    summary = pd.DataFrame({symbol: pd.Series(rows, dtype=object)})
    return summary, changes


def quality(symbol: str) -> pd.DataFrame:
    """Balance-sheet and profitability scores, valuation, and DCF value against price."""
    rows: dict[str, object] = {}
    fs = frame("financial-scores", symbol=symbol)
    if not fs.empty:
        rows["Altman Z (>3 safe, <1.8 distress)"] = round(float(fs.iloc[0].get("altmanZScore", np.nan)), 1)
        rows["Piotroski F (0-9)"] = fs.iloc[0].get("piotroskiScore")
    rt = frame("ratios-ttm", symbol=symbol)
    if not rt.empty:
        r = rt.iloc[0]
        for label, k, pct in (("Gross margin %", "grossProfitMarginTTM", True), ("Operating margin %", "operatingProfitMarginTTM", True),
                              ("Net margin %", "netProfitMarginTTM", True), ("P/E (TTM)", "priceToEarningsRatioTTM", False),
                              ("P/S (TTM)", "priceToSalesRatioTTM", False), ("P/FCF (TTM)", "priceToFreeCashFlowRatioTTM", False),
                              ("Debt / equity", "debtToEquityRatioTTM", False)):
            if r.get(k) is not None:
                rows[label] = round(float(r[k]) * (100 if pct else 1), 1 if pct else 2)
    km = frame("key-metrics-ttm", symbol=symbol)
    if not km.empty:
        k_ = km.iloc[0]
        for label, k, pct in (("EV / EBITDA", "evToEBITDATTM", False), ("ROE %", "returnOnEquityTTM", True),
                              ("ROIC %", "returnOnInvestedCapitalTTM", True), ("FCF yield %", "freeCashFlowYieldTTM", True)):
            if k_.get(k) is not None:
                rows[label] = round(float(k_[k]) * (100 if pct else 1), 1)
    dcf = frame("discounted-cash-flow", symbol=symbol)
    if not dcf.empty and dcf.iloc[0].get("dcf") and dcf.iloc[0].get("Stock Price"):
        d = dcf.iloc[0]
        rows["DCF value vs price"] = f"{float(d['dcf']):.0f} vs {float(d['Stock Price']):.0f} ({(float(d['dcf']) / float(d['Stock Price']) - 1) * 100:+.0f}%)"
    return pd.DataFrame({symbol: pd.Series(rows, dtype=object)})


def segments(symbol: str, kind: str = "product") -> pd.DataFrame:
    """Revenue by product or geography: latest fiscal year, share of total, growth on prior year."""
    path = "revenue-product-segmentation" if kind == "product" else "revenue-geographic-segmentation"
    df = frame(path, symbol=symbol)
    if df.empty or "data" not in df:
        return pd.DataFrame()
    df = df.sort_values("date", ascending=False)
    cur, prev = df.iloc[0], (df.iloc[1] if len(df) > 1 else None)
    now = pd.Series(cur["data"], dtype=float)
    t = pd.DataFrame({"revenue_bn": now / 1e9, "share_%": now / now.sum() * 100})
    if prev is not None:
        t["yoy_%"] = (now / pd.Series(prev["data"], dtype=float).reindex(now.index) - 1) * 100
    t = t.sort_values("revenue_bn", ascending=False).round(1)
    t.attrs["fiscal_year"] = cur.get("fiscalYear")
    return t


def income_quarters(symbol: str) -> pd.DataFrame:
    """Last five quarters in USD bn with margins, quarter-on-quarter and year-on-year revenue growth."""
    df = frame("income-statement", symbol=symbol, period="quarter", limit=5)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date")
    t = pd.DataFrame({
        "revenue_bn": df["revenue"] / 1e9, "gross_margin_%": df["grossProfit"] / df["revenue"] * 100,
        "op_margin_%": df["operatingIncome"] / df["revenue"] * 100, "net_income_bn": df["netIncome"] / 1e9,
        "eps_diluted": df.get("epsDiluted"),
    })
    t["rev_qoq_%"] = t["revenue_bn"].pct_change() * 100
    t["rev_yoy_%"] = t["revenue_bn"].pct_change(4) * 100
    t.index = (df["period"].astype(str) + " FY" + df["fiscalYear"].astype(str) + " (" + df["date"].astype(str) + ")").values
    return t.round(2)


def forward_estimates(symbol: str, years: int = 3) -> pd.DataFrame:
    """Consensus for the next fiscal years: revenue and EPS (average and range), analyst count."""
    df = frame("analyst-estimates", symbol=symbol, period="annual", limit=10)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    fut = df[df["date"] >= pd.Timestamp(date.today())].head(years)
    last = df[df["date"] < pd.Timestamp(date.today())].tail(1)
    sel = pd.concat([last, fut])
    t = pd.DataFrame({
        "revenue_bn": sel["revenueAvg"] / 1e9, "rev_growth_%": (sel["revenueAvg"].pct_change() * 100),
        "eps": sel["epsAvg"], "eps_range": sel["epsLow"].round(2).astype(str) + " - " + sel["epsHigh"].round(2).astype(str),
        "eps_growth_%": sel["epsAvg"].pct_change() * 100, "analysts": sel.get("numAnalystsEps"),
    })
    t.index = ["FY ending " + d.strftime("%b %Y") + (" (last reported est.)" if d < pd.Timestamp(date.today()) else "") for d in sel["date"]]
    return t.round(2)


def filings(symbol: str, days: int = 120, n: int = 6) -> pd.DataFrame:
    """Recent company filings (8-K, 10-Q, 10-K ...), insider forms excluded."""
    today = date.today()
    df = frame("sec-filings-search/symbol", symbol=symbol, **{"from": (today - timedelta(days=days)).isoformat(), "to": today.isoformat()})
    if df.empty or "formType" not in df:
        return pd.DataFrame()
    df = df[~df["formType"].astype(str).str.match(r"^(3|4|5|144)(/A)?$")].head(n)
    t = pd.DataFrame({"form": df["formType"], "link": df.get("finalLink", df.get("link"))})
    t.index = pd.to_datetime(df["filingDate"]).dt.date.astype(str)
    t.index.name = "filed"
    return t


def ratings_trend(symbol: str) -> pd.DataFrame:
    """Analyst rating counts over recent months: is the Street getting more or less bullish?"""
    df = frame("grades-historical", symbol=symbol, limit=5)
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date")
    buy = df["analystRatingsStrongBuy"] + df["analystRatingsBuy"]
    sell = df["analystRatingsSell"] + df["analystRatingsStrongSell"]
    tot = buy + df["analystRatingsHold"] + sell
    t = pd.DataFrame({"buy": buy, "hold": df["analystRatingsHold"], "sell": sell, "buy_%": (buy / tot * 100).round(1)})
    t.index = pd.to_datetime(df["date"]).dt.strftime("%b %Y").values
    return t


def price_moves(symbol: str) -> pd.DataFrame:
    """Percent change over standard windows (1D ... 1Y)."""
    df = frame("stock-price-change", symbol=symbol)
    if df.empty:
        return pd.DataFrame()
    cols = [c for c in ("1D", "5D", "1M", "3M", "6M", "ytd", "1Y") if c in df]
    return df[cols].round(1).set_axis([symbol])


def peers(symbol: str, n: int = 8) -> pd.DataFrame:
    """Closest peers by FMP's grouping, largest first, with market cap in USD bn."""
    df = frame("stock-peers", symbol=symbol)
    if df.empty or "mktCap" not in df:
        return pd.DataFrame()
    df = df.sort_values("mktCap", ascending=False).head(n)
    t = pd.DataFrame({"company": df["companyName"], "price": df["price"], "mkt_cap_bn": (df["mktCap"] / 1e9).round(1)})
    t.index = df["symbol"].values
    return t


US_INDICATORS = {  # FMP name -> (label, how to show the change)
    "CPI": ("CPI index", "yoy"), "federalFunds": ("Fed funds %", "level"),
    "unemploymentRate": ("Unemployment rate %", "level"), "totalNonfarmPayroll": ("Nonfarm payrolls (k)", "diff"),
    "initialClaims": ("Initial jobless claims", "level"), "retailSales": ("Retail sales ($mn)", "mom"),
    "industrialProductionTotalIndex": ("Industrial production index", "mom"), "consumerSentiment": ("Consumer sentiment", "level"),
    "durableGoods": ("Durable goods orders ($mn)", "mom"), "30YearFixedRateMortgageAverage": ("30Y mortgage rate %", "level"),
}


def us_scorecard(asof: date, names: list[str] | None = None) -> pd.DataFrame:
    """Latest print of each US indicator against the one before it (and a year earlier for CPI).

    These are the released values, not forecasts: use them as "where the data stands going in".
    """
    def fmt(v):
        if v is None or pd.isna(v):
            return ""
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"

    rows = {}
    start = (asof - timedelta(days=800)).isoformat()   # two years, so year-on-year always has a base
    for name in names or list(US_INDICATORS):
        label, mode = US_INDICATORS.get(name, (name, "level"))
        df = frame("economic-indicators", name=name, **{"from": start, "to": asof.isoformat()})
        if df.empty or "value" not in df:
            continue
        df = df.sort_values("date")
        last, prev = df.iloc[-1], (df.iloc[-2] if len(df) > 1 else None)
        row = {"period": str(last["date"])[:10], "latest": fmt(last["value"]), "previous": fmt(prev["value"]) if prev is not None else "",
               "previous period": str(prev["date"])[:10] if prev is not None else ""}
        if prev is not None:
            if mode == "mom":
                row["change"] = f"{(last['value'] / prev['value'] - 1) * 100:+.2f}% vs previous"
            elif mode == "diff":
                row["change"] = f"{last['value'] - prev['value']:+,.0f}"
            elif mode == "yoy":
                # Some plans serve under a year of history, so fall back to a 3-month annualised rate.
                when = pd.to_datetime(df["date"])
                t_last = pd.Timestamp(str(last["date"])[:10])
                base12 = df[when <= t_last - pd.DateOffset(months=12)]
                base3 = df[when <= t_last - pd.DateOffset(months=3)]
                if not base12.empty:
                    row["change"] = f"{(last['value'] / base12.iloc[-1]['value'] - 1) * 100:+.2f}% y/y"
                elif not base3.empty:
                    row["change"] = f"{((last['value'] / base3.iloc[-1]['value']) ** 4 - 1) * 100:+.2f}% 3m annualised"
                else:
                    row["change"] = f"{(last['value'] / prev['value'] - 1) * 100:+.2f}% vs previous"
            else:
                row["change"] = fmt(last["value"] - prev["value"]) if abs(last["value"]) >= 1000 else f"{last['value'] - prev['value']:+.2f}"
        rows[label] = row
    return pd.DataFrame(rows).T


def industry_movers(asof: date, n: int = 5, exchange: str = "NASDAQ") -> pd.DataFrame:
    """Best and worst industries by average one-day change on the latest trading day."""
    for back in range(0, 6):
        d = asof - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        df = frame("industry-performance-snapshot", date=d.isoformat())
        if df.empty:
            continue
        df = df[df["exchange"] == exchange] if "exchange" in df and (df["exchange"] == exchange).any() else df
        g = df.groupby("industry")["averageChange"].mean().sort_values(ascending=False).round(2)
        t = pd.concat([g.head(n), g.tail(n)]).to_frame("avg_change_%")
        t.attrs["asof"] = d
        return t
    return pd.DataFrame()


def company_card(symbol: str, price: float | None = None) -> dict:
    """All FMP tables for one company. Each part is independent; a failure leaves it empty."""
    card: dict[str, object] = {"symbol": symbol}

    def part(name, fn, default):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log.warning("FMP %s %s skipped: %s: %s", symbol, name, type(e).__name__, _SECRET.sub("apikey=***", str(e))[:140])
            return default

    card["next"], card["track"] = part("earnings", lambda: earnings_track(symbol), ({}, pd.DataFrame()))
    card["street"], card["rating_changes"] = part("street", lambda: street_view(symbol, price), (pd.DataFrame(), pd.DataFrame()))
    card["quality"] = part("quality", lambda: quality(symbol), pd.DataFrame())
    card["income"] = part("income", lambda: income_quarters(symbol), pd.DataFrame())
    card["forward"] = part("estimates", lambda: forward_estimates(symbol), pd.DataFrame())
    card["products"] = part("product segments", lambda: segments(symbol, "product"), pd.DataFrame())
    card["geography"] = part("geographic segments", lambda: segments(symbol, "geo"), pd.DataFrame())
    card["filings"] = part("filings", lambda: filings(symbol), pd.DataFrame())
    card["ratings_trend"] = part("ratings trend", lambda: ratings_trend(symbol), pd.DataFrame())
    card["moves"] = part("price moves", lambda: price_moves(symbol), pd.DataFrame())
    card["peers"] = part("peers", lambda: peers(symbol), pd.DataFrame())
    card["segment_year"] = card["products"].attrs.get("fiscal_year") if isinstance(card["products"], pd.DataFrame) else None
    return card


# ---------------------------------------------------------------- market-wide tables
def earnings_calendar(start: date, end: date) -> pd.DataFrame:
    """Reports due in the window with EPS and revenue estimates."""
    df = frame("earnings-calendar", **{"from": start.isoformat(), "to": end.isoformat()})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def sector_snapshot(asof: date, exchange: str = "NASDAQ") -> pd.DataFrame:
    """Average one-day change and P/E per sector on the latest trading day at or before asof."""
    for back in range(0, 6):
        d = asof - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        perf = frame("sector-performance-snapshot", date=d.isoformat())
        if perf.empty:
            continue
        perf = perf[perf["exchange"] == exchange] if "exchange" in perf and (perf["exchange"] == exchange).any() else perf
        t = perf.groupby("sector")["averageChange"].mean().to_frame("avg_change_%")
        pe = frame("sector-pe-snapshot", date=d.isoformat())
        if not pe.empty:
            pe = pe[pe["exchange"] == exchange] if "exchange" in pe and (pe["exchange"] == exchange).any() else pe
            t["pe"] = pe.groupby("sector")["pe"].mean()
        t = t.sort_values("avg_change_%", ascending=False).round(2)
        t.attrs["asof"] = d
        return t
    return pd.DataFrame()


def daily_closes(symbols: list[str], start: date) -> pd.DataFrame:
    """Daily closes for equities, FX majors (EURUSD, USDCHF, NZDUSD ...) and commodities (GCUSD ...)."""
    out = {}
    for s in symbols:
        df = frame("historical-price-eod/light", symbol=s, **{"from": start.isoformat(), "to": date.today().isoformat()})
        if not df.empty and {"date", "price"} <= set(df.columns):
            out[s] = df.set_index(pd.to_datetime(df["date"]))["price"].sort_index()
    return pd.DataFrame(out)
