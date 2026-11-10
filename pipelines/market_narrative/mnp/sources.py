"""Data access: OpenBB Platform for everything it covers, MinIO for FX pairs it doesn't."""

from __future__ import annotations

import io
import logging
import os
import re
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger(__name__)


_SECRET = re.compile(r"(?i)(api_?key|token|secret|password)=[^&\s]+")


def _obb():
    from openbb import obb  # imported lazily so the module loads without openbb installed

    obb.user.preferences.output_type = "dataframe"
    try:
        obb.user.preferences.request_timeout = 120  # FRED's release calendar is slow
    except Exception:  # noqa: BLE001
        pass
    return obb


def _safe(fn, *a, retries: int = 1, **kw) -> pd.DataFrame:
    """Call an OpenBB endpoint; return an empty frame on failure (missing key, plan limit, outage).

    Retries timeouts once. Credentials are masked before anything is logged.
    """
    for attempt in range(retries + 1):
        try:
            df = fn(*a, **kw)
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            msg = _SECRET.sub(r"\1=***", str(e)).strip()
            if "timeout" in msg.lower() and attempt < retries:
                continue
            first = next((ln for ln in msg.splitlines() if ln.strip()), msg)
            log.warning("%s(provider=%s) unavailable: %s", getattr(fn, "__name__", "call"),
                        kw.get("provider", "default"), first[:200])
            return pd.DataFrame()
    return pd.DataFrame()


def closes(symbols: list[str], start: date, provider: str, kind: str = "equity") -> pd.DataFrame:
    """Wide frame of close prices, one column per symbol."""
    obb = _obb()
    fn = {
        "equity": obb.equity.price.historical,
        "currency": obb.currency.price.historical,
        "index": obb.index.price.historical,
    }[kind]
    out = {}
    for s in symbols:
        df = _safe(fn, symbol=s, start_date=start, provider=provider)
        if not df.empty and "close" in df:
            out[s] = df["close"]
    wide = pd.DataFrame(out)
    wide.index = pd.to_datetime(wide.index)
    return wide


def ohlcv(symbol: str, start: date, provider: str) -> pd.DataFrame:
    """Full OHLCV for one symbol (lower-case columns, date index)."""
    df = _safe(_obb().equity.price.historical, symbol=symbol, start_date=start, provider=provider)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


def minio_closes(pairs: list[str], start: date) -> pd.DataFrame:
    """Daily FX closes from your private hourly candles (needs medallion.register)."""
    from . import medallion

    if pairs and not medallion.available():
        log.warning("fx_minio pairs skipped: run your connector cell and medallion.register(...) first")
        return pd.DataFrame()
    return medallion.fx_daily_closes(pairs, start)


def news(symbols: list[str], provider: str, limit: int = 50) -> pd.DataFrame:
    obb = _obb()
    frames = [_safe(obb.news.company, symbol=s, provider=provider, limit=limit) for s in symbols]
    frames.append(_safe(obb.news.world, provider="benzinga", limit=limit))  # optional; empty w/o key
    df = pd.concat([f for f in frames if not f.empty], axis=0) if any(not f.empty for f in frames) else pd.DataFrame()
    if df.empty:
        return df
    df = df.reset_index().rename(columns={"index": "date"})
    text_col = "title" if "title" in df else df.columns[1]
    df["text"] = df[text_col].fillna("") + ". " + df.get("text", pd.Series("", index=df.index)).fillna("").astype(str).str[:500]
    return df.drop_duplicates(subset=[text_col])


def earnings_calendar(start: date, end: date, provider: str) -> pd.DataFrame:
    return _safe(_obb().equity.calendar.earnings, start_date=start, end_date=end, provider=provider)


def econ_calendar(start: date, end: date, providers: list[str] | str,
                  auction_terms: list[str] | None = None, currencies: list[str] | None = None,
                  impacts: list[str] | None = None) -> pd.DataFrame:
    """Union of calendars. `medallion` is your own calendar dataset (see medallion.py);
    `fred_rest` and `treasury_auctions` are direct REST (see rest.py); any other name is an
    OpenBB `economy.calendar` provider (fmp, tradingeconomics)."""
    from . import rest

    providers = [providers] if isinstance(providers, str) else providers
    frames = []
    for p in providers:
        try:  # a calendar source must never stop the run
            if p == "medallion":
                from . import medallion

                df = medallion.calendar(start, end, currencies, tuple(impacts or ("high",)))
            elif p == "fred_rest":
                df = rest.fred_release_calendar(start, end)
            elif p == "treasury_auctions":
                df = rest.treasury_auctions(start, end, auction_terms)
            else:
                df = _safe(_obb().economy.calendar, start_date=start, end_date=end, provider=p)
                df = df.reset_index() if not df.empty else df
        except Exception as e:  # noqa: BLE001
            log.warning("calendar source %s skipped: %s: %s", p, type(e).__name__, _SECRET.sub(r"\1=***", str(e))[:200])
            continue
        if not df.empty:
            frames.append(df.assign(calendar_source=p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fred(series: list[str], start: date) -> pd.DataFrame:
    """Wide frame of FRED series (needs fred_api_key)."""
    if not series:
        return pd.DataFrame()
    df = _safe(_obb().economy.fred_series, symbol=",".join(series), start_date=start, provider="fred")
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


_METRIC_COLS = ["market_cap", "pe_ratio", "forward_pe", "peg_ratio", "enterprise_to_ebitda", "earnings_growth",
                "revenue_growth", "gross_margin", "operating_margin", "profit_margin", "return_on_equity",
                "debt_to_equity", "beta"]
_INCOME_COLS = ["total_revenue", "gross_profit", "operating_income", "net_income", "diluted_earnings_per_share"]


def fundamentals(symbol: str, provider: str = "yfinance") -> dict[str, pd.DataFrame]:
    obb = _obb()
    metrics = _safe(obb.equity.fundamental.metrics, symbol=symbol, provider=provider)
    if not metrics.empty:
        metrics = metrics[[c for c in _METRIC_COLS if c in metrics]].T.rename(columns=lambda _: symbol)
        if "market_cap" in metrics.index:
            metrics.loc["market_cap"] = metrics.loc["market_cap"] / 1e9
            metrics = metrics.rename(index={"market_cap": "market_cap_bn"})
    income = _safe(obb.equity.fundamental.income, symbol=symbol, provider=provider, period="quarter", limit=5)
    if not income.empty:
        idx = income["period_ending"] if "period_ending" in income else income.index
        cols = [c for c in _INCOME_COLS if c in income]
        income = income[cols].set_axis(pd.to_datetime(idx).astype(str), axis=0).sort_index()
        for c in cols:
            if c != "diluted_earnings_per_share":
                income[c] = income[c] / 1e9
        income = income.rename(columns=lambda c: c if c.endswith("per_share") else f"{c}_bn")
        if "total_revenue_bn" in income:
            income["revenue_qoq"] = income["total_revenue_bn"].pct_change()
    est = _safe(obb.equity.estimates.consensus, symbol=symbol, provider=provider)
    if not est.empty:
        est = est.drop(columns=[c for c in ("symbol", "currency") if c in est]).T.rename(columns=lambda _: symbol)
    return {"metrics": metrics, "income": income, "estimates": est}


def documents(symbol: str, transcript_provider: str = "fmp", mdna_provider: str = "sec") -> dict[str, str]:
    """Latest earnings-call transcript (FMP, paid plans) and MD&A text (SEC, free). "" if unavailable."""
    obb = _obb()
    out = {}
    for name, fn, prov in (("transcript", obb.equity.fundamental.transcript, transcript_provider),
                           ("mdna", obb.equity.fundamental.management_discussion_analysis, mdna_provider)):
        df = _safe(fn, symbol=symbol, provider=prov)
        out[name] = str(df["content"].iloc[-1]) if not df.empty and "content" in df else ""
    return out


def imf_search(query: str) -> pd.DataFrame:
    """Find IMF indicator symbols (no key). E.g. imf_search("exchange rate")."""
    return _safe(_obb().economy.available_indicators, provider="imf", query=query)


def imf_indicators(symbols: list[str], countries: list[str], start: date) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return _safe(_obb().economy.indicators, provider="imf", symbol=",".join(symbols),
                 country=",".join(countries), start_date=start)


def chokepoints(names: list[str], start: date) -> pd.DataFrame:
    """IMF PortWatch daily transit volumes through shipping chokepoints (oil / metals / geopolitics).

    Fetched one at a time so one bad name doesn't void the rest; `bab_el-mandeb_strait` is
    spelled with a hyphen or an underscore depending on the installed openbb-imf release.
    """
    fn = _obb().economy.shipping.chokepoint_volume
    frames = []
    for n in names:
        variants = [n] + ([n.replace("el_mandeb", "el-mandeb")] if "el_mandeb" in n else
                          [n.replace("el-mandeb", "el_mandeb")] if "el-mandeb" in n else [])
        for v in variants:
            # Some openbb-imf releases fail with KeyError 'features' when a date filter is passed,
            # so fall back to the unfiltered history and trim it here.
            df = _safe(fn, provider="imf", chokepoint=v, start_date=start)
            if df.empty:
                df = _safe(fn, provider="imf", chokepoint=v)
            if not df.empty:
                df = df.reset_index()
                if "date" in df:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[df["date"].dt.date >= start]
                frames.append(df)
                break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def window(today: date | None = None, ahead: int = 7) -> tuple[date, date]:
    today = today or date.today()
    return today, today + timedelta(days=ahead)
