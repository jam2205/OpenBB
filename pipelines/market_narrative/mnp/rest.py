"""Direct REST calls for sources where the OpenBB wrapper is unreliable or needs a paid plan.

- FRED release calendar: OpenBB's `economy.calendar(provider="fred")` scrapes the
  fred.stlouisfed.org website page by page and times out. The official API does it in one call.
- Upcoming Treasury auctions: U.S. Treasury FiscalData, no key; TreasuryDirect as a fallback when
  FiscalData's snapshot is stale.
- Benzinga economic calendar (key): release times, consensus, prior and actual for ~25 countries.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import date

import pandas as pd

log = logging.getLogger(__name__)
_SECRET = re.compile(r"(?i)(api_?key|token|secret|password)=[^&\s'\"]+")
FRED_API = "https://api.stlouisfed.org/fred"
FISCALDATA = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
TREASURYDIRECT = "https://www.treasurydirect.gov/TA_WS"


def _get(url: str, params: dict, timeout: int = 30, retries: int = 2) -> dict | list:
    import requests

    err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"accept": "application/json"})
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")  # 4xx: don't retry
            return r.json()
        except ValueError as e:
            err = e
            break
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(_SECRET.sub(r"\1=***", f"{url.rsplit('/', 2)[-2]}/{url.rsplit('/', 1)[-1]}: {err}"))


def fred_key() -> str | None:
    key = os.environ.get("FRED_API_KEY")
    if not key:
        try:
            from openbb import obb

            key = obb.user.credentials.fred_api_key
            key = key.get_secret_value() if hasattr(key, "get_secret_value") else key
        except Exception:  # noqa: BLE001
            key = None
    return key or None


def fred_release_calendar(start: date, end: date, api_key: str | None = None, max_days: int = 2,
                          include: list[str] | None = None) -> pd.DataFrame:
    """Scheduled FRED releases between start and end: columns date, event, release_id.

    `include` is a list of exact release names to keep (case-insensitive). FRED lists every
    dataset it hosts as a "release" (research papers, World Development Indicators, six CPI
    variants), so without it the calendar is mostly noise. FRED gives dates only, no times.
    """
    api_key = api_key or fred_key()
    if not api_key:
        log.warning("fred_rest calendar skipped: no FRED_API_KEY")
        return pd.DataFrame()
    rows, offset = [], 0
    try:
        while True:
            data = _get(f"{FRED_API}/releases/dates", {
                "api_key": api_key, "file_type": "json",
                "realtime_start": start.isoformat(), "realtime_end": end.isoformat(),
                "include_release_dates_with_no_data": "true",  # required to see future dates
                "sort_order": "asc", "limit": 1000, "offset": offset,
            })
            batch = data.get("release_dates", [])
            rows += batch
            offset += len(batch)
            if len(batch) < 1000:
                break
    except Exception as e:  # noqa: BLE001
        log.warning("fred_rest calendar unavailable: %s", e)
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"release_name": "event"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    # FRED lists a "release date" every day for releases that hold daily series (e.g. the fed funds
    # target lives under "FOMC Press Release"). Those are not events: keep only releases that
    # appear on at most `max_days` distinct days in the window.
    days = df.groupby("event")["date"].transform("nunique")
    df = df[days <= max_days]
    if include:
        df = df[df["event"].astype(str).str.strip().str.lower().isin([i.strip().lower() for i in include])]
    df = df.drop_duplicates(["date", "event"])
    return df[["date", "event", "release_id"]].assign(country="US").reset_index(drop=True)


def _auction_events(df: pd.DataFrame, start: date, end: date, terms: list[str] | None,
                    date_col: str, term_col: str, type_col: str, amt_col: str) -> pd.DataFrame:
    if df.empty or date_col not in df:
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    term = df[term_col].astype(str) if term_col in df else pd.Series("", index=df.index)
    kind = df[type_col].astype(str) if type_col in df else pd.Series("", index=df.index)
    if terms:
        mask = term.str.contains("|".join(re.escape(t) for t in terms), case=False, regex=True)
    else:
        mask = kind.str.lower() != "bill"
    # .to_numpy(bool): an empty object-dtype mask would be read by pandas as a column list.
    keep = mask.to_numpy(dtype=bool)
    df, term, kind = df[keep], term[keep], kind[keep]
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": df["date"].dt.normalize(), "event": "Treasury Auction: " + term + " " + kind,
                        "country": "US", "currency": "USD"})
    if amt_col in df:
        out["offering_amt"] = df[amt_col]
    return out.drop_duplicates(["date", "event"]).sort_values("date").reset_index(drop=True)


def _treasurydirect_auctions(start: date, end: date, terms: list[str] | None) -> pd.DataFrame:
    """TreasuryDirect's list of announced, not yet auctioned securities (no key).

    Written from the public API description and NOT yet run live: if the answer has another
    shape this returns nothing and the report says the auction calendar is unavailable.
    """
    data = _get(f"{TREASURYDIRECT}/securities/upcoming", {"format": "json"})
    rows = data if isinstance(data, list) else []
    return _auction_events(pd.DataFrame(rows), start, end, terms, "auctionDate", "securityTerm", "securityType", "offeringAmount")


def treasury_auctions(start: date, end: date, terms: list[str] | None = None) -> pd.DataFrame:
    """Upcoming Treasury coupon auctions in the window, as calendar events.

    `terms` filters by security term (e.g. ["10-Year", "30-Year"]); bills auction every week,
    so they are excluded unless you list their terms.
    """
    out, stale = pd.DataFrame(), None
    try:
        data = _get(f"{FISCALDATA}/v1/accounting/od/upcoming_auctions",
                    {"page[size]": 500, "sort": "-record_date"})
        df = pd.DataFrame(data.get("data", []))
        if not df.empty and "auction_date" in df:
            latest = pd.to_datetime(df["record_date"]).max()
            if (pd.Timestamp(start) - latest).days > 7:
                stale = latest.date()   # verified 2026-11-10: this dataset can run weeks behind
            df = df[pd.to_datetime(df["record_date"]) == latest]
            out = _auction_events(df, start, end, terms, "auction_date", "security_term", "security_type", "offering_amt")
    except Exception as e:  # noqa: BLE001
        log.info("treasury auctions: FiscalData unavailable: %s", e)
    if out.empty:
        try:
            out = _treasurydirect_auctions(start, end, terms)
        except Exception as e:  # noqa: BLE001
            log.info("treasury auctions: TreasuryDirect unavailable: %s", e)
    if out.empty and stale:
        log.warning("Treasury auction calendar: FiscalData was last updated %s and TreasuryDirect gave nothing, "
                    "so auctions are missing from the events table. Check treasurydirect.gov.", stale)
    return out


# ---------------------------------------------------------------- Benzinga economic calendar
_BZ_CCY = {"USA": "USD", "EUR": "EUR", "DEU": "EUR", "FRA": "EUR", "ITA": "EUR", "ESP": "EUR", "JPN": "JPY", "GBR": "GBP",
           "CHE": "CHF", "CAN": "CAD", "AUS": "AUD", "NZL": "NZD", "CHN": "CNY", "MEX": "MXN", "BRA": "BRL", "KOR": "KRW",
           "ZAF": "ZAR", "IND": "INR", "NOR": "NOK", "SWE": "SEK", "SGP": "SGD", "HKG": "HKD", "HUN": "HUF", "IDN": "IDR"}


def _blank(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["", "none", "nan", "<na>", "null"])


def benzinga_calendar(start: date, end: date, currencies: list[str] | None = None, min_importance: int = 3,
                      api_key: str | None = None) -> pd.DataFrame:
    """Benzinga economic calendar: date (UTC), event, country, currency, impact, forecast,
    previous, actual, surprise.

    Endpoint and field names were verified on a live key (probe run 2026-11-10: 1000 events in 30
    days, 37% with both actual and consensus, importance values 1/2/3/5). Importance 5 is shown
    as "high" and 3 as "medium": a week at importance 3 and above ran to 80 rows, mostly speeches.
    The response does not state a timezone; US Eastern is assumed.
    """
    key = api_key or os.environ.get("BENZINGA_API_KEY")
    if not key:
        return pd.DataFrame()
    rows, page = [], 0
    while page < 5:
        data = _get("https://api.benzinga.com/api/v2.1/calendar/economics", {
            "token": key, "parameters[date_from]": start.isoformat(), "parameters[date_to]": end.isoformat(),
            "pagesize": 1000, "page": page})
        batch = data.get("economics", []) if isinstance(data, dict) else (data or [])
        rows += batch
        page += 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    if df.empty or "date" not in df or "event_name" not in df:
        return pd.DataFrame()
    when = pd.to_datetime(df["date"].astype(str) + " " + df.get("time", pd.Series("", index=df.index)).fillna("").astype(str),
                          errors="coerce")
    when = when.fillna(pd.to_datetime(df["date"], errors="coerce"))
    df["date"] = when.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    df["currency"] = df["country"].map(_BZ_CCY) if "country" in df else None
    imp = pd.to_numeric(df.get("importance"), errors="coerce").fillna(0)
    df = df[imp >= min_importance]
    if currencies:
        df = df[df["currency"].isin(currencies)]
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()
    imp = imp.loc[df.index]

    def num(x) -> str:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return str(x)
        return f"{f:,.0f}" if f == round(f) and abs(f) >= 1000 else f"{f:,.3f}".rstrip("0").rstrip(".")

    def val(col, unit):
        v = df[col].map(num).where(~_blank(df[col]), "") if col in df else pd.Series("", index=df.index)
        u = df[unit].astype(str).str.strip().where(~_blank(df[unit]), "") if unit in df else pd.Series("", index=df.index)
        u = u.where(u.isin(["%", ""]) | (u.str.len() <= 1), " " + u)   # "195K" and "0.3%", but "-865,000 Barrels"
        return (v + u).where(v != "", "")

    # Germany, France, Italy and Spain all map to EUR: say whose number it is.
    member = df["country"].where(df["country"].isin(["DEU", "FRA", "ITA", "ESP"]), "").astype(str) if "country" in df else pd.Series("", index=df.index)
    df = df.assign(event_name=(member + " " + df["event_name"].astype(str)).str.strip())

    period = df["event_period"].astype(str).where(~_blank(df["event_period"]), "") if "event_period" in df else ""
    out = pd.DataFrame({
        "date": df["date"], "currency": df["currency"], "country": df.get("country"),
        "event": (df["event_name"].astype(str) + (" (" + period + ")").where(period != "", "")) if isinstance(period, pd.Series) else df["event_name"].astype(str),
        "impact": imp.map(lambda i: "high" if i >= 4 else "medium" if i >= 3 else "low"), "importance": imp,
        "forecast": val("consensus", "consensus_t"), "previous": val("prior", "prior_t"), "actual": val("actual", "actual_t"),
    })
    if "actual" in df and "consensus" in df:
        out["surprise"] = (pd.to_numeric(df["actual"], errors="coerce") - pd.to_numeric(df["consensus"], errors="coerce")).round(3)
    return out.drop_duplicates(["date", "currency", "event"]).sort_values(["date", "currency"]).reset_index(drop=True)
