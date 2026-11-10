"""Direct REST calls for sources where the OpenBB wrapper is unreliable or needs a paid plan.

- FRED release calendar: OpenBB's `economy.calendar(provider="fred")` scrapes the
  fred.stlouisfed.org website page by page and times out. The official API does it in one call.
- Upcoming Treasury auctions: U.S. Treasury FiscalData, no key.
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


def _get(url: str, params: dict, timeout: int = 30, retries: int = 2) -> dict:
    import requests

    err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
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


def fred_release_calendar(start: date, end: date, api_key: str | None = None, max_days: int = 2) -> pd.DataFrame:
    """Scheduled FRED releases between start and end: columns date, event, release_id."""
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
    return df[["date", "event", "release_id"]].assign(country="US").reset_index(drop=True)


def treasury_auctions(start: date, end: date, terms: list[str] | None = None) -> pd.DataFrame:
    """Upcoming Treasury coupon auctions in the window, as calendar events.

    `terms` filters by security term (e.g. ["10-Year", "30-Year"]); bills auction every week,
    so they are excluded unless you list their terms.
    """
    try:
        data = _get(f"{FISCALDATA}/v1/accounting/od/upcoming_auctions",
                    {"page[size]": 500, "sort": "-record_date"})
    except Exception as e:  # noqa: BLE001
        log.warning("treasury auctions unavailable: %s", e)
        return pd.DataFrame()
    df = pd.DataFrame(data.get("data", []))
    if df.empty or "auction_date" not in df:
        return pd.DataFrame()
    latest = pd.to_datetime(df["record_date"]).max()
    if (pd.Timestamp(start) - latest).days > 7:
        # Verified 2026-11-10: this dataset can run weeks behind, so "upcoming" may be empty.
        log.warning("treasury auctions: FiscalData snapshot is stale (latest %s); no auction events this run",
                    latest.date())
    df = df[pd.to_datetime(df["record_date"]) == latest].copy()
    df["date"] = pd.to_datetime(df["auction_date"], errors="coerce")
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    term = df["security_term"].astype(str) if "security_term" in df else pd.Series("", index=df.index)
    kind = df["security_type"].astype(str) if "security_type" in df else pd.Series("", index=df.index)
    if terms:
        mask = term.str.contains("|".join(re.escape(t) for t in terms), case=False, regex=True)
    else:
        mask = kind.str.lower() != "bill"
    # .to_numpy(bool): an empty object-dtype mask would be read by pandas as a column list.
    keep = mask.to_numpy(dtype=bool)
    df, term, kind = df[keep], term[keep], kind[keep]
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": df["date"], "event": "Treasury Auction: " + term + " " + kind, "country": "US"})
    if "offering_amt" in df:
        out["offering_amt"] = df["offering_amt"]
    return out.sort_values("date").reset_index(drop=True)
