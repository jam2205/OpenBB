"""The pipeline as an HTTP app, so the OpenBB MCP server can hand it to agents as tools.

    cd pipelines/market_narrative
    openbb-mcp --app mnp.api:app --port 8001           # tools: mnp_pair_brief, mnp_pairs, ...

(Use the module form `mnp.api:app`, not the file path: the package uses relative imports.)

Run it on a machine that can reach your data store. The app needs a reader, exactly like the
notebook does; name it in the environment (nothing about your store is written in this file):

    MNP_READER=<module>:<function>     read(name, **kw) -> DataFrame          (required)
    MNP_TABLES=<module>:<function>     tables() -> names, or a list of dicts  (optional)
    MNP_ROLES='{"calendar": "..."}'    JSON role -> dataset overrides         (optional)
    PYTHONPATH must let Python import that module.

Every answer carries `asof` and `data_age_h`, and pair answers carry `stale`: an agent must not
read an old bar as the current market. All tools are read-only and call no model.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from datetime import date, timedelta

import pandas as pd

from . import medallion, pairs

_cfg: dict | None = None


def _load(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _names(tables_fn):
    def names():
        out = tables_fn()
        if isinstance(out, pd.DataFrame):
            out = out.to_dict("records")
        return [next((str(o[k]) for k in ("dataset", "name", "id", "table") if k in o), "") if isinstance(o, dict) else str(o) for o in out]

    return names


def connect() -> dict:
    """Register the reader named in the environment (once). Returns the role map."""
    if medallion.available():
        return medallion.roles()
    spec = os.environ.get("MNP_READER")
    if not spec:
        raise RuntimeError("MNP_READER is not set: name the function that reads your datasets, as <module>:<function>")
    tables = _names(_load(os.environ["MNP_TABLES"])) if os.environ.get("MNP_TABLES") else None
    return medallion.register(_load(spec), tables, json.loads(os.environ.get("MNP_ROLES", "{}")))


def _config() -> dict:
    global _cfg
    if _cfg is None:
        from .run import load_config

        _cfg = load_config(os.environ.get("MNP_CONFIG"))
    return _cfg


def _safe(v):
    """JSON-safe: NaN and NaT become null, timestamps become ISO text, frames become records."""
    if isinstance(v, pd.DataFrame):
        return [_safe(r) for r in v.reset_index().to_dict("records")]
    if isinstance(v, dict):
        return {str(k): _safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(x) for x in v]
    if isinstance(v, (pd.Timestamp, date)):
        return None if pd.isna(v) else v.isoformat()
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def create_app():
    from fastapi import FastAPI, HTTPException, Query

    app = FastAPI(title="Market Narrative Pipeline", version="0.1", description="Per-pair session briefs and macro context from your own data store.")

    def ready():
        try:
            return connect()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"data store not connected: {type(e).__name__}: {str(e)[:200]}") from e

    @app.get("/mnp/health", tags=["mnp"], operation_id="mnp_health")
    def health() -> dict:
        """Is the data store connected, and which dataset plays which role."""
        try:
            return {"connected": True, "roles": connect(), "pairs": _config().get("pairs", {}).get("list", [])}
        except Exception as e:  # noqa: BLE001
            return {"connected": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    @app.get("/mnp/pair_brief", tags=["mnp"], operation_id="mnp_pair_brief")
    def pair_brief(pair: str = Query(..., description="FX pair, e.g. EURUSD"),
                   markdown: bool = Query(False, description="Also return the full written brief")) -> dict:
        """Where one FX pair stands in its trading day and week: range used against the expected
        range, how often the day's low and high held from a state like this (beside the plain
        rate and a random walk), the week's extremes, and the calendar, yield gap, carry,
        positioning and regime label for its two currencies. Descriptive frequencies, never a
        direction call. Check `stale` before using it."""
        ready()
        b = pairs.brief(pair, cfg=_config())
        out = pairs.row(b)
        if "error" in b:
            raise HTTPException(404, b["error"])
        out["calendar"], out["regime_table"] = _safe(b["calendar"]), _safe(b["regime"])
        out["week_table"] = _safe(b["week"].get("table")) if b.get("week") else None
        out["sessions"] = _safe(b["day"]["sessions"])
        if markdown:
            from . import report

            out["markdown"] = report.render_text({"b": b, "p": b["day"], "w": b["week"]}, "pair.md.j2")
        return _safe(out)

    @app.get("/mnp/pairs", tags=["mnp"], operation_id="mnp_pairs")
    def pairs_table(pairs_csv: str | None = Query(None, description="Comma-separated pairs; default is the configured list")) -> dict:
        """One row per pair with the same numbers as mnp_pair_brief: a scan of which pairs have
        spent their range, which extremes are likely in, and where releases are still ahead."""
        ready()
        cfg, shared, rows = _config(), {}, []
        for p in ([x.strip() for x in pairs_csv.split(",") if x.strip()] if pairs_csv else cfg.get("pairs", {}).get("list", [])):
            try:
                rows.append(pairs.row(pairs.brief(p, cfg=cfg, shared=shared)))
            except Exception as e:  # noqa: BLE001
                rows.append({"pair": p.upper(), "error": f"{type(e).__name__}: {str(e)[:120]}"})
        return _safe({"generated_at": pd.Timestamp.now(tz="UTC"), "rows": rows})

    @app.get("/mnp/calendar", tags=["mnp"], operation_id="mnp_calendar")
    def calendar(days: int = Query(7, ge=1, le=31), currencies: str | None = Query(None, description="Comma-separated, e.g. USD,EUR"),
                 impacts: str = Query("high", description="Comma-separated: high, medium, low")) -> dict:
        """Scheduled releases from your calendar dataset for the coming days, with forecast and previous."""
        ready()
        today = date.today()
        df = medallion.calendar(today, today + timedelta(days=days), [c.strip().upper() for c in currencies.split(",")] if currencies else None,
                                tuple(i.strip() for i in impacts.split(",")))
        return _safe({"from": today, "days": days, "events": df})

    @app.get("/mnp/global_yields", tags=["mnp"], operation_id="mnp_global_yields")
    def global_yields() -> dict:
        """10Y government yields by country: level, 1-week and 1-month change, gap to the US, 2s10s."""
        ready()
        t = medallion.global_yields(date.today())
        asof = t.attrs.get("asof")
        return _safe({"asof": asof, "data_age_days": (date.today() - asof).days if asof else None, "rows": t.rename_axis("country") if not t.empty else t})

    @app.get("/mnp/positioning", tags=["mnp"], operation_id="mnp_positioning")
    def positioning(pairs_csv: str | None = Query(None, description="Comma-separated pairs; default is the six largest biases")) -> dict:
        """Latest COT-derived institutional bias per FX pair, with the age of the report."""
        ready()
        return _safe({"rows": medallion.positioning([x.strip().upper() for x in pairs_csv.split(",")] if pairs_csv else None)})

    return app


try:   # `openbb-mcp --app mnp.api:app` looks for a module-level `app`
    app = create_app()
except ImportError:   # fastapi not installed: the rest of the package still imports
    app = None
