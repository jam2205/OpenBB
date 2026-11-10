"""Orchestrator: data -> regime -> NLP -> cross-asset -> model -> templated report."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from . import correlations as corr
from . import regime as rg
from . import report, sources

log = logging.getLogger("mnp")
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())


def _fmt_bp(x) -> str:
    """x in percentage points (0.10 -> +10bp)."""
    return "n/a" if pd.isna(x) else f"{x * 100:+.0f}bp"


def run(config: str | Path | None = None, asof: date | None = None, out_dir: str | Path = "reports",
        use_nlp: bool = True, use_model: bool = True) -> Path:
    cfg = load_config(config)
    asof = asof or date.today()
    start = asof - timedelta(days=cfg["lookback_days"])
    u, ob = cfg["universe"], cfg["openbb"]

    # 1. Prices: OpenBB for equities/metals/bonds/FX; MinIO for uncovered FX
    eq_syms = u["equities"] + list(u["metals"]) + list(u["bonds"])
    prices = pd.concat([
        sources.closes(eq_syms, start, ob["price_provider"], "equity"),
        sources.closes(u["fx_openbb"], start, ob["price_provider"], "currency"),
        sources.minio_closes(u.get("fx_minio", []), cfg["minio"]),
    ], axis=1).sort_index()
    prices = prices.rename(columns={**u["metals"], **u["bonds"]})
    yield_cols = [v for k, v in u["bonds"].items() if k.startswith("^")]
    rets = corr.returns(prices, yield_cols)
    anchor = u["equities"][0]

    # 2. Regime from calendars
    w0, w1 = sources.window(asof)
    econ = sources.econ_calendar(w0, w1, ob["econ_calendar_provider"])
    earn = sources.earnings_calendar(w0, w1, ob["earnings_provider"])
    regime = rg.detect(cfg, econ, earn)
    log.info("Regime: %s", regime["name"])

    # 3. NLP over news
    driver_summary = pd.DataFrame()
    scored = pd.DataFrame()
    if use_nlp:
        from .nlp import NLP

        nlp = NLP({**cfg["models"]})
        news = sources.news(u["equities"], ob["news_provider"])
        scored = nlp.attribute_drivers(news, cfg["drivers"])
        driver_summary = NLP.driver_summary(scored)
    dom = driver_summary.iloc[0] if not driver_summary.empty else None
    top = []
    if dom is not None:
        sub = scored[scored.driver == dom.driver].nlargest(5, "driver_sim")
        title_col = "title" if "title" in sub else "text"
        top = [{"tone": r.tone, "title": getattr(r, title_col)} for r in sub.itertuples()]

    # 4. Cross-asset
    shift = corr.corr_shift(rets, anchor, cfg["corr_window"])
    y10 = u["bonds"].get("^TNX")
    gold = u["metals"].get("GC=F")
    gold_y10 = "n/a"
    if y10 in rets and gold in rets:
        r, f = rets[gold].tail(cfg["corr_window"]).corr(rets[y10]), rets[gold].corr(rets[y10])
        gold_y10 = f"{r:+.2f} / {f:+.2f}"
    y10_move = _fmt_bp(prices[y10].ffill().diff(5).iloc[-1]) if y10 in prices else "n/a"

    rates = sources._safe(sources._obb().fixedincome.government.treasury_rates,
                          start_date=asof - timedelta(days=14), provider="federal_reserve")
    curve = "n/a"
    if not rates.empty and {"year_2", "year_10"} <= set(rates.columns):
        last = rates.iloc[-1]
        curve = _fmt_bp((last["year_10"] - last["year_2"]) * 100)  # provider returns decimals
        rates = rates[[c for c in ("month_3", "year_2", "year_5", "year_10", "year_30") if c in rates]].tail(5)

    # 5. Fundamentals for reporters (earnings regime)
    funds = {}
    if regime["name"] == "earnings_week":
        funds = {s: sources.fundamentals(s, ob["price_provider"]) for s in regime["reporters"]}
    earnings_tone = float(driver_summary.set_index("driver").net_tone.get("earnings", 0.0)) if not driver_summary.empty else 0.0

    # 6. Predictive model
    predictions, model_note = None, "model disabled"
    if use_model:
        try:
            from .predictive import Predictive, build_features

            m = Predictive(cfg["models"]["predictive"])
            predictions = m.predict(build_features(rets, driver_summary)).tail(5)
        except Exception as e:  # noqa: BLE001
            model_note = f"predictive model not wired yet: {e}"

    ctx = dict(
        asof=asof.isoformat(), regime=regime, anchor=anchor, corr_window=cfg["corr_window"],
        snapshot=corr.snapshot(prices, rets), corr_shift=shift,
        corr_flags=shift.index[shift["flag"]].tolist() if not shift.empty else [],
        driver_summary=driver_summary,
        dominant_driver=dom.driver if dom is not None else "n/a",
        dominant_tone=float(dom.net_tone) if dom is not None else 0.0,
        top_headlines=top, fundamentals=funds, earnings_tone=earnings_tone,
        y10_move=y10_move, rates=rates, curve_2s10s=curve, gold_y10=gold_y10,
        predictions=predictions, model_note=model_note,
    )
    path = report.render(ctx, regime["template"], out_dir)
    log.info("Report written: %s", path)
    return path


def main():
    p = argparse.ArgumentParser(description="Market Narrative Pipeline")
    p.add_argument("--config")
    p.add_argument("--asof", type=date.fromisoformat)
    p.add_argument("--out", default="reports")
    p.add_argument("--no-nlp", action="store_true")
    p.add_argument("--no-model", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(run(a.config, a.asof, a.out, not a.no_nlp, not a.no_model))


if __name__ == "__main__":
    main()
