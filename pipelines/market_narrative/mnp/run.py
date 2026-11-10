"""Orchestrator: data -> regime -> NLP -> cross-asset -> model -> templated report."""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import (
    correlations as corr,
    regime as rg,
    report,
    sources,
)

log = logging.getLogger("mnp")
ROOT = Path(__file__).resolve().parent.parent


_REQUIRED = ("models", "predictive", "universe", "openbb", "calendar", "fred", "fmp", "snb", "imf", "regimes", "drivers",
             "corr_window", "lookback_days")


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:  # a missing section silently drops a report block, so say it out loud
        log.warning("config.yaml is missing sections %s: the matching report blocks will be empty", missing)
    return cfg


def _guard(name: str, fn, default):
    """Run an optional report section; on any error log one line and return the default."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        msg = sources._SECRET.sub(r"\1=***", str(e)).strip().splitlines()
        log.warning("section '%s' skipped: %s: %s", name, type(e).__name__, (msg[0] if msg else "")[:200])
        return default


def _rates_section(asof):
    """(table of the last sessions, 2s10s, 10Y change over five sessions, date of the last row).

    The weekly move is taken from this same table, so the line under it cannot disagree with it.
    """
    rates = sources._safe(sources._obb().fixedincome.government.treasury_rates,
                          start_date=asof - timedelta(days=21), provider="federal_reserve")
    curve, move, last_day = "n/a", None, None
    if not rates.empty and {"year_2", "year_10"} <= set(rates.columns):
        rates = rates.dropna(subset=["year_10"])
        last = rates.iloc[-1]
        curve = _fmt_bp((last["year_10"] - last["year_2"]) * 100)  # provider returns decimals
        if len(rates) > 5:
            move = _fmt_bp((last["year_10"] - rates["year_10"].iloc[-6]) * 100)
        last_day = str(rates.index[-1])[:10]
        rates = (rates[[c for c in ("month_3", "year_2", "year_5", "year_10", "year_30") if c in rates]].tail(6) * 100).round(2)
        rates.index = [str(i)[:10] for i in rates.index]
        rates.index.name = "date"
    return rates, curve, move, last_day


class _Notes(logging.Handler):
    """Collects this run's warnings so the report can list what is missing or stale."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.items: list[str] = []

    def emit(self, record):
        msg = sources._SECRET.sub(r"\1=***", record.getMessage()).strip().splitlines()
        if msg and msg[0] not in self.items:
            self.items.append(msg[0][:300])


def _fred_section(cfg, prices, gold, start):
    fcfg = cfg.get("fred", {})
    fr = sources.fred(list(fcfg.get("series", {})), start)
    if fr.empty:
        return pd.DataFrame(), "n/a"
    ff = fr.ffill()
    last_date = ff.index[-1]

    def _chg(days):
        past = ff[ff.index <= last_date - timedelta(days=days)]
        return ff.iloc[-1] - past.iloc[-1] if not past.empty else pd.Series(index=ff.columns, dtype=float)

    tbl = pd.DataFrame({"name": pd.Series(fcfg["series"]), "last": ff.iloc[-1],
                        "chg_1w": _chg(7), "chg_1m": _chg(30)}).dropna(subset=["last"]).round(3)
    gold_real, gd = "n/a", fcfg.get("gold_driver")
    if gold and gd in fr and gold in prices:
        j = pd.concat([np.log(prices[gold]).diff(), fr[gd].diff()], axis=1, keys=["g", "r"]).dropna()
        if len(j) > cfg["corr_window"]:
            gold_real = f"{j.tail(cfg['corr_window']).corr().iloc[0, 1]:+.2f} / {j.corr().iloc[0, 1]:+.2f}"
    return tbl, gold_real


def _chokepoint_section(cfg, asof):
    choke = sources.chokepoints(cfg.get("imf", {}).get("chokepoints", []), asof - timedelta(days=90))
    if choke.empty or "chokepoint" not in choke or "date" not in choke:
        return pd.DataFrame()
    num = choke.select_dtypes("number").columns
    col = next((c for c in ("vessels_total", "capacity_total") if c in num), num[0] if len(num) else None)
    if col is None:
        return pd.DataFrame()
    piv = choke.pivot_table(index="date", columns="chokepoint", values=col).sort_index()
    tbl = pd.DataFrame({"last_7d_avg": piv.tail(7).mean(), "prior_90d_avg": piv.mean()})
    tbl["vs_90d_pct"] = (tbl["last_7d_avg"] / tbl["prior_90d_avg"] - 1) * 100
    return tbl.round(1)


def _fundamentals_section(symbols, ob, nlp):
    funds = {}
    for sym in symbols:
        f = _guard(f"fundamentals {sym}", lambda: sources.fundamentals(sym, ob["price_provider"]),
                   {"metrics": pd.DataFrame(), "income": pd.DataFrame(), "estimates": pd.DataFrame()})
        docs = _guard(f"documents {sym}", lambda: sources.documents(
            sym, ob.get("transcript_provider", "fmp"), ob.get("mdna_provider", "sec")), {})
        empty = {"net_tone": float("nan"), "chunks": 0, "method": "n/a"}
        f["doc_tone"] = {k: (_guard(f"{k} tone {sym}", lambda v=v: nlp.document_tone(v), empty) if nlp else empty)
                         for k, v in docs.items()}
        funds[sym] = f
    return funds


def _with_baseline(pred: pd.DataFrame, bars: dict) -> pd.DataFrame:
    """Add the always-long baseline and withdraw signals that do not beat it.

    `up_share` is the share of h-day periods that rose in the price history fetched for this
    run. A model that always says "up" scores that on direction, so a test accuracy at or below
    it shows no skill, whatever its level. The baseline window is not the model's own test
    window (the model repo does not publish one), so read `edge_pp` as indicative.
    """
    from .predictive import SKILL_MIN

    def base(r):
        df = bars.get(r["ticker"])
        return corr.up_share(df["close"], int(r["horizon_d"])) if df is not None and not df.empty and "close" in df else np.nan

    p = pred.copy()
    p["up_share"] = p.apply(base, axis=1)
    p["edge_pp"] = (p["test_dir_acc"] - p["up_share"]) * 100
    weak = (p["test_dir_acc"] < SKILL_MIN) | (p["edge_pp"] <= 0)
    p = p.drop(columns="shown_signal")
    p["shown_signal"] = p["model_signal"].where(~weak, "no skill")   # last column: the one to read
    p.insert(2, "pred_return_%", p.pop("pred_return") * 100)
    for c, n in (("pred_return_%", 2), ("test_dir_acc", 2), ("up_share", 2), ("edge_pp", 1), ("sharpe_adj", 2)):
        p[c] = p[c].astype(float).round(n)
    return p


def _fmt_bp(x) -> str:
    """X in percentage points (0.10 -> +10bp)."""
    return "n/a" if pd.isna(x) else f"{x * 100:+.0f}bp"


def run(config: str | Path | None = None, asof: date | None = None, out_dir: str | Path = "reports",
        use_nlp: bool = True, use_model: bool = True) -> Path:
    cfg = load_config(config)
    notes = _Notes()
    logging.getLogger("mnp").addHandler(notes)
    try:
        return _run(cfg, asof, out_dir, use_nlp, use_model, notes)
    finally:
        logging.getLogger("mnp").removeHandler(notes)


def _run(cfg: dict, asof: date | None, out_dir, use_nlp: bool, use_model: bool, notes: _Notes) -> Path:
    report.quiet_progress()
    asof = asof or date.today()
    start = asof - timedelta(days=cfg["lookback_days"])
    u, ob = cfg["universe"], cfg["openbb"]

    # 1. Prices: OpenBB for equities/metals/bonds/FX; MinIO for uncovered FX
    eq_syms = u["equities"] + list(u["metals"]) + list(u["bonds"])
    prices = pd.concat([
        sources.closes(eq_syms, start, ob["price_provider"], "equity"),
        sources.closes(u["fx_openbb"], start, ob["price_provider"], "currency"),
        sources.closes(u.get("fx_ecb", []), start, "ecb", "currency"),
        _guard("fx_minio", lambda: sources.minio_closes(u.get("fx_minio", []), start), pd.DataFrame()),
    ], axis=1).sort_index()
    prices = prices.rename(columns={**u["metals"], **u["bonds"]})
    yield_cols = [v for k, v in u["bonds"].items() if k.startswith("^")]
    rets = corr.returns(prices, yield_cols)
    anchor = u["equities"][0]

    # 2. Regime from calendars
    w0, w1 = sources.window(asof)
    ccfg = cfg.get("calendar", {})
    econ = sources.econ_calendar(w0, w1, ob.get("econ_calendar_providers", ["medallion", "benzinga", "fred_rest"]),
                                 ob.get("auction_terms"), ccfg.get("currencies"), ccfg.get("impacts"),
                                 ob.get("econ_calendar_extra", ["treasury_auctions"]), ccfg.get("fred_releases"))
    earn = sources.earnings_calendar(w0, w1, ob["earnings_provider"])
    regime = _guard("regime", lambda: rg.detect(cfg, econ, earn, (w0, w1)),
                    {**cfg["regimes"][-1], "events": [], "reporters": [], "also": []})
    log.info("Regime: %s", regime["name"])

    def _calendar_table():
        if econ.empty:
            return pd.DataFrame()
        t = econ.copy()
        t["date"] = pd.to_datetime(t["date"], utc=True, errors="coerce")
        # A source that knows only the day (FRED, auctions) shows the day, not a made-up 00:00.
        dated_only = (t["date"].dt.hour == 0) & (t["date"].dt.minute == 0)
        t["when (UTC)"] = t["date"].dt.strftime("%a %d %b %H:%M").where(~dated_only, t["date"].dt.strftime("%a %d %b"))
        cols = [c for c in ("when (UTC)", "currency", "event", "impact", "forecast", "previous", "actual", "calendar_source") if c in t]
        t = t.sort_values("date")[cols].reset_index(drop=True)
        t = t.drop(columns=[c for c in ("forecast", "previous", "actual", "impact") if c in t and t[c].fillna("").astype(str).str.strip().eq("").all()])
        cap = ccfg.get("max_rows", 40)
        hidden = max(len(t) - cap, 0)
        if hidden and "impact" in t:   # over the cap: keep the highest-impact rows, still shown in time order
            rank = t["impact"].astype(str).str.lower().map({"high": 0, "medium": 1, "low": 2}).fillna(1)
            t = t.loc[rank.sort_values(kind="stable").index[:cap]].sort_index()
        t = t.head(cap).fillna("").astype(str).reset_index(drop=True)
        t.attrs["hidden"] = hidden
        return t

    calendar_tbl = _guard("calendar table", _calendar_table, pd.DataFrame())
    from . import medallion

    fx_pairs = u.get("fx_openbb", []) + u.get("fx_minio", [])
    carry_tbl = _guard("carry", lambda: medallion.carry(fx_pairs), pd.DataFrame())
    world_yields = _guard("global yields", lambda: medallion.global_yields(asof), pd.DataFrame())
    cot_tbl = _guard("positioning", lambda: medallion.positioning(fx_pairs), pd.DataFrame())

    # 3. NLP over news
    driver_summary = pd.DataFrame()
    scored = pd.DataFrame()
    nlp = None
    if use_nlp:
        from .nlp import NLP

        nlp = NLP({**cfg["models"]})

        def _news():
            news = sources.news(u["equities"], ob["news_provider"])
            sc = nlp.attribute_drivers(news, cfg["drivers"], cfg.get("zero_shot_threshold", 0.35))
            return sc, NLP.driver_summary(sc)

        scored, driver_summary = _guard("news NLP", _news, (pd.DataFrame(), pd.DataFrame()))
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
    rates, curve, y10_move, y10_day = _guard("rates", lambda: _rates_section(asof), (pd.DataFrame(), "n/a", None, None))
    if y10_move is None:   # no Treasury table: fall back to the market series in the snapshot
        y10_move = _fmt_bp(prices[y10].ffill().diff(5).iloc[-1]) if y10 in prices else "n/a"
        y10_day = str(prices[y10].dropna().index[-1])[:10] if y10 in prices and prices[y10].notna().any() else None
    fx_vs_y10 = {p: corr.pair_corr(rets, p, y10, cfg["corr_window"]) for p in u.get("fx_openbb", []) if p.endswith("JPY")}
    fx_vs_y10 = {p: v for p, v in fx_vs_y10.items() if v}
    anchor_y10 = corr.pair_corr(rets, anchor, y10, cfg["corr_window"])
    fred_tbl, gold_real = _guard("fred", lambda: _fred_section(cfg, prices, gold, start), (pd.DataFrame(), "n/a"))
    icfg = cfg.get("imf", {})
    imf_macro = _guard("imf", lambda: sources.imf_indicators(
        icfg.get("indicators", []), icfg.get("countries", []), asof - timedelta(days=730)), pd.DataFrame())
    choke_tbl = _guard("chokepoints", lambda: _chokepoint_section(cfg, asof), pd.DataFrame())
    snb_block = {}
    if cfg.get("snb", {}).get("enabled", True):
        from . import snb as snb_mod

        snb_block = _guard("snb", lambda: snb_mod.block(asof), {})

    # 5. Fundamentals for reporters. FMP cards whenever a focus name reports (any week type);
    #    the yfinance + document-tone block stays for earnings weeks.
    funds = {}
    if regime["name"] == "earnings_week":
        funds = _guard("fundamentals", lambda: _fundamentals_section(regime["reporters"], ob, nlp), {})
    fcfg = cfg.get("fmp", {})
    cards, us_data, sectors, industries = [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if fcfg.get("enabled", True):
        from . import fmp as fmp_mod

        if fmp_mod.available():
            names = list(dict.fromkeys(regime.get("reporters", []) + fcfg.get("always_cards", [])))[: fcfg.get("max_cards", 3)]
            px = prices.ffill().iloc[-1] if not prices.empty else pd.Series(dtype=float)
            cards = [_guard(f"fmp card {s_}", lambda s_=s_: fmp_mod.company_card(s_, float(px[s_]) if s_ in px and pd.notna(px[s_]) else None), None)
                     for s_ in names]
            cards = [c for c in cards if c]
            us_data = _guard("fmp us scorecard", lambda: fmp_mod.us_scorecard(asof, fcfg.get("us_indicators")), pd.DataFrame())
            sectors = _guard("fmp sectors", lambda: fmp_mod.sector_snapshot(asof), pd.DataFrame())
            industries = _guard("fmp industries", lambda: fmp_mod.industry_movers(asof), pd.DataFrame())
    earnings_tone = float(driver_summary.set_index("driver").net_tone.get("earnings", 0.0)) if not driver_summary.empty else 0.0

    # 6. Predictive model (AcyLa per-ETF ensembles; needs >= ~300 bars for its 200-day features)
    predictions, model_note = None, "model disabled"
    if use_model:
        try:
            from .predictive import COVERED, AcyLaModel

            mcfg = cfg["predictive"]
            m = AcyLaModel(cfg["models"]["predictive"], mcfg.get("revision"), mcfg.get("use_lstm", False))
            hist_start = asof - timedelta(days=mcfg.get("history_days", 900))
            targets = [t for t in mcfg["tickers"] if t in COVERED]
            bars = {t: sources.ohlcv(t, hist_start, ob["price_provider"]) for t in targets}
            predictions = m.predict_many(bars)
            model_note = "no predictions returned" if predictions.empty else ""
            if not predictions.empty:
                predictions = _with_baseline(predictions, bars)
        except Exception as e:  # noqa: BLE001
            model_note = f"predictive model unavailable: {e}"
            predictions = None

    ctx = dict(
        asof=asof.isoformat(), regime=regime, anchor=anchor, corr_window=cfg["corr_window"],
        snapshot=corr.snapshot(prices, rets, yield_cols), corr_shift=shift,
        corr_flags=shift.index[shift["flag"]].tolist() if not shift.empty else [],
        driver_summary=driver_summary,
        dominant_driver=dom.driver if dom is not None else "n/a",
        dominant_tone=float(dom.net_tone) if dom is not None else 0.0,
        top_headlines=top,
        driver_method=scored["driver_method"].iloc[0] if not scored.empty else "n/a", fundamentals=funds, earnings_tone=earnings_tone,
        y10_move=y10_move, y10_day=y10_day, fx_vs_y10=fx_vs_y10, anchor_y10=anchor_y10, notes=notes.items,
        world_yields=world_yields, world_yields_asof=world_yields.attrs.get("asof") if not world_yields.empty else None,
        positioning=cot_tbl, calendar_hidden=calendar_tbl.attrs.get("hidden", 0) if not calendar_tbl.empty else 0,
        rates=rates, curve_2s10s=curve, gold_y10=gold_y10,
        predictions=predictions, model_note=model_note,
        fred=fred_tbl, gold_real=gold_real, cards=cards, us_data=us_data, sectors=sectors, industries=industries,
        sectors_asof=sectors.attrs.get("asof") if not sectors.empty else None,
        sectors_exchange=sectors.attrs.get("exchange", "") if not sectors.empty else "", snb=snb_block, calendar_tbl=calendar_tbl, carry=carry_tbl,
        imf_macro=imf_macro.tail(20) if not imf_macro.empty else imf_macro, chokepoints=choke_tbl,
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
