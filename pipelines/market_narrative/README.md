# Market Narrative Pipeline (`mnp`)

A research pipeline that runs separately from `openbb_platform`. It uses OpenBB only as a data
library, adds NLP and fundamentals on top, and writes **report templates that change with the week's regime**.

```
OpenBB (prices, news, calendars, fundamentals, Fed rates) ─┐
MinIO (FX pairs OpenBB doesn't cover) ─────────────────────┤
                                                           ▼
            regime.detect ── central_bank | bond_week | earnings_week | default
                                                           │
 FinBERT tone ─┐                                           │
 ModernBERT ───┴─► driver attribution (rates / inflation / earnings / metals / fx / geopolitics)
                                                           │
 cross-asset: returns, vol, rolling-corr shifts vs anchor (metals, yields, FX)
                                                           │
 AcyLa/multi-asset-predictive-model (adapter) ─────────────┤
                                                           ▼
                  templates/{earnings,macro,default}.md.j2 → reports/*.md (+ .html)
```

## Run
Colab: open `notebooks/colab_quickstart.ipynb`. Locally:
```bash
pip install -r requirements.txt
python -m mnp.run --out reports            # add --no-nlp / --no-model to skip heavy parts
```

## API keys
| Key | Needed? | Unlocks |
|---|---|---|
| none: HuggingFace public models, yfinance, IMF, ECB, Federal Reserve | — | prices, company news, all NLP, AcyLa model, IMF macro + PortWatch, EUR reference FX (ECB), SEC MD&A, Treasury curve |
| `FRED_API_KEY` | **recommended** | US release calendar (bond/Fed week detection without FMP), real yields, breakevens, credit spreads, USD index, claims |
| `FMP_API_KEY` | **recommended** | earnings calendar (works on the basic plan). Economic calendar and transcripts need a higher FMP plan (HTTP 402 otherwise) |
| your private-store credentials | **recommended** | whatever your own connector cell needs; unlocks your economic calendar, FX carry and extra FX pairs |
| `HF_TOKEN` | optional | higher HuggingFace download rate limits |
| `BENZINGA_API_KEY` | optional | world/macro news feed |

The economic calendar is the first of these that returns rows (`config.yaml → econ_calendar_providers`): your own calendar dataset, then Benzinga's economic calendar (`BENZINGA_API_KEY`; release times, consensus, prior, actual), then FRED's `releases/dates` API as a last resort. Only one is used, because two calendars side by side list every release twice. Treasury auctions are always added on top, from FiscalData and, when that is stale, TreasuryDirect (no key; the TreasuryDirect call has not been run live yet). OpenBB's own FRED calendar scrapes the FRED website and times out, so it is not used. Known limits: FRED gives dates without times and calls every dataset it hosts a "release", so only the release names listed under `config.yaml → calendar.fred_releases` count; central-bank decision days come from the dated list in `config.yaml → regimes`. Benzinga's response does not state a timezone; US Eastern is assumed. Anything missing or stale in a run is listed under "Data notes for this run" at the end of the report. IMF PortWatch's public server now requires a token, so the chokepoint section is off by default. When several week types apply (say CPI week plus an NVDA print), the first rule in `config.yaml → regimes` picks the template and the header lists the others. Earnings weeks use FMP's earnings calendar. MD&A text comes from SEC filings (free). Transcripts need a paid FMP plan and are skipped otherwise. API keys are masked in all log output. The notebook accepts common alternative secret names; see `ALIASES` in its second cell.
In Colab, store keys under Secrets (🔑). The notebook copies them into the environment and OpenBB.

## Your own data
`mnp/medallion.py` lets the pipeline read datasets from your private store through whatever
connector cell your notebook already uses. Register the reader; each role is matched to one of your
datasets by name pattern, and you can override any of them:

```python
medallion.register(read, tables)                                # returns {role: dataset}
medallion.register(read, tables, datasets={"calendar": "..."})  # choose one yourself; "" switches a role off
```

| Role | Used for |
|---|---|
| `calendar` | Week detection and the "This week's scheduled events" table. `RATE_DECISION` rows make a central-bank week; high-impact `inflation` / `labour` / `auction` rows make a bond week |
| `carry` | FX carry table (policy-rate differentials) |
| `candles` | Daily closes for pairs listed under `universe.fx_minio` |
| `yields` | "Global 10Y yields": level, 1-week and 1-month change, spread to the US and 2s10s per country. A country whose series stopped updating is left out |
| `positioning` | "FX positioning (COT)": latest institutional bias per pair, with the report's age |

No addresses, bucket names or credentials live in this repository, and nothing is downloaded until a section needs it. If the calendar has no `actual`
values, the report shows forecast and previous but cannot compute a surprise.

## Per-pair session briefs
`mnp/pairs.py` writes one brief per FX pair (`config.yaml → pairs.list`) from your own data store
only: `pairs.run()` in the notebook, or `python -c "from mnp import pairs; pairs.run()"` after
registering a reader. Each brief has:

| Part | What it says | From |
|---|---|---|
| State | One of `event_ahead`, `range_spent`, `early_session`, `range_open`; picks the opening paragraph | calendar + range used |
| Today's range | Day low and high, how much of the expected range is used, how often each extreme held from a state like this, where the other side would be if one holds | `candles` (hourly), `range_forecast` or ADR20 |
| This week | Week low and high so far, how often they stood from this day on, and the plain weekday table | `candles` |
| Fundamentals | Releases for the two currencies, 10Y yield gap and its change, carry, COT positioning, regime label | `calendar`, `yields`, `carry`, `positioning`, `regime` |

The day rolls at 17:00 New York. Every hold rate is printed beside two comparisons: the plain
rate at that hour, and **the same state on a volatility-matched random walk** (the pair's own
bars with each bar's direction randomised). The second one matters: on synthetic noise with no
structure at all, a low that was 7 hours old with 70-100% of the range used "held" 89% of the
time against a 73% plain rate. So a high rate is a real probability, useful for placing a target,
but only a rate outside the random-walk band says the market is doing something worth
researching. The briefs never call a direction, and nothing in them has been through a
registered test with a control: they are context for a setup and a source of hypotheses.

`pairs.run()` also writes `reports/pairs/pairs_latest.json`, one flat row per pair.

## Serving it to agents (OpenBB MCP)
`mnp/api.py` is a small read-only HTTP app. The OpenBB MCP server in this repository turns its
routes into tools:

```bash
pip install fastapi openbb-mcp-server
cd pipelines/market_narrative
MNP_READER=<module>:<read function> MNP_TABLES=<module>:<tables function> openbb-mcp --app mnp.api:app --port 8001
```

Tools: `mnp_pair_brief`, `mnp_pairs`, `mnp_calendar`, `mnp_global_yields`, `mnp_positioning`,
`mnp_health`. It must run on a machine that can reach your data store, and it reads through the
function you name in `MNP_READER`, so no address is stored here. Pair answers carry `stale`. The
routes are tested; starting them under `openbb-mcp` has not been run yet.

## Swiss National Bank
`mnp/snb.py` reads the SNB data portal (data.snb.ch, no key) and adds a Switzerland / CHF section:
policy rate and SARON, **sight deposits** (the usual intervention proxy), the SNB's own quarterly
**FX transactions**, the CHF effective exchange rate, a Confederation yield curve interpolated from
individual bonds, official rates of the Fed / ECB / BoE / BoJ against the SNB, and the currency
split of the SNB's reserves. All seven cubes were verified live on 2026-11-10. Each table carries its
own "as of" date and is flagged when older than its release rhythm allows. The portal's older yield
cubes (`rendoblid`, `rendopar`) stopped in July 2025 and are deliberately not used.

## FMP
FMP serves different endpoints per plan and answers 402 for the rest, so sources are chosen from
evidence: `mnp/fmp_probe.py` asks each endpoint once (`probe()` for 46 core ones,
`probe(extended=True)` for about 85 more) and reports which your key can use.

`mnp/fmp.py` is built on the ones verified on a Starter-level plan (2026-11-10):

| In the report | From |
|---|---|
| **Earnings card** for each focus name reporting this week: next date with EPS and revenue expected, actual-vs-estimate history, quarterly results, consensus for coming fiscal years, analyst ratings and price targets, rating counts by month, upgrades and downgrades, quality scores (Altman Z, Piotroski), valuation and DCF, revenue by product and geography, peers, recent filings | earnings, income-statement, analyst-estimates, grades*, price-target*, financial-scores, ratios-ttm, key-metrics-ttm, discounted-cash-flow, revenue-*-segmentation, stock-peers, sec-filings-search |
| **US data scorecard**: latest released print of CPI, unemployment, payrolls, claims, retail sales, sentiment (the policy rate is in the FRED table; FMP's monthly figure disagreed with it) | economic-indicators |
| **Sector moves and valuation**, best and worst industries | sector-/industry-performance-snapshot, sector-pe-snapshot |

Cards appear in every report type, so an earnings name is not lost in a macro week. Not on that
plan, and so not used: news, economic calendar, transcripts, COT, insider search, intraday bars.
Plan limits respected: at most 5 rows on earnings and statements, annual estimates and segments only.

## Benzinga
Benzinga is the only keyed news source here (FMP's news is not on a Starter-level plan), so it is what
the tone and driver models read and what any fine-tuning set would be built from.
`mnp/benzinga_probe.py` checks the key before you train: endpoints served, how far back news goes,
whether articles have bodies, which channels and tags they carry, FX / commodity / rates coverage,
and whether the economic calendar has `actual` and `consensus`. First live run 2026-11-10: news
goes back 10 years, 86% of articles have a body, and the economic calendar had actual and consensus
for 37% of events, so it now feeds the report's calendar (`rest.benzinga_calendar`).

## Where to extend
| Want to… | Edit |
|---|---|
| Add assets / MinIO FX pairs | `config.yaml → universe` |
| Change what counts as a bond / earnings / Fed week | `config.yaml → regimes` (first match wins) |
| Add a news driver | `config.yaml → drivers` (one descriptive sentence per driver) |
| New report type (e.g. OPEC week) | add a `regimes` rule + `mnp/templates/<name>.md.j2` (reuse the `_common` macros) |
| Change which ETFs get model forecasts | `config.yaml → predictive.tickers` (14 covered ETFs only) |

## Notes
- **Predictive model (`AcyLa/multi-asset-predictive-model`):** per-ETF LightGBM (+ optional LSTM)
  models forecast returns 1, 5 and 21 days ahead. The pipeline runs the repo's own feature code on OpenBB OHLCV.
  Caveats from reviewing its code:
  - The published Sharpe overlaps h-day returns, which inflates it by about √h. The report shows `sharpe_adj` = Sharpe/√h instead (SPY 21-day: 7.2 → ~1.6).
  - Its accuracy comes from one 70/15/15 split with no purge gap and no buy-and-hold baseline, so treat it as an upper bound. The report adds a baseline itself: `up_share`, how often the asset rose over that horizon in the fetched history. A signal is shown only when test accuracy is at least 55% and above `up_share`.
  - **GLD, SLV and USO test below 50% directional accuracy**, so they display as "no skill". Use the cross-asset and NLP sections for metals.
  - The model files are pickles, so loading them runs code from that repo. Pin `predictive.revision` to a reviewed commit.
- **Models (all pre-trained, no training needed):**
  | Job | Model |
  |---|---|
  | News / document tone | `beethogedeon/Modern-FinBERT-large` (ModernBERT-large fine-tuned on Financial PhraseBank) + `yiyanghkust/finbert-tone` as a second opinion |
  | Headline → driver | `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (zero-shot; edit labels in `config.yaml → drivers`) |
  | Embeddings / long-doc search | `nomic-ai/modernbert-embed-base` (8k context, needs the task prefix) |

  Optional upgrade: `mnp.finetune` trains `answerdotai/ModernBERT-large` heads (raw MLM, per its
  model card it must be fine-tuned) into `heads/tone` and `heads/driver`. When those folders
  exist they override the defaults.
- **Earnings weeks** add management-language tone for the latest call transcript and MD&A
  (FMP). Each document is scored chunk by chunk, and the report quotes the most cautious passage.
- **Flash Attention 2** is used automatically on Ampere+ GPUs if `flash-attn` is installed. On a T4 it falls back to fp16.
- **Missing provider keys** (FMP, Benzinga) don't break a run. Those sections come out empty and
  regime detection falls back to `default`.
- **Units:** yfinance `^TNX` is in percent and Fed `treasury_rates` is in decimals. Yield
  series are differenced, not log-returned.
