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

The economic calendar comes from direct REST calls (`mnp/rest.py`): FRED's `releases/dates` API with your FRED key, and upcoming Treasury auctions from FiscalData with no key. OpenBB's own FRED calendar scrapes the FRED website and times out, so it is not used. Known limits, verified live on 2026-11-10: FRED lists releases that hold daily series (such as "FOMC Press Release") on every day, so those are filtered out and central-bank decision days come from the dated list in `config.yaml → regimes`. FiscalData's upcoming-auctions feed can lag by weeks, in which case no auction events are added and a warning is logged. IMF PortWatch's public server now requires a token, so the chokepoint section is off by default. When several week types apply (say CPI week plus an NVDA print), the first rule in `config.yaml → regimes` picks the template and the header lists the others. Earnings weeks use FMP's earnings calendar. MD&A text comes from SEC filings (free). Transcripts need a paid FMP plan and are skipped otherwise. API keys are masked in all log output. The notebook accepts common alternative secret names; see `ALIASES` in its second cell.
In Colab, store keys under Secrets (🔑). The notebook copies them into the environment and OpenBB.

## Your own data
`mnp/medallion.py` lets the pipeline read datasets from your private store through whatever
connector cell your notebook already uses. Register the reader and say which dataset plays which role:

```python
medallion.register(read, tables, datasets={"calendar": "...", "carry": "...", "candles": "..."})
```

| Role | Used for |
|---|---|
| `calendar` | Week detection and the "This week's scheduled events" table. `RATE_DECISION` rows make a central-bank week; high-impact `inflation` / `labour` / `auction` rows make a bond week |
| `carry` | FX carry table (policy-rate differentials) |
| `candles` | Daily closes for pairs listed under `universe.fx_minio` |

No addresses, bucket names or dataset names live in this repository. If the calendar has no `actual`
values, the report shows forecast and previous but cannot compute a surprise.

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
  - Its accuracy comes from one 70/15/15 split with no purge gap and no buy-and-hold baseline, so treat it as an upper bound.
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
