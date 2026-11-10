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

## Where to extend
| Want to… | Edit |
|---|---|
| Add assets / MinIO FX pairs | `config.yaml → universe` |
| Change what counts as a bond / earnings / Fed week | `config.yaml → regimes` (first match wins) |
| Add a news driver | `config.yaml → drivers` (one descriptive sentence per driver) |
| New report type (e.g. OPEC week) | add a `regimes` rule + `mnp/templates/<name>.md.j2` (reuse the `_common` macros) |
| Wire the predictive model | run `Predictive(...).inspect()`, then set `FEATURE_COLUMNS` / `predict()` in `mnp/predictive.py` |

## Notes
- **Predictive model:** the adapter detects whether the repo ships transformers, joblib/pkl,
  ONNX, or torch weights. Its input features are not yet known, so the report shows a
  "not wired" note until `predict()` matches the model card. Loading a pickle runs code, so
  only load it from a repo you trust.
- **ModernBERT-large** is a base encoder, not one tuned for sentence similarity. Driver attribution
  ranks headlines well, but the similarity scores are not calibrated. Fine-tune it later on labelled
  headlines for sharper attribution.
- **Missing provider keys** (FMP, Benzinga) don't break a run. Those sections come out empty and
  regime detection falls back to `default`.
- **Units:** yfinance `^TNX` is in percent and Fed `treasury_rates` is in decimals. Yield
  series are differenced, not log-returned.
