# Feature engineering of Binance Data

This module builds a **clean, feature-rich OHLCV dataset** for machine-learning on crypto candlesticks.
It loads raw K-line CSVs, validates schema, standardizes timestamps, engineers technical indicators (MAs, RSI, MACD, Bollinger Bands, ATR, rolling volatility, volume stats), and constructs **supervised targets** (binary/multiclass/regression) over future horizons.
Use the single convenience function `get_features_dataset(...)` for an **end-to-end** "load -> features -> targets -> (optional) save" workflow.

---

## Installation

### Required packages

`pyarrow` is needed only if you plan to save Parquet outputs.

```bash
pip install pandas numpy pyarrow
```

---

## Expected Input (Raw CSV)

Your CSV should contain at least the following columns (case-sensitive):

- `open_time` (UTC parseable timestamp)
- `close_time` (UTC parseable timestamp)
- `symbol`
- `open`, `high`, `low`, `close`
- `volume`

Optional but supported (recommended):

- `quote_volume`, `taker_buy_base`, `taker_buy_quote`, `trades`

**Notes**

- Rows are automatically sorted by `open_time` (renamed to `ts` internally).
- Duplicate timestamps are dropped.
- Basic value sanity checks are applied (e.g., `low ≤ min(open,close,high)`; `high ≥ max(open,close,low)`).

---

## What the Workflow Produces

- A **time-indexed** DataFrame with column `ts` (UTC) and:

  - Cleaned OHLCV columns
  - Engineered features (see API list below)
  - Supervised targets (`target_*`) aligned with features
- Drop_nan rows implied by rolling windows are **dropped** automatically.
- The **last `max(horizons)` rows** are removed to avoid look-ahead bias during target creation.

---

## Functions

### Utilities and Loading

- `infer_freq_alias`
- `load_and_prepare`
- `find_time_gaps`

### Core Feature Builders

- `add_price_returns_and_spread`
- `exponential_moving_average`
- `add_moving_averages`
- `add_rsi`
- `add_macd`
- `add_bbands`
- `add_atr`
- `add_roll_volatility`
- `add_volume_features`

### Composite Pipeline

- `WindowsConfig`
- `add_all_features`

### Targets & Dataset Assembly

- `future_return`
- `make_supervised_targets`
- `build_features`
- `get_features_dataset`

---

## Parameters

- **interval**: e.g., `"1m"`, `"5m"`, `"1h"`; used for sanity/metadata.
- **expected_symbol**: set it to assert the CSV contains the expected trading pair.
- **horizons**: prediction steps ahead (bars); e.g., `(1, 5, 15, 30)`.
- **task**:
  - `"binclass"` -> `target_up_<h>` in {0,1}
  - `"multiclass"` -> `target_cls_<h>` in {-1,0,1} (uses `thresh`)
  - `"reg"` -> `target_ret_<h>` continuous
- **thresh**: classification threshold for “flat” band or up/down cutoff.
- **out_path**: file path (saves exactly there) or **directory**(auto-names from input stem).
- **file_format**: `"csv"` or `"parquet"`.

---

## Notes

- **Missing columns**: Ensure required OHLCV fields exist and are numeric where applicable.
- **Timezone**: Timestamps are parsed as **UTC**.
- **NaNs at head**: Expected due to rolling drop_nan. They’re trimmed automatically.
- **NaNs at tail**: Expected if you inspect before the module drops the last `max(horizons)` rows.