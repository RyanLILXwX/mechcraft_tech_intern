# Binance Spot Kline Downloader

A robust command line interface script to download Binance Spot candlestick (kline) data into tidy CSV/Parquet files. It handles pagination, rate limits with retries and exponential backoff, gap detection, data typing, UTC timestamps, and per-symbol file outputs suitable for downstream feature engineering and ML pipelines.

---

## Features

- Multi-symbol download in one run
- Intervals supported: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`
- Time window by `--days` **or** explicit `--start` / `--end` (mutually exclusive with `--days`)
- Relative times: `Xd` (days) and `Xh` (hours) in `--start`/`--end` (e.g., `7d`, `12h`)
- Pagination using `startTime` only for stability
- Automatic rate-limit handling (HTTP 429) + exponential backoff
- Gap detection summary printed to console
- Per-symbol outputs with deduplication by `(symbol, open_time)`
- File naming: `{SYMBOL}_{INTERVAL}_{YYYYMMDDstart}_{YYYYMMDDend}.{csv|parquet}`

---

## Functions

- `parse_time`
- `to_ms`
- `request_with_retry`
- `fetch_symbol_klines`
- `fetch_klines`
- `check_gaps`
- `save_df`
- `parse_arguments`
- `get_binance_dataset`
- **Dataclass:** `FetchConfig`

___

## Requirements

- **Python**: 3.9+
- **Packages**:
    - `requests`
    - `pandas`
    - Optional (for Parquet output): `pyarrow` or `fastparquet`

---

## Installation

```bash
pip install requests pandas
# Optional, download for parquet format
pip install pyarrow
# or
pip install fastparquet
```

Place `binance_downloader.py` anywhere in your project (e.g., at repo root or in a `scripts/` folder).

---

## Command Line Interface Options

- `--symbols` (SYMBOL [SYMBOL ...])
  Symbols to fetch (e.g., `BTCUSDT ETHUSDT`). Defaults to `BTCUSDT ETHUSDT BNBUSDT ADAUSDT`.

- `--interval` (1m,3m,5m,15m,30m,1h)
  Kline interval (default: `1m`).

- `--start` (str)
  ISO/date string (e.g., `2024-08-01`, `2024-08-01T00:00Z`) **or** relative (`7d`, `12h`).

- `--end` (str)
  ISO/date string or relative (`7d`, `12h`). Defaults to “now” if omitted.

- `--days` (int)
  Fetch the last N days **instead of** `--start/--end`. Mutually exclusive with `--start/--end`.

- `--out` (PATH)
  Output directory (default: `./data`).

- `--format` (csv or parquet)
  Output file format (default: `csv`). If Parquet engine is missing, it will fall back to CSV.

- `--sleep` (float)
  Seconds to sleep between API calls (default: `0.5`).

- `--retries` (int)
  Max retries on errors (default: `5`).

- `--timeout` (int)
  HTTP timeout seconds per request (default: `20`).

## Outputs

- One file per symbol in the chosen directory (created if missing).
- Filename pattern:
  - `BTCUSDT_1m_20240801_20240807.csv`
  - `ETHUSDT_5m_20240801_20240807.parquet`

- Printed console info includes:
  - Fetch ranges per symbol
  - Row counts per symbol
  - Optional gap detection table (informational)
  - Write confirmations per file

---

## Notes & Tips

- **Time zone**: All times are stored as **UTC**.
- **Gap detection**: Printed to console for awareness (maintenance windows, listing gaps). No file output for gaps by default.
- **Parquet**: Install `pyarrow` or `fastparquet` to enable; otherwise the script falls back to CSV and prints a warning.
- **Rate limits**: The script honors `Retry-After` when present and uses exponential backoff. Adjust `--sleep` and `--retries` as needed.