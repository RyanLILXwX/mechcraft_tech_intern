# Binace Data Pipeline - (binance_pipeline.py)

This python file provides a modular pipeline to fetch, process, and prepare Binance Spot market data for machine learning. 

---

## Workflow Overview

**Downloader (`binance_downloader.py`)**
   Fetch raw kline data from Binance API. Handles retries, gap detection, and saves to disk (CSV/Parquet).

**Feature Engineering (`feature_engineering.py`)**
   Load raw OHLCV data, compute technical indicators (RSI, MACD, Bollinger Bands, ATR, rolling volatility, volume features), and build supervised targets.

**Data Split (`get_data.py`)**
   Split processed features into train/validation/test sets. Exports ready-to-use datasets for machine learning models.

**Pipeline Orchestration (`binance_pipeline.py`)**
   Main entry point that chains all steps:
   - Run downloader
   - Run feature engineering
   - Run data split
   - Save outputs in structured directories (`data/`, `processed_data/`, etc.)

---

## binance_downloader.py

### Arguments

- `--symbols`: Trading pairs (e.g., `BTCUSDT ETHUSDT`). Default: `BTCUSDT ETHUSDT BNBUSDT ADAUSDT`.

- `--interval`: (1m,3m,5m,15m,30m,1h) Kline interval. Default: `1m`.

- `--start` / `--end`: Time window. Accepts ISO date (`2024-08-01`), datetime (`2024-08-01T00:00Z`), or relative (`7d`, `12h`). Mutually exclusive with `--days`.

- `--days`(int): Fetch the last N days of data (alternative to start/end).

- `--out`(PATH): Output directory for raw files. Default: `./data`.

- `--format`(csv or parquet): File format. Default: `csv`.

- `--sleep`(float): Sleep between requests (default: `0.5`).

- `--retries`(int): Max retries on errors (default: `5`).

- `--timeout`(int): HTTP timeout seconds (default: `20`).

---

### Command Line Usage

Run the pipeline end-to-end:

```bash
python binance_downloader.py [OPTIONS]
```

#### Command line used right now

```bash
python binance_downloader.py --symbols BTCUSDT ETHUSDT BNBUSDT ADAUSDT --interval 1m --days 7 --out ./data
```

#### Example usage

#### 1. Fetch & process last 3 days of BTC/ETH at 1m resolution

```bash
python binance_pipeline.py --symbols BTCUSDT ETHUSDT --interval 1m --days 3 --out ./data --format parquet
```

#### 2. Fetch ADA/BNB between explicit dates, save as CSV

```bash
python binance_pipeline.py --symbols ADAUSDT BNBUSDT --interval 5m --start 2024-08-01 --end 2024-08-05 --out ./data --format csv
```

#### 3. Multi-symbol 15m klines for the last 7 days

```bash
python binance_pipeline.py --symbols BTCUSDT ETHUSDT BNBUSDT --interval 15m --days 7 --out ./data --format parquet
```

#### 4. Fetch last 7 days of 1-minute candles** for multiple symbols:

   ```bash
   python binance_downloader.py --symbols BTCUSDT ETHUSDT BNBUSDT ADAUSDT --interval 1m --days 7 --out ./data
   ```

#### 5. Fetch candles between **specific dates**:

   ```bash
   python binance_downloader.py --symbols BTCUSDT --interval 1m --start 2024-08-01 --end 2024-08-10 --out ./data --format parquet
   ```

#### 6. Fetch candles for the **last 12 hours**:

   ```bash
   python binance_downloader.py --symbols ETHUSDT --interval 5m --start 12h --out ./data
   ```

---

### Output Files

- One file per symbol.
- Naming convention:

  ```
  SYMBOL_INTERVAL_STARTDATE_ENDDATE.csv
  ```

  Example:

  ```
  BTCUSDT_1m_20240801_20240810.csv
  ```

Each file contains:

```
symbol, open_time, open, high,low, close,volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
```

- `open_time` and `close_time` are **UTC timestamps**.
- Prices/volumes are floats; `trades` is integer.

---

### Notes

- Binance API limit: max **1000 candles per request**. The script auto-paginates until your `--end`.

- `User-Agent` is set to `"binance-downloader/1.0 (https://github.com/RyanLILXwX/Mechcraft-Tech.git)"`. You can replace it with your project or repo link (not mandatory).

- If **Parquet** fails (missing pyarrow/fastparquet), the script falls back to CSV automatically.

---

## feature_engineering.py