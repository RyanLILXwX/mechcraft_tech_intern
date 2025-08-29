# Binance Downloader

A robust Python script for downloading **Spot Kline (candlestick)** data from the Binance API. It supports multiple symbols, configurable time windows, retry and backoff handling for rate limits, and outputs clean **CSV/Parquet** datasets.

---

## Features

* Fetch Kline (candlestick) data from **Binance Spot REST API**.
* Supported intervals: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`.
* Robust pagination (up to **1000 candles per request**).
* Handles rate limiting (HTTP 429) with **exponential backoff**.
* Time window options:

  * Absolute time: e.g., `--start 2024-08-01 --end 2024-08-10`
  * Relative time: e.g., `--start 7d` or `--start 12h`
  * Shortcut: `--days 7` for the last 7 days
* Outputs per-symbol **CSV/Parquet** files with UTC timestamps.
* Optional quick gap check to detect missing candles.

---

## Requirements

Python 3.9+ and the following dependencies:

```bash
pip install requests pandas
```

If you want **Parquet output**:

```bash
pip install pyarrow   # or fastparquet
```

---

## Usage

### Command line

```bash
python binance_downloader.py [OPTIONS]
```

### Examples

1. Fetch **last 7 days of 1-minute candles** for multiple symbols:

   ```bash
   python binance_downloader.py --symbols BTCUSDT ETHUSDT BNBUSDT ADAUSDT --interval 1m --days 7 --out ./data
   ```

2. Fetch candles between **specific dates**:

   ```bash
   python binance_downloader.py --symbols BTCUSDT --interval 1m --start 2024-08-01 --end 2024-08-10 --out ./data --format parquet
   ```

3. Fetch candles for the **last 12 hours**:

   ```bash
   python binance_downloader.py --symbols ETHUSDT --interval 5m --start 12h --out ./data
   ```

---

## Arguments

### Supported Command Line Arguments

* `--symbols`: Specify the trading pairs to fetch, multiple values are allowed, e.g. `BTCUSDT ETHUSDT`. Default: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `ADAUSDT`.

* `--interval`: Kline interval, e.g. `1m`, `3m`, `5m`, `15m`, `30m`, `1h`. Default: `1m`.

* `--start`: Start time, can be an absolute time (e.g. `2024-08-01` or `2024-08-01T00:00Z`) or a relative time (e.g. `7d` for 7 days ago, `12h` for 12 hours ago).

* `--end`: End time. Default: current UTC time.

* `--days`: Shortcut to specify the most recent N days of data (mutually exclusive with `--start/--end`).

* `--out`: Output directory. Default: `./data`.

* `--format`: Output format, either `csv` or `parquet`. Default: `csv`.

* `--sleep`: Waiting time (in seconds) between requests. Default: `0.5`. It’s recommended not to set this too low to avoid rate limits.

* `--retries`: Maximum number of retries in case of errors. Default: `5`.

* `--timeout`: Timeout (in seconds) for a single request. Default: `20`.

---

## Output Files

* One file per symbol.
* Naming convention:

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

* `open_time` and `close_time` are **UTC timestamps**.
* Prices/volumes are floats; `trades` is integer.

---

## Gap Check

After fetching, the script performs a **simple gap check** per symbol.
It prints the first 20 detected gaps (if any) to the console:

```
[INFO] Detected potential gaps (maintenance/listing gaps). This is informational.
 symbol              gap_before  missing_candles
 BTCUSDT 2024-08-05T12:00:00+00:00               3
```

---

## Programmatic Usage

**(Recommended)** You can import functions inside Python, e.g.:

```python
from binance_downloader import run_binance_downloader

run_pipeline()
```
Or you can also use the certain functions to get the dataframe.

```python
from binance_downloader import fetch_klines, FetchConfig

cfg = FetchConfig(symbols=["BTCUSDT"], interval="1m", days=1, out_dir="./data")
df = fetch_klines(cfg)
print(df.head())
```

---

## Notes

* Binance API limit: max **1000 candles per request**. The script auto-paginates until your `--end`.

* `User-Agent` is set to `"binance-downloader/1.0 (https://github.com/RyanLIL-XwX/Mechcraft-Tech.git)"`. You can replace it with your project or repo link (not mandatory).

* If **Parquet** fails (missing pyarrow/fastparquet), the script falls back to CSV automatically.