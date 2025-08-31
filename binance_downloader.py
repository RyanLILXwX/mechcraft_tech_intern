from pathlib import Path # Processing file and directory paths
import argparse # Command line argument parsing
import time # Sleep throttling and retry waiting
import sys
from dataclasses import dataclass # Concisely define configuration data structure
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone # Handling time zones and time windows
import requests # Send the HTTP requests
import pandas as pd # Handling tabular data and timestamps

BINANCE_BASE = "https://api.binance.com" # Binance Spot API Base Domain
KLINES_EP = "/api/v3/klines" # Spot K-line interface path
UA = "binance-downloader/1.0 (https://github.com/RyanLIL-XwX/Mechcraft-Tech.git)" # Customize User-Agent for easier server-side identification
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"] # Default transaction type list

MAX_LIMIT = 1000 # The maximum number of requests returned in a single request

INTERVAL_MS = {
"1m": 60_000, 
"3m": 3 * 60_000, 
"5m": 5 * 60_000, 
"15m": 15 * 60_000, 
"30m": 30 * 60_000,  
"1h": 60 * 60_000}

# Configuration data class: centralize the management of operating parameters
@dataclass
class FetchConfig:
    symbols: List[str]
    interval: str = "1m" # K-line cycle
    start: Optional[str] = None
    end: Optional[str] = None
    days: Optional[int] = None
    out_dir: str = "./data"
    file_format: str = "csv"
    sleep_sec: float = 0.5 # The number of seconds after each requests
    max_retries: int = 5 # the max time of retries after an error
    timeout: int = 20 # overtime period for each HTTP requests

def parse_time(time: Optional[str]) -> Optional[datetime]:
    """
    Parse a command-line time argument into a UTC-aware datetime object.
    If time is a number with a character d, which means how many days ago from now.
    If time is a number with a character h, which means how many hours ago from now.
    Otherwise, time is a absolute time format.

    Args:
        time (Optional[str]): Time string from command-line input, or None.

    Returns:
        Optional[datetime]: Parsed UTC-aware datetime object, or None if input is None.
    """
    if (time == None):
        return None
    time = time.strip().lower()
    # day
    if (time.endswith("d") and time[:-1].isdigit()):
        days = int(time[:-1])
        return datetime.now(timezone.utc) - timedelta(days=days)
    # hour
    if (time.endswith("h") and time[:-1].isdigit()):
        hours = int(time[:-1])
        return datetime.now(timezone.utc) - timedelta(hours=hours)
    # A formal date input
    dt = pd.to_datetime(time, utc=True)
    return dt.to_pydatetime()

def to_ms(dt: datetime) -> int:
    """
    Switch the datetime to millisecond.

    Args:
        dt (datetime): The datetime

    Returns:
        int: The datetime in ms.
    """
    if (dt.tzinfo is None):
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def request_with_retry(url: str, params: Dict[str, Any], max_retries: int, timeout: int, sleep_sec: float) -> List[Any]:
    """
    Send an HTTP GET request with retry mechanism, handling rate limits and network errors.
    Uses requests.get with query parameters, timeout, and custom User-Agent.
    If response status code is 429 (Too Many Requests), read `Retry-After` header or wait a default duration before retrying.
    On network or JSON parsing errors, retries with exponential backoff.
    Raises an exception if all retry attempts fail.
    Raises RuntimeError if Binance returns an error JSON object (with "code"/"msg").
    Returns parsed JSON data (usually a list of kline arrays) on success.

    Args:
        url (str): Target request URL.
        params (Dict[str, Any]): Query parameters appended to the URL.
        max_retries (int): Maximum retry attempts.
        timeout (int): Timeout (in seconds) for each request.
        sleep_sec (float): Initial sleep duration for rate-limit wait and backoff.

    Returns:
        List[Any]: Parsed JSON data, typically a list (e.g., Binance kline arrays).

    Raises:
        RuntimeError: If Binance API returns an error response.
        requests.RequestException: If repeated network request failures occur.
        ValueError: If response cannot be parsed as JSON.
    """
    tries = 0
    while True:
        tries += 1
        try:
            resp = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=timeout)
            if (resp.status_code == 429):
                # Rate-limited: Read the Retry-After header (if any) and wait
                retry_after = int(resp.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, sleep_sec))
                if (tries <= max_retries):
                    continue
                resp.raise_for_status()
            resp.raise_for_status() # If the status code belongs to other 2xx codes, raise error.
            data = resp.json()
            if (isinstance(data, dict) and data.get("code")):
                # Binance: When an error is returned as an object, contain code and msg.
                raise RuntimeError(f"Binance error: {data}")
            return data # Returns a JSON list (K-line array)
        except (requests.RequestException, ValueError) as e:
            # Network or parsing errors: Retry after exponential backoff.
            if (tries > max_retries):
                raise
            time.sleep(sleep_sec * (2 ** (tries - 1)))

def fetch_symbol_klines(symbol: str, interval: str, start_ms: int, end_ms: int, cfg: FetchConfig) -> pd.DataFrame:
    """
    Fetch historical candlestick (kline) data for a given trading symbol from Binance,
    handling pagination, rate-limiting, and data normalization.

    Requests candlestick data page by page from Binance API using `startTime` and pagination logic.
    Automatically retries requests with backoff when network errors or rate limits occur.
    If an empty response is returned, skips ahead by a large interval window.
    Converts the raw kline array into a structured pandas DataFrame with meaningful column names.
    Cleans and normalizes data:
        * Converts numeric fields to float or int types.
        * Converts timestamps to UTC datetime.
        * Ensures rows are sorted by open time and deduplicated.
    Returns an empty DataFrame with predefined columns if no data is found.

    Args:
        symbol (str): Trading symbol (e.g., "BTCUSDT").
        interval (str): Kline interval (e.g., "1m", "5m", "1h").
        start_ms (int): Start timestamp in milliseconds (inclusive).
        end_ms (int): End timestamp in milliseconds (exclusive).
        cfg (FetchConfig): Configuration object with retry, timeout, and sleep settings.

    Returns:
        pd.DataFrame: A cleaned and structured DataFrame with the following columns:
        symbol, open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote
    """
    url = BINANCE_BASE + KLINES_EP
    all_rows = [] # Collect all the rows from all pages
    cur = start_ms
    # The time span of each K line
    step = INTERVAL_MS[interval]
    while cur < end_ms:
        params = {
            "symbol": symbol.upper(), 
            "interval": interval, 
            "startTime": cur, 
            "limit": MAX_LIMIT}
        # Do not pass endTime, let the server return the latest 1000 roots based on startTime (more stable paging).
        data = request_with_retry(url, params, cfg.max_retries, cfg.timeout, cfg.sleep_sec)
        if (not data):
            # A blank page means that there are no data inside this interval.
            cur += step * MAX_LIMIT
            continue
        for row in data:
            # The format in Binance file:
            # 0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close time,
            # 7 quote asset volume, 8 number of trades, 9 taker buy base vol,
            # 10 taker buy quote vol, 11 ignore
            # The current row has exceeded the end time specified by the user
            if (row[0] >= end_ms):
                break
            all_rows.append(row)
        last_open = data[-1][0] # The last opening time of the current returned page
        cur = last_open + step
        time.sleep(cfg.sleep_sec) # Avoid triggering frequency limiting
        # Finish collecting data
        if (len(data) < MAX_LIMIT and cur >= end_ms):
            break
    if (not all_rows):
        # If collecting data is unscessful, keep the default format
        return pd.DataFrame(columns=[
            "symbol", "open_time", "open", "high", "low", "close", "volume", 
            "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"])

    # Convert the original array to a DataFrame and name the columns
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time", 
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "_ignore"])

    df.drop(columns=["_ignore"], inplace=True) # Discard placeholder columns
    df.insert(0, "symbol", symbol.upper()) # Insert symbol in the first column

    # Switch the data tyoe: Basic info of trades turn to float; time turn to UTC datetime
    num_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Clean repeated data. If the data has the same symbol and the same open time, only keep the last one.
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset=["symbol","open_time"], keep="last", inplace=True, ignore_index=True)
    return df

def fetch_klines(cfg: FetchConfig) -> pd.DataFrame:
    """
    Fetch historical kline data for multiple trading symbols from Binance Spot API.
    This function determines the time range to fetch based on user configuration 
    (either a fixed number of days or explicit start/end time), validates the interval, 
    and then retrieves candlestick data for each requested symbol.
    The results are combined into a single pandas DataFrame.

    Args:
        cfg (FetchConfig): Configuration object containing fetch parameters:
            - symbols (List[str]): List of trading symbols such as ["BTCUSDT", "ETHUSDT"].
            - interval (str): Kline interval such as "1m", "5m", "1h".
            - days (Optional[int]): If set, fetch data for the past N days until now.
            - start (Optional[str]): Start time
            - end (Optional[str]): End time
            - other config options such as output format, retry policy.

    Raises:
        ValueError: If both `days` and `start/end` are provided, manually exclusive.
        ValueError: If `interval` is not supported.

    Returns:
        pd.DataFrame: Combined candlestick data for all symbols, with columns like [timestamp, open, high, low, close, volume, symbol].
    """
    # Cannot use both day parameter or start and end parameters
    if ((cfg.days is not None) and (cfg.start or cfg.end)):
        raise ValueError("Use either __days or __start/__end, not both.")
    if (cfg.days != None):
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=cfg.days)
    else:
        end_dt = parse_time(cfg.end) or datetime.now(timezone.utc)
        start_dt = parse_time(cfg.start)
        if (start_dt == None):
            # set the end time to 7 days ago
            start_dt = end_dt - timedelta(days=7)
    if (cfg.interval not in INTERVAL_MS):
        raise ValueError(f"Unsupported interval: {cfg.interval}. Supported: {list(INTERVAL_MS.keys())}")
    start_ms = to_ms(start_dt)
    end_ms = to_ms(end_dt)
    all_df = []
    for sym in cfg.symbols:
        # only print the information in terminal
        print(f"[INFO] Fetching {sym} {cfg.interval} from {start_dt.isoformat()} to {end_dt.isoformat()} ...", file=sys.stderr)
        df = fetch_symbol_klines(sym, cfg.interval, start_ms, end_ms, cfg)
        print(f"[INFO] {sym}: fetched {len(df)} rows.", file=sys.stderr)
        all_df.append(df)
    if (not all_df):
        return pd.DataFrame()
    return pd.concat(all_df, ignore_index=True)

def check_gaps(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Detect missing candlesticks' gaps in a kline DataFrame for each trading symbol.
    This function compares the actual open_time sequence of candles with the expected 
    continuous sequence, previous open_time + interval. If a candle is missing, 
    it records the timestamp where the gap occurs and the number of missing candles.

    Args:
        df (pd.DataFrame): Kline data with at least ["symbol", "open_time"] columns.
        interval (str): Interval string (e.g., "1m", "5m", "1h"), used to determine expected spacing.

    Returns:
        pd.DataFrame: A DataFrame containing gap information with columns:
            - "symbol": the trading pair symbol.
            - "gap_before": timestamp of the candle where a gap was detected.
            - "missing_candles": number of missing candles before this candle.
    """
    # Convert interval string (e.g., "1m") into milliseconds per candle
    ms = INTERVAL_MS[interval]
    out = []
    # Process each trading symbol separately
    for sym, g in df.groupby("symbol"):
        # Sort candles by open_time to ensure chronological order
        g = g.sort_values("open_time")
        # Expect the next open time equals a period of time after the last open time
        expected = g["open_time"].shift(1) + pd.to_timedelta(ms, unit="ms")
        # If the actual open time != expected, gap found
        gaps = g.loc[g["open_time"] != expected]
        # The first row is always flagged (no previous candle), so skip it
        gaps = gaps.iloc[1:]
        for idx, row in gaps.iterrows():
            # Use the true previous candle's open_time instead of calculating directly
            prev_open = g.loc[idx - 1, "open_time"]
            # Calculate the missing candles
            miss = int((row["open_time"] - prev_open).total_seconds() * 1000 // ms) - 1
            # Record result: which symbol, at what time, and how many candles are missing
            out.append({"symbol": sym, "gap_before": row["open_time"].isoformat(), "missing_candles": max(miss, 0)})
    return pd.DataFrame(out)

def save_df(df: pd.DataFrame, out_dir: str, file_format: str, interval: str):
    """
    Save kline data to disk, grouped by trading symbol. Each symbol's data is saved into a 
    separate file, named according to: {symbol}_{interval}_{start_date}_{end_date}.{csv|parquet}

    Args:
        df (pd.DataFrame): DataFrame containing kline data, must include "symbol" and "open_time" columns.
        out_dir (str): Directory path where files will be saved.
        file_format (str): Output file format, must be either "csv" or "parquet".
        interval (str): Kline interval string (e.g., "1m", "5m", "1h"), included in the filename.

    Raises:
        ValueError: If `file_format` is not "csv" or "parquet".
    """
    # Create the output directory (recursively), if it does not already exist
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save one file per symbol. Easier for incremental updates and merges later.
    for sym, g in df.groupby("symbol"):
        # Build a base filename with symbol, interval, start date, end date
        fname_base = f"{sym}_{interval}_{g['open_time'].min().strftime('%Y%m%d')}_{g['open_time'].max().strftime('%Y%m%d')}"
        if (file_format == "csv"):
            path = out / f"{fname_base}.csv"
            g.to_csv(path, index=False)
        elif (file_format == "parquet"):
            path = out / f"{fname_base}.parquet"
            try:
                g.to_parquet(path, index=False)
            except Exception as e:
                # If pyarrow/fastparquet is missing, fallback to CSV
                alt = out / f"{fname_base}.csv"
                g.to_csv(alt, index=False)
                print(f"[WARN] Parquet not available ({e}); wrote CSV instead: {alt}", file=sys.stderr)
                continue
        else:
            # Unsupported file format
            raise ValueError("file_format must be 'csv' or 'parquet'")
        print(f"[OK] Wrote {path} ({len(g)} rows)")

def parse_arguments() -> FetchConfig:
    """
    Parse command-line arguments for Binance kline downloader.
    This function uses argparse to define and parse available CLI options.
    The parsed values are wrapped into a FetchConfig object, which is used
    throughout the pipeline for fetching and saving kline data.

    Returns:
        FetchConfig: A configuration object containing all user-specified parameters:
            - symbols (List[str]): Trading symbols, e.g., ["BTCUSDT", "ETHUSDT"].
            - interval (str): Kline interval (e.g., "1m", "5m", "1h").
            - start (Optional[str]): Start time (ISO8601 string, date string, or relative like "7d").
            - end (Optional[str]): End time (ISO8601 string, date string; defaults to now).
            - days (Optional[int]): Number of days to fetch (alternative to start/end).
            - out_dir (str): Output directory for saved files.
            - file_format (str): Output file format ("csv" or "parquet").
            - sleep_sec (float): Delay (in seconds) between API requests.
            - max_retries (int): Maximum number of retries for failed requests.
            - timeout (int): HTTP timeout (seconds).
    """
    # Initialize an argument parser with description
    p = argparse.ArgumentParser(description="Download Binance Spot klines into tidy CSV/Parquet files.")

    # Symbols to fetch (one or more); default is predefined trading pairs
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols like BTCUSDT ETHUSDT ...")

    # Interval for klines (1m, 5m, 1h, etc.); default 1m
    p.add_argument("--interval", default="1m", choices=list(INTERVAL_MS.keys()), help="Kline interval (default: 1m)")

    # Start time for data fetch (string input: ISO8601, simple date, or relative like '7d')
    p.add_argument("--start", type=str, default=None, help="Start time (e.g., '2024-08-01' or '2024-08-01T00:00Z' or '7d')")

    # End time for data fetch (string input, defaults to now if not provided)
    p.add_argument("--end", type=str, default=None, help="End time (default: now)")

    # Alternative to start/end: fetch last N days
    p.add_argument("--days", type=int, default=None, help="Fetch last N days (alternative to start/end)")

    # Output directory (default ./data)
    p.add_argument("--out", dest="out_dir", default="./data", help="Output directory (default: ./data)")

    # Output file format: CSV or Parquet (default CSV)
    p.add_argument("--format", dest="file_format", default="csv", choices=["csv","parquet"], help="Output format")

    # Sleep time between API requests (to avoid rate limits)
    p.add_argument("--sleep", dest="sleep_sec", type=float, default=0.5, help="Sleep seconds between requests")

    # Maximum retries for API errors
    p.add_argument("--retries", dest="max_retries", type=int, default=5, help="Max retries on errors")

    # HTTP timeout for requests
    p.add_argument("--timeout", dest="timeout", type=int, default=20, help="HTTP timeout seconds")

    # Parse command-line arguments by argparse.ArgumentParser
    args = p.parse_args()

    # Wrap parsed arguments into FetchConfig dataclass and return
    return FetchConfig(
        symbols=args.symbols,
        interval=args.interval,
        start=args.start,
        end=args.end,
        days=args.days,
        out_dir=args.out_dir,
        file_format=args.file_format,
        sleep_sec=args.sleep_sec,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )

def get_binance_dataset():
    """
    Fetch k lines data from Binance API and make some simple detection to see if the data is
    collected successfully. Then save the data to a csv file.
    """
    # Read configuration parameters
    cfg = parse_arguments()
    df = fetch_klines(cfg)
    if (df.empty):
        print("[WARN] No data fetched. Check your symbols/time window.", file=sys.stderr)
        return
    gaps = check_gaps(df, cfg.interval)
    if (not gaps.empty):
        print("[INFO] Detected potential gaps (maintenance/listing gaps). This is informational.")
        print(gaps.head(20).to_string(index=False))
    save_df(df, cfg.out_dir, cfg.file_format, cfg.interval)