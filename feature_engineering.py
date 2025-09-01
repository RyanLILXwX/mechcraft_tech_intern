from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path

def infer_freq_alias(interval: str) -> str:
    """
    Convert a given interval string (e.g., "1m", "5min", "2h") into a pandas-compatible frequency alias.

    Args:
        interval (str): Time interval string from the exchange or user input, such as "1m", "15min", "4h", "1d".
        
    Returns:
        str: A pandas offset alias string (e.g., "min", "5min", "h", "d") that can be used in resampling or time series operations.
        Default time value is set to 5min.
    """
    # Mapped the normal time into time frequency symbols that pandas can recognize
    pandas_time_dict = {
        "1m": "1min", 
        "3m": "3min", 
        "5m": "5min", 
        "15m": "15min", 
        "30m": "30min", 
        "1h": "1h", 
        "4h": "4h", 
        "1d": "1d"}
    if (interval in pandas_time_dict):
        return pandas_time_dict[interval]
    # Fallback: try to parse like "5min" or "5m"
    s = interval.lower().replace("minute", "m").replace("min", "m")
    if (s.endswith("m") and s[:-1].isdigit()):
        return f"{s[:-1]}min"
    if (s.endswith("h") and s[:-1].isdigit()):
        return f"{s[:-1]}h"
    if (s.endswith("d") and s[:-1].isdigit()):
        return f"{s[:-1]}d"
    # Default to minutes
    return "5min"

def load_and_prepare(path: str, interval: str = "5m", expected_symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Load a candlestick (K-line) CSV file and perform minimal cleaning/validation.
    This function standardizes timestamp columns, ensures numeric stability, and prepares a 
    strictly increasing, time-indexed DataFrame suitable for time series analysis or resampling.
    - Reads the CSV into a DataFrame.
    - Parses timestamp columns ("open_time" and "close_time") as UTC datetimes.
    - Optionally validates that the file contains the expected trading symbol.
    - Renames "open_time" to "ts" as the canonical timestamp for each candle.
    - Sorts rows chronologically, removes duplicate timestamps.
    - Enforces numeric dtypes for OHLCV-related columns.
    - Removes rows with invalid timestamps or missing closing prices.
    - Clips obviously inconsistent high/low values to ensure logical bounds.
    - Attaches interval metadata (`interval`, `freq_alias`) to DataFrame attributes.

    Args:
        path (str): File path to the candlestick CSV file.
        interval (str, optional): The trading interval (e.g., "1m", "5m", "1h"). Default is "5m".
        expected_symbol (Optional[str], optional): If provided, checks that this symbol exists in 
        the file. Raises ValueError if not found.

    Returns:
        pd.DataFrame:
            A cleaned DataFrame with:
            - Column "ts" as the datetime index (strictly increasing).
            - Standardized OHLCV columns with numeric dtypes.
            - DataFrame attributes:
                - `interval`: the original interval string.
                - `freq_alias`: the pandas frequency alias (e.g., "min", "5min", "h").
    """
    df = pd.read_csv(path)
    # Parse timestamps safely
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
    # Check if the symbol in the file is equal to the expected_symbol
    if (expected_symbol is not None):
        # Check what kind of symbols in the file, remove duplicated symbol type
        symbols = set(df["symbol"].astype(str).unique())
        if (expected_symbol not in symbols):
            raise ValueError(f"expected_symbol={expected_symbol} not found in file symbols={symbols}")
    # Define canonical timestamp for the candle: use open_time, unify all types of open_time to ts
    df = df.rename(columns={"open_time": "ts"})
    # Sort and drop dups
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    # Enforce dtypes for numeric stability
    numeric_cols = ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]
    for c in numeric_cols:
        if (c in df.columns):
            # If there are non-numeric values ​​in this column (such as "null", empty strings, abc), 
            # they will be automatically converted to NaN instead of reporting an error.
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Remove rows with invalid timestamps or prices
    df = df[df["ts"].notna() & df["close"].notna()]
    # Clip obviously broken values, usually low <= open, close <= high
    # If the number is typed in the wrong place, we just need to replace the number with a comparatively reasonable value.
    df["high"] = np.maximum(df["high"].values, df[["open","close","low"]].max(axis=1).values)
    df["low"] = np.minimum(df["low"].values, df[["open","close","high"]].min(axis=1).values)
    # Attach interval frequency info with default .attrs attribute
    df.attrs["interval"] = interval
    df.attrs["freq_alias"] = infer_freq_alias(interval)
    return df

def find_time_gaps(df: pd.DataFrame, max_allow_miss: int = 0) -> pd.DataFrame:
    """
    Detect missing candlestick intervals (gaps) in a time-indexed DataFrame.
    This function checks for missing candles by comparing the DataFrame"s
    timestamps against a complete date range generated from the interval
    frequency. Any absent timestamps are grouped into contiguous segments
    (gap ranges) and summarized as start time, end time, and the number of
    missing candles. Although in the binance_downloader.py, we already checked the gap
    missing problem, but we still need to check again in this section. Avoid local csv
    file storage issues.

    Args:
        df (pd.DataFrame): Input DataFrame that must contain a "ts" column (timestamps). 
        Typically produced by `load_and_prepare`.
        
        max_allow_miss (int, optional): Threshold for ignoring small gaps. If > 0, consecutive missing 
        candles up to this count will not be reported. Default is 0 (report all missing gaps).

    Returns:
        pd.DataFrame: A DataFrame summarizing the missing intervals with columns:
            - "gap_start": Timestamp of the first missing candle in the segment.
            - "gap_end": Timestamp of the last missing candle in the segment.
            - "missing_count": Number of consecutive missing candles in the segment.
            - If no gaps are found, returns an empty DataFrame with the same column names but no rows.
    """
    if ("ts" not in df):
        raise ValueError("DataFrame must contain 'ts' column. Call load_and_prepare first.")
    freq = df.attrs.get("freq_alias", "5min")
    # Use pd.date_range to generate a complete, continuous sequence of timestamps. Start from the earliest 
    # time of the current data df["ts"].min() to the latest time df["ts"].max(), incrementing by freq 
    # and set to the UTC time zone.
    full = pd.DataFrame(index=pd.date_range(df["ts"].min(), df["ts"].max(), freq=freq, tz="UTC"))
    full.index.name = "ts"
    # .assign(_present=1): Add a new column _present and mark those timestamps that actually exist in the original data with 1.
    # A DataFrame based on a complete time index has only one column, _present, which indicates whether the time point appears in the original data.
    # 1 = present, NaN = missing
    merged = full.join(df.set_index("ts").assign(_present=1)[["_present"]], how="left")
    missing = merged[merged["_present"].isna()].index
    if (len(missing) == 0):
        return pd.DataFrame(columns=["gap_start","gap_end","missing_count"]) # no gaps
    # Derive fixed step from freq (works for fixed intervals like "min","5min","h","4h","d").
    step = pd.Timedelta(freq)
    # Group consecutive missing timestamps into segments
    gaps = []
    n_missing = len(missing)
    if (n_missing == 1):
        # Single missing timestamp segment
        if (max_allow_miss < 1):
            gaps.append((missing[0], missing[0], 1))
    else:
        start = prev = missing[0]
        for t in missing[1:]:
            if (t - prev == step):
                # Still within the same consecutive-missing segment
                prev = t
            else:
                # Close the previous segment [start, prev]
                count = int((prev - start) / step) + 1
                if (max_allow_miss < count):
                    gaps.append((start, prev, count))
                # Start a new segment at t
                start = prev = t
        # Close the last open segment
        count = int((prev - start) / step) + 1
        if (max_allow_miss < count):
            gaps.append((start, prev, count))
    if (gaps != []):
        return pd.DataFrame(gaps, columns=["gap_start", "gap_end", "missing_count"])
    else:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_count"])

# -------------------------------------------------------------------------
# Basic price and return calculations and technical indicator calculations

def add_price_returns_and_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic return features for OHLC data and append them to the DataFrame. Adds three time-series features:
        - Simple return: (close_t - close_{t-1}) / close_{t-1}
        - Log return: log(close_t) - log(close_{t-1})
        - High-low spread: (high - low) / close
        The function works on a copy of the input DataFrame (no in-place mutation).

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the numeric columns: "close", "high", and "low".
        Typically indexed by a timestamp.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with three added columns: 
            - "ret_simple": simple returns computed via pct_change on "close"
            - "ret_log": log returns computed as diff of log("close")
            - "hl_spread": (high - low) / close (with close==0 treated as NaN)
    """
    df = df.copy()
    df["ret_simple"] = df["close"].pct_change()
    df["ret_log"] = np.log(df["close"]).diff()
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    return df

def exponential_moving_average(s: pd.Series, span: int) -> pd.Series:
    """
    Calculate exponential moving average.

    Args:
        s (pd.Series): A series of timestamp
        span (int): Smooth window size

    Returns:
        pd.Series: Returns a pd.Series of the same length as the input sequence, containing the EMA values.
    """
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def add_moving_averages(df: pd.DataFrame, windows: Iterable[int] = (5, 10, 20, 50)) -> pd.DataFrame:
    """
    For each window length in `windows`, this function computes:
        - Price SMA_w: simple moving average of "close" over w periods
        - Price EMA_w: exponential moving average of "close" over w periods
        - Volume SMA_w: simple moving average of "volume" over w periods

    Args:
        df (pd.DataFrame): Input DataFrame containing at least numeric columns "close" and "volume".
        windows (Iterable[int], optional): An iterable of window lengths.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with added columns:
            - "sma_{w}": rolling mean of "close" with window w (min_periods=w)
            - "ema_{w}": exponential moving average of "close" with window w
            - "vol_sma_{w}": rolling mean of "volume" with window w (min_periods=w)
    """
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(w, min_periods=w).mean()
        df[f"ema_{w}"] = exponential_moving_average(df["close"], w)
        df[f"vol_sma_{w}"] = df["volume"].rolling(w, min_periods=w).mean()
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute the Relative Strength Index (RSI) using Wilder"s smoothing and add it as a new column.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least a "close" price column.
        period (int, optional): Lookback period for RSI calculation. Defaults to 14.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an additional column "rsi_<period>".
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute the Moving Average Convergence Divergence (MACD) and add related columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least a "close" price column.
        fast (int, optional): Period for the fast EMA. Defaults to 12.
        slow (int, optional): Period for the slow EMA. Defaults to 26.
        signal (int, optional): Period for the signal line EMA. Defaults to 9.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with added columns:
            - "macd_<fast>_<slow>": MACD line
            - "macd_signal_<signal>": Signal line
            - "macd_hist": Histogram (MACD minus signal line)
    """
    df = df.copy()
    ema_fast = exponential_moving_average(df["close"], fast)
    ema_slow = exponential_moving_average(df["close"], slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - signal_line
    df[f"macd_{fast}_{slow}"] = macd
    df[f"macd_signal_{signal}"] = signal_line
    df["macd_hist"] = hist
    return df

def add_bbands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands and %B indicator and add them as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least a "close" price column.
        period (int, optional): Lookback period for the moving average. Defaults to 20.
        num_std (float, optional): Number of standard deviations for the bands. Defaults to 2.0.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with added columns:
            - "bb_mid_<period>": Middle band (moving average)
            - "bb_up_<period>": Upper band
            - "bb_low_<period>": Lower band
            - "bb_percent_b": %B indicator ((close - lower) / (upper - lower))
    """
    df = df.copy()
    ma = df["close"].rolling(period, min_periods=period).mean()
    std = df["close"].rolling(period, min_periods=period).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    df[f"bb_mid_{period}"] = ma
    df[f"bb_up_{period}"] = upper
    df[f"bb_low_{period}"] = lower
    df["bb_percent_b"] = (df["close"] - lower) / (upper - lower)
    return df

def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute the Average True Range (ATR) and add it as a new column.

    Args:
        df (pd.DataFrame): Input DataFrame containing "high", "low", and "close" price columns.
        period (int, optional): Lookback period for ATR calculation. Defaults to 14.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with an added column:
            - "atr_<period>": Average True Range
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return df

def add_roll_volatility(df: pd.DataFrame, windows: Iterable[int] = (12, 24, 72)) -> pd.DataFrame:
    """
    Compute rolling volatility of log returns over specified window sizes and add them as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain or be augmented with a "ret_log" column.
        windows (Iterable[int], optional): Iterable of window lengths (in periods) for volatility calculation. Defaults to (12, 24, 72).

    Returns:
        pd.DataFrame: A copy of the input DataFrame with additional columns:
            - "vol_logret_<w>": Rolling volatility of log returns over window <w>.
    """
    df = add_price_returns_and_spread(df)
    for w in windows:
        df[f"vol_logret_{w}"] = df["ret_log"].rolling(w, min_periods=w).std(ddof=0) * math.sqrt(1.0)
    return df

def add_volume_features(df: pd.DataFrame, windows: Iterable[int] = (5, 20, 50)) -> pd.DataFrame:
    """
    Compute volume-based statistics and taker buy ratios, and add them as new columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing "volume" and "close" columns. 
            Optionally may include "quote_volume", "taker_buy_base", "taker_buy_quote", and "trades".
        windows (Iterable[int], optional): Iterable of window lengths (in periods) for rolling 
            statistics. Defaults to (5, 20, 50).

    Returns:
        pd.DataFrame: A copy of the input DataFrame with additional columns:
            - "dollar_volume": Dollar volume (from "quote_volume" if available, else close * volume)
            - "taker_buy_ratio": Taker buy base / volume (if available)
            - "taker_buy_quote_ratio": Taker buy quote / quote volume (if available)
            - "vol_roll_mean_<w>": Rolling mean of volume over window <w>
            - "vol_roll_std_<w>": Rolling standard deviation of volume over window <w>
            - "trades_roll_mean_<w>": Rolling mean of trades over window <w> (if available)
            - "trades_roll_std_<w>": Rolling standard deviation of trades over window <w> (if available)
            - "dollar_vol_roll_mean_<w>": Rolling mean of dollar volume over window <w>
    """
    df = df.copy()
    # Dollar volume
    if ("quote_volume" in df.columns):
        df["dollar_volume"] = df["quote_volume"]
    else:
        df["dollar_volume"] = df["close"] * df["volume"]
    # Taker buy ratios
    if ("taker_buy_base" in df.columns and "volume" in df.columns):
        df["taker_buy_ratio"] = (df["taker_buy_base"] / df["volume"]).replace([np.inf, -np.inf], np.nan)
    if ("taker_buy_quote" in df.columns and "quote_volume" in df.columns):
        df["taker_buy_quote_ratio"] = (df["taker_buy_quote"] / df["quote_volume"]).replace([np.inf, -np.inf], np.nan)
    # Rolling stats
    for w in windows:
        df[f"vol_roll_mean_{w}"] = df["volume"].rolling(w, min_periods=w).mean()
        df[f"vol_roll_std_{w}"]  = df["volume"].rolling(w, min_periods=w).std(ddof=0)
        df[f"trades_roll_mean_{w}"] = df["trades"].rolling(w, min_periods=w).mean() if "trades" in df.columns else np.nan
        df[f"trades_roll_std_{w}"]  = df["trades"].rolling(w, min_periods=w).std(ddof=0) if "trades" in df.columns else np.nan
        df[f"dollar_vol_roll_mean_{w}"] = df["dollar_volume"].rolling(w, min_periods=w).mean()
    return df

# -------------------------------------------------------------------------
# Composite feature engineering pipeline and build feature used for split train/valid/test dataset

@dataclass
class WindowsConfig:
    ma: Tuple[int, ...] = (5, 10, 20, 50) # Moving Average window length
    vol: Tuple[int, ...] = (12, 24, 72) # Rolling volatility window length
    vol_feat: Tuple[int, ...] = (5, 20, 50) # Window length of volume features
    rsi: int = 14 # Period of RSI
    macd_fast: int = 12 # macd parameters
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20 # Bollinger Bands Period and Standard Deviation Multiples
    bb_std: float = 2.0
    atr_period: int = 14

def add_all_features(df: pd.DataFrame, windows_cfg: Optional[WindowsConfig] = None) -> pd.DataFrame:
    """
    Build a comprehensive, clean feature set for OHLCV data. Applies a standardized pipeline to compute 
    price returns/spread, moving averages, RSI, MACD (line/signal/hist), Bollinger Bands and %B, ATR, rolling volatility of
    log returns, and volume-based statistics. Finally, drops the leading drop_nan rows implied by the 
    largest lookback to remove NaNs and alignment artifacts.

    Args:
        df (pd.DataFrame): Input OHLCV DataFrame (optionally with quote/taker fields).
        windows_cfg (Optional[WindowsConfig]): Indicator parameter configuration. If None,
            defaults to WindowsConfig().

    Returns:
        pd.DataFrame: A copy of the DataFrame with all feature columns added and the
        initial drop_nan rows removed.
    """
    cfg = windows_cfg or WindowsConfig()
    df = df.copy()
    df = add_price_returns_and_spread(df)
    df = add_moving_averages(df, windows=cfg.ma)
    df = add_rsi(df, period=cfg.rsi)
    df = add_macd(df, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    df = add_bbands(df, period=cfg.bb_period, num_std=cfg.bb_std)
    df = add_atr(df, period=cfg.atr_period)
    df = add_roll_volatility(df, windows=cfg.vol)
    df = add_volume_features(df, windows=cfg.vol_feat)

    # Drop rows with NaNs introduced by rolling windows at the beginning
    # The rolling mean/variance, Bollinger Bands, ATR, RSI, and MACD will all show NaN or unstable 
    # estimates in the early stages. The rows before the maximum window length are uniformly trimmed.
    drop_nan = max(
        max(cfg.ma) if len(cfg.ma) else 0, 
        cfg.rsi, 
        cfg.macd_slow, 
        cfg.bb_period, 
        cfg.atr_period, 
        max(cfg.vol) if len(cfg.vol) else 0, 
        max(cfg.vol_feat) if len(cfg.vol_feat) else 0)
    if (drop_nan > 0):
        df = df.iloc[drop_nan:].copy()
    return df

def future_return(series: pd.Series, horizon: int) -> pd.Series:
    """
    Calculate the log return over a future time horizon

    Args:
        series (pd.Series): The series of certain assets" close price
        horizon (int): Time period

    Returns:
        pd.Series: A pandas series where each element is the future log return of the input price series.
    """
    return np.log(series.shift(-horizon)) - np.log(series)

def make_supervised_targets(df: pd.DataFrame, horizons: Iterable[int] = (1, 5, 15), task: str = "binclass", thresh: float = 0.0) -> pd.DataFrame:
    """
    Generate supervised learning targets based on future log returns.
    For each prediction horizon, this function computes the forward log return
    of the "close" price and derives task-specific target columns:
    - Regression ("reg"): continuous log return values (target_ret_<h>).
    - Binary classification ("binclass"): up/down label (target_up_<h>), 1 if
      return > thresh, else 0.
    - Multiclass classification ("multiclass"): trinary label (target_cls_<h>),
      with bins [-inf, -thresh), [-thresh, +thresh], (+thresh, inf] mapped to
      -1, 0, 1.
    To ensure alignment, the last `max(horizons)` rows are dropped since future prices are unavailable for computing returns.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least the "close" column.
        horizons (Iterable[int], optional): Horizons (in bars) over which to compute future returns. Defaults to (1, 5, 15).
        task (str, optional): Task type. One of {"binclass", "reg", "multiclass"}. Defaults to "binclass".
        thresh (float, optional): Threshold for classification. For multiclass, defines the middle band around zero. Defaults to 0.0.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with additional target columns (`target_*`) aligned with features and with tail rows dropped.
    """
    # Make a copy so we don"t modify the original DataFrame in place
    df = df.copy()
    # Loop over each prediction horizon (e.g., 1 bar, 5 bars, 15 bars)
    for h in horizons:
        # Compute the future log return over horizon h
        fr = future_return(df["close"], h)
        if (task == "reg"):
            # Regression task:
            # Directly use the future log return as the target value
            df[f"target_ret_{h}"] = fr
        elif (task == "multiclass"):
            # Multiclass task:
            # We want 3 categories: -1 (down), 0 (flat), 1 (up)
            # Initialize all labels as NaN
            lab = pd.Series(np.nan, index=df.index)
            # Return greater than +thresh, label = 1 (up)
            lab = lab.mask(fr >  thresh, 1)
            # Return between -thresh and +thresh, label = 0 (flat)
            lab = lab.mask((fr >= -thresh) & (fr <= thresh), 0)
            # Return less than -thresh, label = -1 (down)
            lab = lab.mask(fr < -thresh, -1)
            # Store as nullable integer type (so NaN can be preserved as <NA>)
            df[f"target_cls_{h}"] = lab.astype("Int64")
        # Default is "binclass". Binary classification task: Label = 1 if return > thresh, else 0. NaN values remain as <NA>
        else:
            df[f"target_up_{h}"] = (fr > thresh).astype("Int64")
    # After creating all target columns, remove rows at the tail. Because for the maximum horizon, those last rows don"t have
    # enough future data to compute returns, so labels are NaN.
    max_h = max(horizons) if horizons else 0
    if (max_h > 0):
        df = df.iloc[:-max_h].copy()
    # Return the DataFrame with targets added and aligned
    return df

def build_features(df: pd.DataFrame, windows_cfg: Optional[WindowsConfig] = None, horizons: Iterable[int] = (1, 5, 15), 
                   task: str = "binclass", thresh: float = 0.0, log_gaps: bool = False) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Build features and supervised targets on a pre-loaded OHLCV DataFrame.
    This function does NOT split datasets. It only: optionally logs time gaps, engineers features with `add_all_features`, and 
    generates supervised targets with `make_supervised_targets`. Last, return the augmented DataFrame along 
    with feature/target column lists.

    Args:
        df: Pre-loaded OHLCV DataFrame (optionally with quote/taker fields).
        windows_cfg: Parameter config for technical indicators. If None, use defaults.
        horizons: Prediction horizons (in bars) for targets.
        task: "binclass" | "reg" | "multiclass".
        thresh: Threshold for classification targets.
        log_gaps: If True, print the first few detected time gaps.

    Returns:
        (df_out, feature_cols, target_cols)
    """
    # Work on a copy to avoid side effects
    df = df.copy()
    # log time gaps for visibility
    if (log_gaps):
        gaps = find_time_gaps(df)
        if (len(gaps) > 0):
            print("[INFO] Detected time gaps:")
            print(gaps.head(20).to_string(index=False))
    # Feature engineering (price returns/spread, MAs, RSI, MACD, BB, ATR, vol, volume features, etc.)
    df = add_all_features(df, windows_cfg=windows_cfg)
    # Supervised targets aligned and tail-trimmed inside make_supervised_targets
    df = make_supervised_targets(df, horizons=horizons, task=task, thresh=thresh)
    # Separate features and targets for downstream split in get_data.py
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if c not in target_cols]
    return df, feature_cols, target_cols

def get_features_dataset(csv_path: str, interval: str = "5m", expected_symbol: Optional[str] = None, 
                         windows_cfg: Optional[WindowsConfig] = None, horizons: Iterable[int] = (1, 5, 15), 
                         task: str = "binclass", thresh: float = 0.0, out_path: Optional[str] = None, 
                         file_format: str = "csv", log_gaps: bool = True):
    """
    End-to-end convenience wrapper: load -> build features/targets -> save.

    Args:
        csv_path: Input raw OHLCV CSV path.
        interval: Sanity check/metadata for loader (e.g., "5m").
        expected_symbol: If provided, assert/verify symbol consistency on load.
        windows_cfg: Indicator parameter configuration.
        horizons: Horizons (in bars) for target construction.
        task: "binclass" | "reg" | "multiclass".
        thresh: Threshold for classification targets.
        out_path: If provided, save the final dataset to this path (folder or file).
        file_format: "parquet" | "csv". Only used if out_path is provided.
        log_gaps: Whether to print detected time gaps.

    Returns:
        A tuple: (df_out, feature_cols, target_cols, saved_path_or_None)
    """
    # Load and basic preparation (schema, time index, symbol checks, etc.)
    df = load_and_prepare(csv_path, interval=interval, expected_symbol=expected_symbol)
    # Build features + targets; no splitting here
    df_out, feature_cols, target_cols = build_features(df, windows_cfg, horizons, task, thresh, log_gaps)
    saved_path = None
    if (out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if (out_path.is_dir()):
            # Derive a filename from the source name
            stem = Path(csv_path).stem
            saved_path = out_path / f"{stem}_features.{"parquet" if file_format=="parquet" else "csv"}"
        else:
            saved_path = out_path
        # Save only features + targets (index preserved)
        to_save = df_out[feature_cols + target_cols]
        if (file_format.lower() == "parquet"):
            to_save.to_parquet(saved_path, index=True)
        else:
            to_save.to_csv(saved_path, index=True)
        print(f"[INFO] Features dataset saved to: {saved_path}")

    return df_out, feature_cols, target_cols, saved_path