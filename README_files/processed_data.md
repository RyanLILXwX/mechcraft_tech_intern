# Processed Data Dictionary

This document explains each column in the processed dataset, with detailed notes on - **financial meaning**, **programming use**, and **calculation**.

---

## symbol

- **Financial**: Identifies the trading pair (e.g., BTCUSDT). Defines the asset being traded; critical for context since features and behaviors differ across assets.
- **Programming**: String identifier used for grouping and partitioning. All rolling calculations are performed within each symbol group.
- **Calculation**: Directly from exchange API metadata.

---

### ts

- **Financial**: Represents the candlestick open time (start of interval). Used to align trades in chronological order.
- **Programming**: Integer timestamp (ms since epoch). Converted to datetime for resampling or visualization.
- **Calculation**: Provided by exchange. Typically `open_time` field.

### open, high, low, close

- **Financial**: Price levels for each interval. Essential for technical analysis.
- **Programming**: Float values used in return calculations, indicators, and volatility features.
- **Calculation**: From exchange Kline data.

### volume

- **Financial**: Quantity of base asset traded. Reflects liquidity.
- **Programming**: Float feature for activity level. Used in rolling statistics and turnover measures.
- **Calculation**: From exchange Kline data (base asset volume).

### close_time

- **Financial**: End time of candlestick.
- **Programming**: Integer timestamp, rarely used in modeling.
- **Calculation**: From exchange Kline data.

### quote_volume

- **Financial**: Total traded value in quote asset (e.g., USDT). Liquidity measure.
- **Programming**: Float feature alternative to dollar_volume.
- **Calculation**: From exchange, not recomputed.

### trades

- **Financial**: Number of executed trades. Microstructure measure of market activity.
- **Programming**: Integer feature, often combined with volume to assess average trade size.
- **Calculation**: From exchange.

### taker_buy_base, taker_buy_quote

- **Financial**: Market order buy volume in base and quote terms. Indicates demand pressure.
- **Programming**: Floats used to build buy ratios. Capture order flow imbalance.
- **Calculation**: From exchange fields.

### ret_simple

- **Financial**: Simple percentage return between closes. Captures price change magnitude.
- **Programming**: Float input to models. Sensitive to scale.
- **Calculation**: `(close_t - close_{t-1}) / close_{t-1}`.

### ret_log

- **Financial**: Logarithmic return. Preferred for normality and additivity.
- **Programming**: Float, used for volatility calculations.
- **Calculation**: `log(close_t / close_{t-1})`.

### hl_spread

- **Financial**: Intrabar price dispersion. Proxy for volatility.
- **Programming**: Float feature capturing high-low relative movement.
- **Calculation**: `(high - low) / close`.

### sma_5, sma_10, sma_20, sma_50

- **Financial**: Simple moving averages. Trend-following measures over N periods.
- **Programming**: Float features representing smoothed price signals.
- **Calculation**: Rolling mean of close price with window N.

### ema_5, ema_10, ema_20, ema_50

- **Financial**: Exponential moving averages give more weight to recent prices.
- **Programming**: Float features, faster reaction than SMA.
- **Calculation**: `ewm(span=N, adjust=False).mean()` of close.

### vol_sma_5, vol_sma_10, vol_sma_20, vol_sma_50

- **Financial**: Average traded volume; liquidity baselines.
- **Programming**: Float features measuring sustained activity levels.
- **Calculation**: Rolling mean of volume.

### rsi_14

- **Financial**: Relative Strength Index, momentum oscillator 0–100.
- **Programming**: Feature for momentum, with thresholds at 70/30.
- **Calculation**: Smoothed ratio of gains to losses over 14 periods using Wilder’s method.

### macd_12_26, macd_signal_9, macd_hist

- **Financial**: Moving Average Convergence Divergence. Shows momentum and trend shifts.
- **Programming**: Used for crossover signals and momentum strength.
- **Calculation**: MACD = EMA(12) - EMA(26); signal = EMA(9) of MACD; hist = MACD - signal.

### bb_mid_20, bb_up_20, bb_low_20, bb_percent_b

- **Financial**: Bollinger Bands. Measure relative deviation from mean.
- **Programming**: Features to capture overextension.
- **Calculation**: Mid = SMA(20); upper/lower = mid ± 2×std; %B = (close - lower)/(upper - lower).

### atr_14

- **Financial**: Average True Range, volatility indicator.
- **Programming**: Float capturing risk/volatility magnitude.
- **Calculation**: Rolling average of true range over 14 periods.

### vol_logret_12, vol_logret_24, vol_logret_72

- **Financial**: Realized volatility measures over short/medium/long horizons.
- **Programming**: Float inputs for volatility modeling.
- **Calculation**: Rolling std of log returns over N periods.

### dollar_volume

- **Financial**: Dollar turnover. Key liquidity measure.
- **Programming**: Float used for scaling volume features.
- **Calculation**: `close volume`.

### taker_buy_ratio

- **Financial**: Proportion of buy-side trades in volume.
- **Programming**: Float capturing order flow imbalance.
- **Calculation**: `taker_buy_base / volume`.

### taker_buy_quote_ratio

- **Financial**: Proportion of buy-side trades in quote volume.
- **Programming**: Similar to taker_buy_ratio but in monetary terms.
- **Calculation**: `taker_buy_quote / quote_volume`.

### vol_roll_mean_5, vol_roll_mean_20, vol_roll_mean_50

- **Financial**: Average volume over past N periods.
- **Programming**: Baseline liquidity context.
- **Calculation**: Rolling mean of volume.

### vol_roll_std_5, vol_roll_std_20, vol_roll_std_50

- **Financial**: Variability of traded volume.
- **Programming**: Feature for volatility of activity.
- **Calculation**: Rolling std of volume.

### trades_roll_mean_5, trades_roll_mean_20, trades_roll_mean_50

- **Financial**: Average number of trades recently.
- **Programming**: Feature for execution intensity.
- **Calculation**: Rolling mean of trades.

### trades_roll_std_5, trades_roll_std_20, trades_roll_std_50

- **Financial**: Variability in number of trades.
- **Programming**: Captures irregular bursts of trading.
- **Calculation**: Rolling std of trades.

### dollar_vol_roll_mean_5, dollar_vol_roll_mean_20, dollar_vol_roll_mean_50

- **Financial**: Average turnover in USD terms.
- **Programming**: Feature for liquidity trend.
- **Calculation**: Rolling mean of dollar_volume.

### target_up_1, target_up_5, target_up_15

- **Financial**: Future price direction labels for classification.
- **Programming**: Binary supervised learning targets.
- **Calculation**: Compare close price to shifted close (1, 5, or 15 steps ahead).