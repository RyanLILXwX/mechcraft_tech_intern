# Raw Data Dictionary

## The corresponding index of the column shown in the official document

In the Binance REST API documentation, `/api/v3/klines` returns each K-line as an array list, and the fields in it are in a fixed order.

We use Binance v3 API instead of v1 because v3 is the stable, officially supported version with more features, fewer limitations, and better reliability for accurate data.

Website: https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data

- Index 0 is open_time
- Index 1 is open
- Index 2 is high
- Index 3 is low
- Index 4 is close
- Index 5 is volume
- Index 6 is close_time
- Index 7 is quote_volume
- Index 8 is trades
- Index 9 is taker_buy_base
- Index 10 is taker_buy_quote
- Index 11 is the ignore field (usually not used)

## symbol

- **Financial**: Trading pair identifier in BASE/QUOTE notation (e.g., BTCUSDT). Determines the asset being bought/sold (base) and the currency it is priced in (quote). Critical for interpreting price level, liquidity, volatility regimes, and trading hours, patterns unique to each market.
- **Programming**: String key used for grouping, sorting, partitioning, and join keys. All time‑series computations (rolling windows, returns) should be performed **within** each `symbol`. Case is typically uppercase; treat as categorical to avoid unnecessary memory.
- **Calculation**: Direct field returned by the exchange for each kline row (in Binance Spot Klines, this corresponds to the trading pair requested). No transformation other than reading as string.

### open_time

- **Financial**: Candlestick **start time** (open) in milliseconds since UNIX epoch (UTC). Defines the temporal bucket to which all OHLCV measures belong for that bar. Aligns trades and events at the beginning of the interval.
- **Programming**: 64‑bit integer time key; primary sort/index within each symbol. Convert to timezone‑aware datetime when needed (`pd.to_datetime(open_time, unit='ms', utc=True)`). Expect strict monotonic increase per symbol at fixed step sizes equal to the configured interval.
- **Calculation**: Directly from the exchange kline payload (Binance array index 0). No arithmetic; downstream may validate that `next_open_time = open_time + interval_ms`.

### open

- **Financial**: First traded price in the interval. Serves as the reference price at bar start, which can differ from previous close due to gaps.
- **Programming**: Floating‑point input to return/indicator calculations; often used with `close` to compute bar direction. Validate numeric parsing (many APIs return numbers as strings).
- **Calculation**: Direct field from the kline (Binance index 1). Parsed to `float64`.

### high

- **Financial**: Maximum traded price within the interval. Captures the upper extreme that is useful for range and breakout measures.
- **Programming**: Float used in volatility, ATR, and Bollinger computations. Check invariants with `low`/`open`/`close`.
- **Calculation**: Direct field from kline (Binance index 2). Parsed to `float64`.

### low

- **Financial**: Minimum traded price within the interval. Captures the lower extreme that is combined with `high` for range/risk.
- **Programming**: Float used in ATR, spread, and pattern features. Should satisfy `low ≤ min(open, close) ≤ max(open, close) ≤ high`.
- **Calculation**: Direct field from kline (Binance index 3). Parsed to `float64`.

### close

- **Financial**: Last traded price in the interval. Widely used as the canonical price for returns, trend, and most indicators.
- **Programming**: Primary price series for returns (`pct_change`, logreturns), moving averages, and labels. Float64 recommended for numeric stability.
- **Calculation**: Direct field from kline (Binance index 4). Parsed to `float64`.

### volume

- **Financial**: Total **base asset** quantity traded during the interval (e.g., BTC for BTCUSDT). Key liquidity measure and activity proxy.
- **Programming**: Float feature used for liquidity filters, rolling volume stats, and constructing turnover (`close * volume`). Zero values may occur on illiquid markets; handle divisions safely.
- **Calculation**: Direct field from kline (Binance index 5). Parsed to `float64`.

### close_time

- **Financial**: Candlestick **end time** (close) in milliseconds since UNIX epoch (UTC). Marks the end of trading aggregation for the bar.
- **Programming**: 64‑bit integer timestamp typically equal to `open_time + interval_ms` at bar boundary (some venues use end‑exclusive conventions). Usually used for validation rather than indexing.
- **Calculation**: Direct field from kline (Binance index 6). No transformation; downstream may sanity‑check against `open_time`.

### quote_volume

- **Financial**: Total traded **quote currency** value during the interval (e.g., USDT for BTCUSDT). More comparable across assets than base `volume` because it is monetary.
- **Programming**: Float feature for turnover/liquidity. Prefer `quote_volume` over `close*volume` when available, since it aggregates executed trade values at actual trade prices.
- **Calculation**: Direct field from kline (Binance index 7). Parsed to `float64`. Not derived from other columns in this dataset.

### trades

- **Financial**: Count of executed trades in the interval. Microstructure intensity metric. Higher counts often correlate with tighter spreads and more stable price discovery.
- **Programming**: Integer feature; combine with `volume` to infer average trade size (`volume / trades` where `trades>0`). Useful for rolling means/stds and anomaly detection.
- **Calculation**: Direct field from kline (Binance index 8). Parsed to integer.

### taker_buy_base

- **Financial**: Volume of **aggressive buy** executions in base asset terms (quantity lifted by market buys). Gauges buy‑side pressure/order‑flow imbalance.
- **Programming**: Float used for flow ratios and imbalance features (e.g., `taker_buy_base / volume`). Sensitive to zero denominators; handle with NaN or 0 by policy.
- **Calculation**: Direct field from kline (Binance index 9). Parsed to `float64`.

### taker_buy_quote

- **Financial**: Monetary value of **aggressive buy** executions in quote currency. Complements `taker_buy_base` by putting buy pressure in currency terms.
- **Programming**: Float used for monetary‑side pressure ratios (e.g., `taker_buy_quote / quote_volume`) and to cross‑validate `taker_buy_base` using contemporaneous prices.
- **Calculation**: Direct field from kline (Binance index 10). Parsed to `float64`. No derivation from other columns.