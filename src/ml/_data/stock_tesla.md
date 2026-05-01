# Tesla (TSLA) Trading Dataset - 5 Year Historical (Simulated)

This dataset provides a high-fidelity simulation of Tesla Inc. (TSLA) stock performance over a 5-year period, designed for algorithmic trading analysis, strategy backtesting, and market regime detection.

## Dataset Summary
- **Records**: ~1,304 Business Days
- **Timeframe**: Last 5 Years
- **Symbols Included**: TSLA, SPY (Benchmark)
- **Target Variables**: `Close`, `Volatility_20d`

## Feature Engineering & Data Quality

| Feature Name | Category | Description | Data Quality / complexity |
| :--- | :--- | :--- | :--- |
| **Date** | Temporal | Business date of the trading session. | Standard datetime objects. |
| **Date_Str** | Temporal | String representation of the date. | **Inconsistent**: Mix of YYYY-MM-DD and Unix timestamps. |
| **Open, High, Low, Close** | Price | Standard OHLC data. | Contains intentional **missing values** (NaN) and **extreme spikes** (1.5x) to simulate flash crashes. |
| **Volume** | Liquidity | Number of shares traded. | Large integers; used for liquidity analysis. |
| **SPY_Close** | Market | Closing price of the S&P 500 ETF (Benchmark). | Used for relative strength and beta calculations. |
| **Split_Factor** | Corporate | Multiplier for stock splits. | Includes historical splits: 5.0 (2020) and 3.0 (2022). |
| **SMA_50 / SMA_200** | Technical | Simple Moving Averages (50 and 200 day). | Standard momentum indicators. |
| **RSI** | Momentum | Relative Strength Index (14-day). | Values 0-100; identifies overbought/oversold conditions. |
| **Volatility_20d** | Risk | 20-day rolling annualized volatility. | Captures changes in market risk regimes. |
| **ATR** | Risk | Average True Range (14-day). | Measures daily price range and volatility. |
| **Rel_Strength_SPY**| Market | TSLA price / SPY price ratio. | Measures outperformance against the broader market. |

## Trading Analysis Challenges
1. **Split Adjustment**: The `Close` and `Open` prices are "raw". To conduct long-term trend analysis, the researcher must use the `Split_Factor` to back-adjust historical prices.
2. **Date Standardization**: The `Date_Str` column must be parsed to handle the mixed Unix/ISO formats.
3. **Outlier Filtering**: Flash crash spikes (1.5x) must be detected and filtered or smoothed to avoid training bias.
4. **Regime Change**: The high volatility of TSLA (simulated at 3% daily std dev) creates distinct "regimes" that challenge standard linear models.
