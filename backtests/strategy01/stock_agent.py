"""Stock agent representation for company pricing and market-cap data."""

from __future__ import annotations

import pandas as pd
import polars as pl
from tqdm import tqdm


class StockAgent:
    """Encapsulates a company's pricing and market capitalization timeseries."""

    def __init__(self, info: pl.DataFrame, price: pl.DataFrame, mktcap: pl.DataFrame):
        self.info = info
        self.price = price
        self.mktcap = mktcap
        self.ts_prices = self._build_timeseries()

    def _build_timeseries(self) -> pd.DataFrame:
        price_ts = self.price.select([
            pl.col("pricingDate"),
            pl.col("VWAP").cast(pl.Int64),
            pl.col("priceClose").cast(pl.Int64),
            pl.col("volume").cast(pl.Int64),
        ]).sort("pricingDate")
        mktcap_ts = self.mktcap.select([
            pl.col("pricingDate"),
            pl.col("marketCap").cast(pl.Int64),
            pl.col("sharesOutstanding").cast(pl.Int64),
        ]).sort("pricingDate")
        merged_ts = price_ts.join(mktcap_ts, on="pricingDate", how="inner")
        return merged_ts.to_pandas()

    def summary(self) -> dict:
        return {
            "companyName": (
                f"{self.info['companyName'][0]}, "
                f"tradingItemId: {self.info['tradingItemId'][0]}, "
                f"companyId: {self.info['companyId'][0]}"
            ),
            "industry": (
                f"Symbol: {self.info['tickerSymbol'][0]}, "
                f"Industry: {self.info['indu_desc'][0]}, "
                f"Desc: {self.info['desc_1'][0]}, "
                f"Desc2: {self.info['desc_2'][0]}"
            ),
            "price_head": f"row count: {self.price.height}, columns: {self.price.columns}",
            "mktcap_head": f"row count: {self.mktcap.height}, columns: {self.mktcap.columns}",
        }


def build_stock_agents(
    df_compinfo: pl.DataFrame,
    df_compprice: pl.DataFrame,
    df_mktcap: pl.DataFrame,
    *,
    min_history_days: int,
) -> list[dict]:
    df_compinfo2 = df_compinfo.unique(subset=["tickerSymbol"], keep="first").to_pandas()
    df_compinfo2.dropna(subset=["tradingItemId", "tickerSymbol"], inplace=True)
    df_compinfo2["tradingItemId"] = df_compinfo2["tradingItemId"].astype(int)
    df_compinfo2["companyId"] = df_compinfo2["companyId"].astype(int)
    print(f"Unique companies count: {len(df_compinfo2)}")

    stocks: list[dict] = []
    for _, rec in tqdm(df_compinfo2.iterrows(), total=len(df_compinfo2), desc="Creating stocks", disable=True):
        ticker = rec["tickerSymbol"]
        trading_item_id = rec["tradingItemId"]
        comp_name = rec["companyName"]
        comp_id = rec["companyId"]
        info_df = df_compinfo.filter(pl.col("tickerSymbol") == ticker)
        price_df = df_compprice.filter(pl.col("tradingItemId") == trading_item_id)
        mktcap_df = df_mktcap.filter(pl.col("companyId") == comp_id)
        stock = StockAgent(info_df, price_df, mktcap_df)
        if stock.ts_prices.shape[0] < min_history_days:
            continue
        stocks.append({"ticker": ticker, "companyName": comp_name, "stock": stock})

    print(f"Created {len(stocks)} agents with sufficient data.")
    return stocks


def compute_marketcap_timeseries(stocks: list[dict]) -> tuple[pd.DataFrame, pd.Series, list[pd.Timestamp]]:
    ts_marketcap = []
    for stock in stocks:
        symbol = stock["ticker"]
        ts = stock["stock"].ts_prices[["pricingDate", "marketCap"]].set_index("pricingDate").iloc[:, 0]
        ts.name = symbol
        ts_marketcap.append(ts)

    ts_marketcap_df = pd.concat(ts_marketcap, axis=1)
    ts_total_marketcap = ts_marketcap_df.sum(axis=1)
    simul_dates = ts_marketcap_df.index.sort_values().to_list()
    if simul_dates:
        print(
            f"total simulation dates: {len(simul_dates)}, "
            f"start: {simul_dates[0]}, end: {simul_dates[-1]}"
        )
    return ts_marketcap_df, ts_total_marketcap, simul_dates


def prepare_parameters(
    stock: dict,
    simuldate: pd.Timestamp,
    minimum_records: int,
    ticker_filter: set[str] | None,
):
    ticker = stock["ticker"]
    if ticker_filter and ticker not in ticker_filter:
        return None

    tsobj = stock["stock"].ts_prices.copy()
    tsobj.set_index("pricingDate", inplace=True)
    tsobj = tsobj.sort_index()
    tsobj = tsobj.loc[:simuldate].iloc[-(minimum_records + 1) :]
    tsobj.dropna(subset=["VWAP", "priceClose", "volume", "marketCap", "sharesOutstanding"], inplace=True)
    if tsobj.empty or tsobj.shape[0] < (minimum_records + 1):
        return None

    tstarget = tsobj.iloc[-1].copy()
    ts_hist = tsobj.iloc[:-1]
    vwap_mean = ts_hist["VWAP"].mean()
    vwap_std = ts_hist["VWAP"].std()
    vol_mean = ts_hist["volume"].mean()
    vol_std = ts_hist["volume"].std()
    vol = tstarget["volume"]
    return ticker, ts_hist, tstarget, vwap_mean, vwap_std, vol_mean, vol_std, vol


def build_prepared_stock_signals(
    stock: dict,
    minimum_records: int,
) -> tuple[str, pd.DataFrame | None]:
    ticker = stock["ticker"]
    ts_prices = stock["stock"].ts_prices.copy()
    if ts_prices.empty:
        return ticker, None

    ts_prices = ts_prices.sort_values("pricingDate")
    ts_prices["pricingDate"] = pd.to_datetime(ts_prices["pricingDate"])
    ts_prices.set_index("pricingDate", inplace=True)

    required_columns = ["VWAP", "volume", "priceClose", "marketCap", "sharesOutstanding"]
    ts_prices = ts_prices.dropna(subset=required_columns)
    if ts_prices.shape[0] <= minimum_records:
        return ticker, None

    rolling_window = ts_prices["VWAP"].rolling(window=minimum_records, min_periods=minimum_records)
    ts_prices["vwap_mean"] = rolling_window.mean().shift(1)
    ts_prices["vwap_std"] = rolling_window.std().shift(1)

    vol_window = ts_prices["volume"].rolling(window=minimum_records, min_periods=minimum_records)
    ts_prices["vol_mean"] = vol_window.mean().shift(1)
    ts_prices["vol_std"] = vol_window.std().shift(1)

    ts_prices.dropna(subset=["vwap_mean", "vwap_std", "vol_mean", "vol_std"], inplace=True)
    if ts_prices.empty:
        return ticker, None

    return ticker, ts_prices
