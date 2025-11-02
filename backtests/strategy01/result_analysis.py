"""Utilities for persisting and reporting simulation results."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def persist_results(
    output_dir: Path,
    prefix: str,
    simulation_results: Mapping[str, Sequence],
    shares_owned: Mapping[str, Sequence],
    revenue_records: Mapping[str, Sequence],
) -> None:
    """Persist simulation artefacts to pickle files under the given directory."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"simulation_results_{prefix}.pkl", "wb") as f:
        pickle.dump(simulation_results, f)
    with open(output_dir / f"shares_owned_{prefix}.pkl", "wb") as f:
        pickle.dump(shares_owned, f)
    with open(output_dir / f"revenue_records_{prefix}.pkl", "wb") as f:
        pickle.dump(revenue_records, f)

    generate_company_revenue_analysis(output_dir, prefix, shares_owned, revenue_records)


def report_simulation_summary(
    simulation_results: Mapping[str, Sequence],
    shares_owned: Mapping[str, Sequence],
    revenue_records: Mapping[str, Sequence],
) -> None:
    """Print a concise overview of simulation artefacts for quick inspection."""
    print(
        "len simulation_results: {results}, shares_owned: {holdings}, revenue_records: {revenues}".format(
            results=len(simulation_results),
            holdings=len(shares_owned),
            revenues=len(revenue_records),
        )
    )


def report_final_outcome(cash_balance: float, output_dir: Path) -> None:
    """Emit a short message about the final balance and where artefacts were stored."""
    print(f"Final cash balance: {cash_balance:,.0f}")
    print(f"Results written to {output_dir.expanduser().resolve()}")


DEFAULT_COMPINFO_FILENAME = "comp_naics_code_common_stock_kr.parquet"


def generate_company_revenue_analysis(
    output_dir: Path,
    prefix: str,
    shares_owned: Mapping[str, Sequence],
    revenue_records: Mapping[str, Sequence],
    compinfo_path: Path | None = None,
) -> Path | None:
    """Create an Excel overview of revenue performance for each company."""
    revenue_df = _prepare_revenue_dataframe(revenue_records, shares_owned)
    if revenue_df.empty:
        print("No revenue records available for Excel export; skipping company analysis.")
        return None

    compinfo_df = _load_company_info(compinfo_path)
    if compinfo_df.empty:
        print("Company info parquet missing; skipping company revenue analysis export.")
        return None

    merged_df = revenue_df.merge(
        compinfo_df,
        left_on="ticker",
        right_on="tickerSymbol",
        how="left",
    )
    grouped = merged_df.groupby("ticker")
    revenue_by_ticker = grouped["revenue"].sum().sort_values(ascending=False)
    revenue_rank_df = revenue_by_ticker.to_frame().merge(
        compinfo_df,
        left_index=True,
        right_on="tickerSymbol",
        how="left",
    )

    profit_bin_counts = _bin_profit_distribution(merged_df)
    revenue_rank_df = revenue_rank_df.merge(
        profit_bin_counts,
        left_on="tickerSymbol",
        right_index=True,
        how="left",
    )
    if "total_trades" in revenue_rank_df:
        numerator = revenue_rank_df.get("p_above20", pd.Series(0, index=revenue_rank_df.index)).fillna(0)
        denominator = revenue_rank_df["total_trades"].replace(0, np.nan)
        revenue_rank_df["p_above20_ratio"] = (numerator / denominator).fillna(0)
    else:
        revenue_rank_df["p_above20_ratio"] = 0

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_prefix = prefix or "default"
    output_path = output_dir / f"company_revenue_analysis_{sanitized_prefix}.xlsx"
    revenue_rank_df.to_excel(output_path, index=False)
    print(f"Company revenue analysis written to {output_path}")
    return output_path


def _prepare_revenue_dataframe(
    revenue_records: Mapping[str, Sequence],
    shares_owned: Mapping[str, Sequence],
) -> pd.DataFrame:
    """Combine revenue snapshots with holdings-derived detail for analysis."""
    revenue_df = _flatten_revenue_records(revenue_records)
    holdings_sales_df = _extract_sales_from_holdings(shares_owned)

    if revenue_df.empty and holdings_sales_df.empty:
        return pd.DataFrame()

    if revenue_df.empty:
        df = holdings_sales_df
    elif holdings_sales_df.empty:
        df = revenue_df
    else:
        df = pd.concat([revenue_df, holdings_sales_df], ignore_index=True, sort=False)
        df.sort_values(["ticker", "sell_date"], inplace=True)
        df.drop_duplicates(
            subset=["ticker", "sell_date", "sold_price", "shares_sold"],
            keep="last",
            inplace=True,
        )

    if "sell_date" in df.columns:
        df["sell_date"] = pd.to_datetime(df["sell_date"], errors="coerce")

    numeric_cols = ["revenue", "shares_sold", "sold_price", "bought_price"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    # derive revenue when holdings supply enough data
    derived_revenue = np.where(
        df["shares_sold"].notna() & df["bought_price"].notna() & df["sold_price"].notna(),
        df["shares_sold"].abs() * (df["sold_price"] - df["bought_price"]),
        np.nan,
    )
    derived_series = pd.Series(derived_revenue, index=df.index)
    fill_mask = df["revenue"].isna() & derived_series.notna()
    df.loc[fill_mask, "revenue"] = derived_series[fill_mask]

    df.dropna(subset=["ticker", "sell_date", "revenue"], inplace=True)
    if df.empty:
        return df

    df["profit_pct"] = np.where(
        df["bought_price"] > 0,
        (df["sold_price"] - df["bought_price"]) / df["bought_price"],
        np.nan,
    )

    return df


def _flatten_revenue_records(revenue_records: Mapping[str, Sequence]) -> pd.DataFrame:
    """Normalise stored revenue logs to a tabular structure."""
    flattened: list[dict[str, Any]] = []
    for ticker, rows in revenue_records.items():
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            record = dict(row)
            record.setdefault("ticker", ticker)
            flattened.append(record)

    if not flattened:
        return pd.DataFrame()

    return pd.DataFrame(flattened)


def _extract_sales_from_holdings(shares_owned: Mapping[str, Sequence]) -> pd.DataFrame:
    """Derive sell transactions from holdings history to enrich revenue data."""
    sale_rows: list[dict[str, Any]] = []
    for ticker, entries in shares_owned.items():
        buy_price_lookup: dict[Any, float] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            buy_date = entry.get("buy_date")
            buy_price = entry.get("buy_price")
            if buy_date is not None and buy_price is not None and entry.get("shares", 0) >= 0:
                buy_price_lookup[buy_date] = float(buy_price)

        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            sell_date = entry.get("sold_date")
            sold_price = entry.get("sold_price")
            if sell_date is None or sold_price is None:
                continue
            shares_sold = abs(float(entry.get("shares", 0)))
            if shares_sold == 0:
                continue
            buy_date = entry.get("buy_date")
            buy_price = entry.get("buy_price")
            if buy_price is None and buy_date in buy_price_lookup:
                buy_price = buy_price_lookup[buy_date]
            revenue = entry.get("revenue")
            if revenue is None and buy_price is not None:
                revenue = shares_sold * (float(sold_price) - float(buy_price))

            sale_rows.append(
                {
                    "ticker": ticker,
                    "sell_date": sell_date,
                    "shares_sold": shares_sold,
                    "sold_price": float(sold_price),
                    "bought_price": float(buy_price) if buy_price is not None else np.nan,
                    "revenue": float(revenue) if revenue is not None else np.nan,
                    "buy_date": buy_date,
                }
            )

    if not sale_rows:
        return pd.DataFrame()

    return pd.DataFrame(sale_rows)


def _bin_profit_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Attach profit bins and count occurrences for later ratios."""
    if df.empty:
        return pd.DataFrame()

    bins = [-np.inf, -0.2, -0.05, 0.05, 0.2, np.inf]
    labels = ["p_neg20below", "p_neg5to-20", "p_neg5to5", "p_5to20", "p_above20"]
    df = df.copy()
    df["profit_bin"] = pd.cut(df["profit_pct"], bins=bins, labels=labels)

    counts = df.groupby(["ticker", "profit_bin"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=labels, fill_value=0)
    counts["total_trades"] = counts.sum(axis=1)
    return counts


def _load_company_info(compinfo_path: Path | None) -> pd.DataFrame:
    """Load company meta data and trim to relevant descriptive columns."""
    resolved_path = None
    if compinfo_path:
        resolved_path = Path(compinfo_path).expanduser().resolve()
    else:
        env_dir = os.environ.get("CAPIQ_DATA_DIR")
        if env_dir:
            candidate = Path(env_dir).expanduser() / DEFAULT_COMPINFO_FILENAME
            if candidate.exists():
                resolved_path = candidate.resolve()

    if resolved_path is None or not resolved_path.exists():
        return pd.DataFrame()

    compinfo = pd.read_parquet(resolved_path)
    desired_columns = [
        "tickerSymbol",
        "companyName",
        "indu_desc",
        "desc_1",
        "desc_2",
        "desc_3",
        "desc_4",
        "desc_5",
    ]
    available_columns = [col for col in desired_columns if col in compinfo.columns]
    deduped = compinfo[available_columns].drop_duplicates(subset=["tickerSymbol"], keep="first")
    return deduped
