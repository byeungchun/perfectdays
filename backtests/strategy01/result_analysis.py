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

    generate_company_revenue_analysis(output_dir, prefix, revenue_records)


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
    revenue_records: Mapping[str, Sequence],
    compinfo_path: Path | None = None,
) -> Path | None:
    """Create an Excel overview of revenue performance for each company."""
    revenue_df = _prepare_revenue_dataframe(revenue_records)
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
) -> pd.DataFrame:
    """Flatten nested revenue records and clean columns for analysis."""
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

    df = pd.DataFrame(flattened)
    if "sell_date" in df.columns:
        df["sell_date"] = pd.to_datetime(df["sell_date"], errors="coerce")
    numeric_cols = ["revenue", "shares_sold", "sold_price", "bought_price"]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df.dropna(subset=["ticker", "sell_date", "revenue"], inplace=True)
    if df.empty:
        return df

    if "bought_price" in df.columns and "sold_price" in df.columns:
        df["profit_pct"] = np.where(
            df["bought_price"] > 0,
            (df["sold_price"] - df["bought_price"]) / df["bought_price"],
            np.nan,
        )
    else:
        df["profit_pct"] = np.nan

    return df


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
