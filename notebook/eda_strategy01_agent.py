"""CLI-friendly version of the strategy 01 agent notebook."""

import argparse
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from investor_agent import InvestorAgent
from result_analysis import persist_results, report_final_outcome, report_simulation_summary
from stock_agent import build_stock_agents, compute_marketcap_timeseries, prepare_parameters


def load_environment(env_path: Path) -> Path:
    env_path = env_path.expanduser().resolve()
    env_file = env_path / ".env" if env_path.is_dir() else env_path
    if not env_file.exists():
        raise FileNotFoundError(f"Unable to locate .env file at {env_file}")
    load_dotenv(env_file)
    data_dir = os.environ.get("CAPIQ_DATA_DIR")
    if not data_dir:
        raise KeyError("CAPIQ_DATA_DIR must be defined in the environment")
    return Path(data_dir)


def load_capiq_frames(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    compinfo_file = data_dir / "comp_naics_code_common_stock_kr.parquet"
    compprice_file = data_dir / "capiq_price_equity_kr.parquet"
    compmarketcap_file = data_dir / "capiq_marketcap_kr.parquet"
    df_compinfo = pl.read_parquet(str(compinfo_file))
    df_compprice = pl.read_parquet(str(compprice_file))
    df_mktcap = pl.read_parquet(str(compmarketcap_file))
    return df_compinfo, df_compprice, df_mktcap


def run_simulation(
    stocks: list[dict],
    simul_dates: list,
    *,
    minimum_records: int,
    z_vol: float,
    vwap_rel_std_max: float,
    budget: float,
    ticker_filter: set[str] | None,
    enhanced_sell: bool,
) -> tuple[dict, dict, dict, float]:
    investor = InvestorAgent(
        budget=budget,
        z_vol=z_vol,
        vwap_rel_std_max=vwap_rel_std_max,
        enhanced_sell=enhanced_sell,
    )
    simuldates_done: list = []

    for simuldate in tqdm(simul_dates, desc="Restructuring stocks for simulation"):
        if len(simuldates_done) < minimum_records:
            simuldates_done.append(simuldate)
            continue

        for stock in stocks:
            prepared = prepare_parameters(stock, simuldate, minimum_records, ticker_filter)
            if not prepared:
                continue
            investor.process_stock(simuldate, prepared)

    return investor.results()


def parse_args() -> argparse.Namespace:
    default_env = Path(__file__).resolve().parents[1] / ".env"
    parser = argparse.ArgumentParser(description="Run the strategy 01 agent simulation.")
    parser.add_argument("--env-path", default=default_env, type=Path, help="Path to the .env file or its parent directory.")
    parser.add_argument(
        "--output-dir",
        default=Path.home() / "Downloads",
        type=Path,
        help="Directory where pickle outputs will be written.",
    )
    parser.add_argument("--minimum-records", default=20, type=int, help="Minimum lookback window for signals.")
    parser.add_argument("--min-agent-days", default=240, type=int, help="Minimum history days required to keep an agent.")
    parser.add_argument("--budget", default=100_000_000, type=float, help="Total simulation budget.")
    parser.add_argument("--z-vol", default=1.0, type=float, help="Z-score threshold for volume popularity.")
    parser.add_argument(
        "--vwap-rel-std-max",
        default=0.005,
        type=float,
        help="Relative VWAP std threshold for stability.",
    )
    parser.add_argument(
        "--ticker",
        dest="tickers",
        action="append",
        help="Ticker(s) to include. Repeat for multiple tickers. Defaults to ['A010420'].",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Process all tickers instead of the default filter.",
    )
    parser.add_argument(
        "--enhanced-sell",
        action="store_true",
        help="Enable enhanced selling logic for trade exits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        data_dir = load_environment(args.env_path)
    except Exception as exc:  # pragma: no cover - CLI feedback
        print(f"Failed to initialise environment: {exc}")
        return 1

    df_compinfo, df_compprice, df_mktcap = load_capiq_frames(data_dir)
    stocks = build_stock_agents(
        df_compinfo,
        df_compprice,
        df_mktcap,
        min_history_days=args.min_agent_days,
    )
    if not stocks:
        print("No stocks met the minimum history requirement.")
        return 1

    _, _, simul_dates = compute_marketcap_timeseries(stocks)
    if not simul_dates:
        print("No simulation dates available after aggregating market data.")
        return 1

    if args.all_tickers:
        ticker_filter = None
    else:
        ticker_filter = set(args.tickers) if args.tickers else {"A010420"}

    simulation_results, shares_owned, revenue_records, cash_balance = run_simulation(
        stocks,
        simul_dates,
        minimum_records=args.minimum_records,
        z_vol=args.z_vol,
        vwap_rel_std_max=args.vwap_rel_std_max,
        budget=args.budget,
        ticker_filter=ticker_filter,
        enhanced_sell=args.enhanced_sell,
    )

    report_simulation_summary(simulation_results, shares_owned, revenue_records)
    persist_results(args.output_dir, "strategy01", simulation_results, shares_owned, revenue_records)
    report_final_outcome(cash_balance, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
