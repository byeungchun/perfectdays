"""CLI-friendly version of the strategy 01 agent notebook."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import polars as pl
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

if __package__ is None or __package__ == "":  # pragma: no cover - script execution fallback
    sys.path.append(str(Path(__file__).resolve().parent))
    from investor_agent import InvestorAgent  # type: ignore
    from result_analysis import (  # type: ignore
        persist_results,
        report_final_outcome,
        report_simulation_summary,
    )
    from stock_agent import (  # type: ignore
        build_prepared_stock_signals,
        build_stock_agents,
        compute_marketcap_timeseries,
    )
else:  # pragma: no cover - module execution
    from .investor_agent import InvestorAgent
    from .result_analysis import persist_results, report_final_outcome, report_simulation_summary
    from .stock_agent import build_prepared_stock_signals, build_stock_agents, compute_marketcap_timeseries

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import pandas as pd


def load_strategy_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = config_path.expanduser().resolve()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Strategy config must be a mapping of parameter names to values")
    return dict(data)


DEFAULT_GENERAL_SETTINGS: dict[str, Any] = {
    "env_path": (Path(__file__).resolve().parents[1] / ".env").resolve(),
    "output_dir": Path.home() / "Downloads",
    "tickers": None,
    "max_workers": None,
}


DEFAULT_STRATEGY_SETTINGS: dict[str, Any] = {
    "budget": 100_000_000.0,
    "z_vol": 1.0,
    "vwap_rel_std_max": 0.005,
    "enhanced_sell": False,
    "sell_after_days": 5,
    "min_hold_days": 0,
    "take_profit_pct": None,
    "stop_loss_pct": None,
    "trailing_stop_pct": None,
    "partial_sell_ratio": None,
    "prioritize_time_exit": False,
}


def _build_signal_lookup(
    stocks: list[dict],
    minimum_records: int,
    max_workers: int | None,
) -> dict[str, "pd.DataFrame"]:
    import pandas as pd  # local import to avoid mandatory dependency when unused

    def compute(stock: dict) -> tuple[str, pd.DataFrame | None]:
        return build_prepared_stock_signals(stock, minimum_records)

    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(compute, stocks))
    else:
        results = [compute(stock) for stock in stocks]

    lookup: dict[str, pd.DataFrame] = {}
    for ticker, frame in results:
        if frame is not None:
            lookup[ticker] = frame
    return lookup


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
    max_workers: int | None,
    sell_after_days: int,
    min_hold_days: int,
    take_profit_pct: float | None,
    stop_loss_pct: float | None,
    trailing_stop_pct: float | None,
    partial_sell_ratio: float | None,
    prioritize_time_exit: bool,
) -> tuple[dict, dict, dict, float]:
    investor = InvestorAgent(
        budget=budget,
        z_vol=z_vol,
        vwap_rel_std_max=vwap_rel_std_max,
        enhanced_sell=enhanced_sell,
        sell_after_days=sell_after_days,
        min_hold_days=min_hold_days,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        partial_sell_ratio=partial_sell_ratio,
        prioritize_time_exit=prioritize_time_exit,
    )
    simuldates_done: list = []
    signal_lookup = _build_signal_lookup(stocks, minimum_records, max_workers)

    for simuldate in tqdm(simul_dates, desc="Running simulation", disable=True):
        if len(simuldates_done) < minimum_records:
            simuldates_done.append(simuldate)
            continue

        for stock in stocks:
            ticker = stock["ticker"]
            if ticker_filter and ticker not in ticker_filter:
                continue

            signals = signal_lookup.get(ticker)
            if signals is None or simuldate not in signals.index:
                continue

            row = signals.loc[simuldate]
            prepared = (
                ticker,
                None,
                row,
                float(row["vwap_mean"]),
                float(row["vwap_std"]),
                float(row["vol_mean"]),
                float(row["vol_std"]),
                float(row["volume"]),
            )
            investor.process_stock(simuldate, prepared)

    return investor.results()


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().parents[3] / "config" / "strategy01.yaml"
    parser = argparse.ArgumentParser(description="Run the strategy 01 agent simulation.")
    parser.add_argument(
        "--env-path",
        default=None,
        type=Path,
        help="Path to the .env file or its parent directory.",
    )
    parser.add_argument(
        "--config",
        default=default_config,
        type=Path,
        help="Path to a YAML file containing InvestorAgent parameters.",
    )
    parser.add_argument(
    "--output-dir",
    default=None,
    type=Path,
    help="Directory where pickle outputs will be written.",
    )
    parser.add_argument("--minimum-records", default=20, type=int, help="Minimum lookback window for signals.")
    parser.add_argument("--min-agent-days", default=240, type=int, help="Minimum history days required to keep an agent.")
    parser.add_argument("--budget", type=float, default=None, help="Override total simulation budget from config.")
    parser.add_argument("--z-vol", type=float, default=None, help="Override Z-score threshold for volume popularity.")
    parser.add_argument(
        "--vwap-rel-std-max",
        default=None,
        type=float,
        help="Override relative VWAP std threshold for stability.",
    )
    parser.add_argument(
        "--ticker",
        dest="tickers",
        action="append",
        help="Ticker(s) to include. Repeat for multiple tickers. If omitted, all tickers are processed.",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Process all tickers (default when --ticker is not provided).",
    )
    parser.add_argument(
        "--enhanced-sell",
        dest="enhanced_sell",
        action="store_true",
        help="Enable enhanced selling logic regardless of config settings.",
    )
    parser.add_argument(
        "--no-enhanced-sell",
        dest="enhanced_sell",
        action="store_false",
        help="Disable enhanced selling logic regardless of config settings.",
    )
    parser.add_argument(
        "--sell-after-days",
        type=int,
        default=None,
        help="Override number of days after which to force a sale.",
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=None,
        help="Set minimum holding days required before rule-based exits.",
    )
    parser.add_argument(
        "--take-profit-pct",
        type=float,
        default=None,
        help="Take-profit proportion (e.g. 0.05 for a 5 percent gain).",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=None,
        help="Stop-loss proportion (e.g. 0.03 for a 3 percent drop).",
    )
    parser.add_argument(
        "--trailing-stop-pct",
        type=float,
        default=None,
        help="Trailing stop proportion based on peak price since entry.",
    )
    parser.add_argument(
        "--partial-sell-ratio",
        type=float,
        default=None,
        help="Fraction of holdings to sell when exit triggers (<1 keeps a residual position).",
    )
    parser.add_argument(
        "--prioritize-time-exit",
        dest="prioritize_time_exit",
        action="store_true",
        help="When enabled, time-based exits supersede other sell rules.",
    )
    parser.add_argument(
        "--no-prioritize-time-exit",
        dest="prioritize_time_exit",
        action="store_false",
        help="Ensure rule-based exits can trigger before the time-based exit.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Number of worker threads for per-stock signal precomputation. "
            "Values <= 1 run sequentially."
        ),
    )
    parser.set_defaults(enhanced_sell=None, prioritize_time_exit=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        strategy_cfg = load_strategy_config(args.config)
    except Exception as exc:
        print(f"Failed to load strategy config: {exc}")
        return 1

    config_path = args.config.expanduser().resolve() if args.config else None
    if config_path and not config_path.exists():
        print(f"Strategy config not found at {config_path}; relying on CLI defaults and overrides.")
    elif strategy_cfg:
        print(f"Loaded strategy config from {config_path}")
    else:
        print("Strategy config provided no values; relying on CLI defaults and overrides.")

    def resolve(name: str, cli_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if strategy_cfg and name in strategy_cfg and strategy_cfg[name] is not None:
            return strategy_cfg[name]
        return default

    def resolve_path(name: str, cli_value: Path | None, default: Path) -> Path:
        value = resolve(name, cli_value, default)
        if value is None:
            raise ValueError(f"Parameter '{name}' must be a valid path")
        if isinstance(value, Path):
            path_value = value
        else:
            path_value = Path(str(value))
        return path_value.expanduser().resolve()

    def resolve_numeric(
        name: str,
        cli_value: Any,
        default: Any,
        *,
        cast,
        allow_none: bool = False,
    ) -> Any:
        value = resolve(name, cli_value, default)
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"Parameter '{name}' is required but missing in config/CLI")
        try:
            return cast(value)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Parameter '{name}' must be {cast.__name__}-compatible, got {value!r}") from err

    def resolve_bool(name: str, cli_value: Any, default: bool) -> bool:
        value = resolve(name, cli_value, default)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        if isinstance(value, (int, float)) and value in {0, 1}:
            return bool(value)
        raise ValueError(f"Parameter '{name}' must be boolean-compatible, got {value!r}")

    try:
        env_path = resolve_path("env_path", args.env_path, DEFAULT_GENERAL_SETTINGS["env_path"])
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1

    try:
        data_dir = load_environment(env_path)
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

    try:
        output_dir = resolve_path("output_dir", args.output_dir, DEFAULT_GENERAL_SETTINGS["output_dir"])
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_tickers:
        ticker_filter = None
    else:
        if args.tickers:
            ticker_filter = set(args.tickers)
        else:
            cfg_tickers = resolve("tickers", None, DEFAULT_GENERAL_SETTINGS["tickers"])
            if cfg_tickers:
                if isinstance(cfg_tickers, str):
                    symbol = cfg_tickers.strip()
                    ticker_filter = {symbol} if symbol else None
                else:
                    try:
                        candidates = {str(t).strip() for t in cfg_tickers if str(t).strip()}
                        ticker_filter = candidates if candidates else None
                    except TypeError:
                        print("Configuration error: 'tickers' must be a string or iterable of strings.")
                        return 1
            else:
                ticker_filter = None

    cfg_max_workers = resolve("max_workers", args.max_workers, DEFAULT_GENERAL_SETTINGS["max_workers"])
    effective_max_workers: int | None
    if cfg_max_workers is None:
        effective_max_workers = None
    else:
        try:
            effective_max_workers = int(cfg_max_workers)
        except (TypeError, ValueError):
            print(f"Configuration error: 'max_workers' must be an integer-compatible value, got {cfg_max_workers!r}")
            return 1
        if effective_max_workers <= 0:
            effective_max_workers = None

    try:
        strategy_params = {
            "budget": resolve_numeric("budget", args.budget, DEFAULT_STRATEGY_SETTINGS["budget"], cast=float),
            "z_vol": resolve_numeric("z_vol", args.z_vol, DEFAULT_STRATEGY_SETTINGS["z_vol"], cast=float),
            "vwap_rel_std_max": resolve_numeric(
                "vwap_rel_std_max",
                args.vwap_rel_std_max,
                DEFAULT_STRATEGY_SETTINGS["vwap_rel_std_max"],
                cast=float,
            ),
            "enhanced_sell": resolve_bool(
                "enhanced_sell",
                args.enhanced_sell,
                DEFAULT_STRATEGY_SETTINGS["enhanced_sell"],
            ),
            "sell_after_days": resolve_numeric(
                "sell_after_days",
                args.sell_after_days,
                DEFAULT_STRATEGY_SETTINGS["sell_after_days"],
                cast=int,
            ),
            "min_hold_days": resolve_numeric(
                "min_hold_days",
                args.min_hold_days,
                DEFAULT_STRATEGY_SETTINGS["min_hold_days"],
                cast=int,
            ),
            "take_profit_pct": resolve_numeric(
                "take_profit_pct",
                args.take_profit_pct,
                DEFAULT_STRATEGY_SETTINGS["take_profit_pct"],
                cast=float,
                allow_none=True,
            ),
            "stop_loss_pct": resolve_numeric(
                "stop_loss_pct",
                args.stop_loss_pct,
                DEFAULT_STRATEGY_SETTINGS["stop_loss_pct"],
                cast=float,
                allow_none=True,
            ),
            "trailing_stop_pct": resolve_numeric(
                "trailing_stop_pct",
                args.trailing_stop_pct,
                DEFAULT_STRATEGY_SETTINGS["trailing_stop_pct"],
                cast=float,
                allow_none=True,
            ),
            "partial_sell_ratio": resolve_numeric(
                "partial_sell_ratio",
                args.partial_sell_ratio,
                DEFAULT_STRATEGY_SETTINGS["partial_sell_ratio"],
                cast=float,
                allow_none=True,
            ),
            "prioritize_time_exit": resolve_bool(
                "prioritize_time_exit",
                args.prioritize_time_exit,
                DEFAULT_STRATEGY_SETTINGS["prioritize_time_exit"],
            ),
        }
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 1

    for key in ("take_profit_pct", "stop_loss_pct", "trailing_stop_pct"):
        value = strategy_params[key]
        if value is not None and value < 0:
            print(f"Configuration error: '{key}' must be non-negative; received {value}")
            return 1
    ratio = strategy_params["partial_sell_ratio"]
    if ratio is not None and not 0 < ratio <= 1:
        print("Configuration error: 'partial_sell_ratio' must be between 0 and 1 (exclusive of 0).")
        return 1
    if strategy_params["sell_after_days"] < 0:
        print("Configuration error: 'sell_after_days' must be non-negative.")
        return 1
    if strategy_params["min_hold_days"] < 0:
        print("Configuration error: 'min_hold_days' must be non-negative.")
        return 1

    print(
        "Investor configuration:",
        {
            "budget": strategy_params["budget"],
            "z_vol": strategy_params["z_vol"],
            "vwap_rel_std_max": strategy_params["vwap_rel_std_max"],
            "enhanced_sell": strategy_params["enhanced_sell"],
            "sell_after_days": strategy_params["sell_after_days"],
            "min_hold_days": strategy_params["min_hold_days"],
        },
    )

    simulation_results, shares_owned, revenue_records, cash_balance = run_simulation(
        stocks,
        simul_dates,
        minimum_records=args.minimum_records,
        z_vol=strategy_params["z_vol"],
        vwap_rel_std_max=strategy_params["vwap_rel_std_max"],
        budget=strategy_params["budget"],
        ticker_filter=ticker_filter,
        enhanced_sell=strategy_params["enhanced_sell"],
        max_workers=effective_max_workers,
        sell_after_days=strategy_params["sell_after_days"],
        min_hold_days=strategy_params["min_hold_days"],
        take_profit_pct=strategy_params["take_profit_pct"],
        stop_loss_pct=strategy_params["stop_loss_pct"],
        trailing_stop_pct=strategy_params["trailing_stop_pct"],
        partial_sell_ratio=strategy_params["partial_sell_ratio"],
        prioritize_time_exit=strategy_params["prioritize_time_exit"],
    )

    report_simulation_summary(simulation_results, shares_owned, revenue_records)
    persist_results(output_dir, "strategy01", simulation_results, shares_owned, revenue_records)
    report_final_outcome(cash_balance, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
