"""Utilities for persisting and reporting simulation results."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, Sequence


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
