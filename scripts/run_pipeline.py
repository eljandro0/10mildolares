from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest.engine import run_backtest  # noqa: E402
from common.io import read_csv_required, write_csv  # noqa: E402
from ingestion.spensiones import SPensionesClient  # noqa: E402
from portfolio.selector import load_racional_catalog  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent portfolio pipeline vs Fondo A benchmark")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--scenario", default="balanceado", choices=["agresivo", "balanceado", "defensivo"])
    p.add_argument("--rebalance", default="monthly", choices=["weekly", "monthly"])
    p.add_argument("--capital", type=float, default=10_000_000)
    p.add_argument("--fetch-benchmark", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    raw_dir = ROOT / "data" / "raw"
    cat_dir = ROOT / "data" / "catalog"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = raw_dir / "fondo_a_valor_cuota.csv"
    if args.fetch_benchmark or not benchmark_path.exists():
        client = SPensionesClient()
        benchmark_df = client.fetch_fondo_a(start, end, out_csv=benchmark_path)
    else:
        benchmark_df = read_csv_required(benchmark_path)
        benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
        benchmark_df = benchmark_df[
            (benchmark_df["date"] >= pd.Timestamp(start)) & (benchmark_df["date"] <= pd.Timestamp(end))
        ].copy()

    prices_path = raw_dir / "instrument_prices.csv"
    prices_df = read_csv_required(prices_path)
    prices_df["date"] = pd.to_datetime(prices_df["date"])
    needed_price_cols = {"date", "ticker", "close"}
    missing = needed_price_cols.difference(prices_df.columns)
    if missing:
        raise ValueError(f"instrument_prices.csv missing columns: {sorted(missing)}")

    catalog = load_racional_catalog(str(cat_dir / "racional_instruments.csv"))

    result = run_backtest(
        price_df=prices_df,
        benchmark_df=benchmark_df,
        catalog_df=catalog,
        initial_capital=args.capital,
        rebalance=args.rebalance,
        scenario=args.scenario,
    )

    write_csv(result.equity_curve, out_dir / "equity_curve.csv")
    write_csv(result.metrics, out_dir / "metrics.csv")
    write_csv(result.orders, out_dir / "orders.csv")
    write_csv(result.weights, out_dir / "weights.csv")
    decision_log = pd.DataFrame(
        [
            {
                "date": pd.Timestamp.utcnow().isoformat(),
                "event": "run_pipeline",
                "details": f"scenario={args.scenario}; rebalance={args.rebalance}; capital={args.capital}",
            },
            {
                "date": pd.Timestamp.utcnow().isoformat(),
                "event": "policy",
                "details": "Portfolio selection independent of Fondo A composition",
            },
        ]
    )
    write_csv(decision_log, out_dir / "decision_log.csv")

    print("Pipeline completed.")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()

