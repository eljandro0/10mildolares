from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy.regime_aware import (  # noqa: E402
    build_recommended_weights,
    compare_entry_methods,
    compute_market_indicators,
    fetch_price_panel,
    run_event_study,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Asymmetric Rebalancing + Regime-Aware Strategy")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--capital", type=float, default=100_000.0)
    p.add_argument("--outdir", default="data/processed/regime_strategy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    universe = ["SPY", "VOO", "VT", "QQQ", "SMH", "NVDA", "MSFT", "XLE", "XOM", "ITA", "LMT"]
    price_df = fetch_price_panel(universe, start=date(2019, 1, 1), end=end, include_vix=True)
    market = compute_market_indicators(price_df)
    event_study = run_event_study(price_df, tickers=universe)
    weights = build_recommended_weights(event_study)
    bt = compare_entry_methods(
        price_df=price_df,
        investable_weights=weights,
        market_regime=market.regime,
        start=start,
        end=end,
        initial_capital=args.capital,
    )
    bt = bt.__class__(
        metrics=bt.metrics,
        equity_curves=bt.equity_curves,
        event_study=event_study,
        deployment_log=bt.deployment_log,
    )

    price_df.to_csv(outdir / "price_panel.csv", index=False)
    event_study.to_csv(outdir / "event_study.csv", index=False)
    bt.metrics.to_csv(outdir / "metrics.csv", index=False)
    bt.equity_curves.to_csv(outdir / "equity_curves.csv", index=False)
    bt.deployment_log.to_csv(outdir / "deployment_log.csv", index=False)

    summary = {
        "asof": market.asof.isoformat(),
        "market_regime": market.regime,
        "spy_close": market.spy_close,
        "spy_sma50": market.spy_sma50,
        "spy_sma200": market.spy_sma200,
        "spy_drawdown_252": market.spy_drawdown_252,
        "spy_mom_63d": market.spy_mom_63d,
        "vix_close": market.vix_close,
        "vix_pctile_252": market.vix_pctile_252,
        "weights": {k: float(v) for k, v in weights.to_dict().items()},
    }
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Regime strategy done. Outputs in {outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
