from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ingestion.spensiones import SPensionesClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark simulation vs Fondo A AFP UNO (Cuenta 2 proxy)")
    p.add_argument("--capital-usd", type=float, default=10_000.0)
    p.add_argument("--start-history", default="2010-01-01")
    p.add_argument("--end-date", default=date.today().isoformat())
    p.add_argument("--outdir", default="data/processed/benchmark")
    return p.parse_args()


def fetch_usdclp() -> float:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/CLP=X?range=5d&interval=1d"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    js = resp.json()["chart"]["result"][0]
    closes = [c for c in js["indicators"]["quote"][0]["close"] if c is not None]
    if not closes:
        raise ValueError("No USDCLP close data")
    return float(closes[-1])


def rolling_12m_distribution(df: pd.DataFrame) -> pd.Series:
    s = df.sort_values("date").set_index("date")["valor_cuota"]
    monthly = s.resample("ME").last().dropna()
    r12 = monthly.pct_change(12).dropna()
    if r12.empty:
        raise ValueError("Insufficient Fondo A history for rolling 12m returns")
    return r12


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_history)
    end = date.fromisoformat(args.end_date)

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    sp = SPensionesClient()
    fondo_df = sp.fetch_fondo_a(start_date=start, end_date=end, out_csv=outdir / "fondo_a_history.csv")
    r12 = rolling_12m_distribution(fondo_df)
    usdclp = fetch_usdclp()

    capital_usd = float(args.capital_usd)
    capital_clp = capital_usd * usdclp

    latest = fondo_df.sort_values("date").iloc[-1]
    cuota_hoy = float(latest["valor_cuota"])
    cuotas = capital_clp / cuota_hoy

    pct = {
        "ret_12m_p5": float(r12.quantile(0.05)),
        "ret_12m_mediana": float(r12.quantile(0.50)),
        "ret_12m_p95": float(r12.quantile(0.95)),
        "ret_12m_promedio": float(r12.mean()),
    }

    sim_rows: list[dict[str, float | str]] = []
    for name, ret in [
        ("bajista_p5", pct["ret_12m_p5"]),
        ("base_mediana", pct["ret_12m_mediana"]),
        ("alcista_p95", pct["ret_12m_p95"]),
    ]:
        final_clp = capital_clp * (1.0 + ret)
        final_usd = final_clp / usdclp
        sim_rows.append(
            {
                "escenario": name,
                "retorno_12m": ret,
                "capital_inicial_usd": capital_usd,
                "capital_final_usd": final_usd,
                "capital_inicial_clp": capital_clp,
                "capital_final_clp": final_clp,
            }
        )
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(outdir / "fondo_a_12m_scenarios.csv", index=False)

    summary = {
        "asof": pd.to_datetime(latest["date"]).date().isoformat(),
        "capital_usd": capital_usd,
        "usdclp": usdclp,
        "capital_clp": capital_clp,
        "fondo_a_valor_cuota_hoy": cuota_hoy,
        "cuotas_compradas_hoy": cuotas,
        "returns_12m": pct,
        "n_rolling_obs": int(r12.shape[0]),
    }
    with (outdir / "fondo_a_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Benchmark simulation complete.")
    print(json.dumps(summary, indent=2))
    print(sim_df.to_string(index=False))


if __name__ == "__main__":
    main()
