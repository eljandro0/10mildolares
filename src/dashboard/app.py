from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
INPUT_POSITIONS = ROOT / "data" / "inputs" / "positions.csv"
BENCHMARK_ENTRY = ROOT / "data" / "inputs" / "fondo_a_benchmark_entry.json"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    js = resp.json()["chart"]["result"][0]
    out = pd.DataFrame({"timestamp": js["timestamp"], "close": js["indicators"]["quote"][0]["close"]}).dropna()
    out["date"] = (
        pd.to_datetime(out["timestamp"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    out["ticker"] = ticker
    return out[["date", "ticker", "close"]]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_fondo_a_uno_series(start_year: int, end_year: int, fecconf: str | None = None) -> pd.DataFrame:
    conf = fecconf or datetime.now(UTC).strftime("%Y%m%d")
    url = "https://www.spensiones.cl/apps/valoresCuotaFondo/vcfAFPxls.php"
    params = {"aaaaini": str(start_year), "aaaafin": str(end_year), "tf": "A", "fecconf": conf}
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    txt = resp.content.decode("utf-8", errors="ignore")

    rows: list[dict[str, object]] = []
    for line in txt.splitlines():
        if not re.match(r"^\d{4}-\d{2}-\d{2};", line):
            continue
        parts = line.split(";")
        if len(parts) < 3:
            continue
        d = pd.to_datetime(parts[0].strip(), errors="coerce")
        v = pd.to_numeric(parts[-2].strip().replace(".", "").replace(",", "."), errors="coerce")
        if pd.notna(d) and pd.notna(v):
            rows.append({"date": d.normalize(), "valor_cuota": float(v)})

    out = pd.DataFrame(rows).dropna().sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        raise ValueError("No se pudo obtener serie Fondo A UNO")
    return out


@st.cache_data(ttl=900, show_spinner=False)
def fetch_usdclp_history(start: date, end: date) -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/CLP=X"
    params = {
        "period1": int(datetime.combine(start, datetime.min.time()).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()),
        "interval": "1d",
        "events": "history",
    }
    resp = requests.get(url, params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    js = resp.json()["chart"]["result"][0]
    out = pd.DataFrame({"timestamp": js["timestamp"], "usdclp": js["indicators"]["quote"][0]["close"]}).dropna()
    out["date"] = (
        pd.to_datetime(out["timestamp"], unit="s", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    return out[["date", "usdclp"]].sort_values("date")


def load_positions() -> pd.DataFrame:
    if not INPUT_POSITIONS.exists():
        raise FileNotFoundError(f"No existe {INPUT_POSITIONS}")
    df = pd.read_csv(INPUT_POSITIONS)
    needed = {"ticker", "quantity", "avg_cost"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"positions.csv sin columnas: {sorted(missing)}")
    df["ticker"] = df["ticker"].str.upper()
    return df


def load_benchmark_entry() -> dict[str, object] | None:
    if not BENCHMARK_ENTRY.exists():
        return None
    return json.loads(BENCHMARK_ENTRY.read_text(encoding="utf-8"))


def save_benchmark_entry(payload: dict[str, object]) -> None:
    BENCHMARK_ENTRY.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_ENTRY.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_portfolio_snapshot(positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = datetime.now(UTC).date()
    start = end - timedelta(days=420)

    hist = pd.concat(
        [fetch_price_history(tk, start, end) for tk in positions["ticker"].tolist()],
        ignore_index=True,
    ).sort_values(["date", "ticker"])
    latest_date = hist["date"].max()
    latest = hist[hist["date"] == latest_date][["ticker", "close"]].rename(columns={"close": "last_price"})
    snap = positions.merge(latest, on="ticker", how="left")
    snap["invested_usd"] = snap["quantity"] * snap["avg_cost"]
    snap["market_value_usd"] = snap["quantity"] * snap["last_price"]
    snap["pnl_usd"] = snap["market_value_usd"] - snap["invested_usd"]
    snap["pnl_pct"] = snap["market_value_usd"] / snap["invested_usd"] - 1.0
    total = snap["market_value_usd"].sum()
    snap["weight_now"] = snap["market_value_usd"] / total if total > 0 else 0.0
    return snap, hist


def build_portfolio_curve(positions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    piv = history.pivot(index="date", columns="ticker", values="close").sort_index()
    for tk in positions["ticker"]:
        if tk not in piv.columns:
            piv[tk] = pd.NA
    piv = piv[positions["ticker"].tolist()].ffill().dropna()
    qty = positions.set_index("ticker")["quantity"]
    invested = float((positions["quantity"] * positions["avg_cost"]).sum())
    curve = (piv * qty).sum(axis=1).rename("portfolio_value_usd").to_frame().reset_index()
    curve["invested_usd"] = invested
    return curve


def build_fondo_a_benchmark(initial_usd: float) -> tuple[pd.DataFrame, dict[str, object]]:
    today = datetime.now(UTC).date()
    cuota = fetch_fondo_a_uno_series(start_year=today.year - 2, end_year=today.year)
    last = cuota.iloc[-1]
    last_cuota = float(last["valor_cuota"])
    last_cuota_date = pd.to_datetime(last["date"]).date()

    fx_hist = fetch_usdclp_history(last_cuota_date - timedelta(days=20), today)
    fx_now = float(fx_hist.iloc[-1]["usdclp"])

    entry = load_benchmark_entry()
    if entry is None:
        entry = {
            "capital_usd": float(initial_usd),
            "entry_fx": fx_now,
            "entry_cuota": last_cuota,
            "entry_cuota_date": last_cuota_date.isoformat(),
            "initialized_at_utc": datetime.now(UTC).isoformat(),
        }
        save_benchmark_entry(entry)

    capital_usd = float(entry["capital_usd"])
    entry_fx = float(entry["entry_fx"])
    entry_cuota = float(entry["entry_cuota"])
    cuotas = capital_usd * entry_fx / entry_cuota

    start_date = pd.to_datetime(entry["entry_cuota_date"])
    cuota = cuota[cuota["date"] >= start_date].copy()
    fx = fetch_usdclp_history(start_date.date() - timedelta(days=10), today)
    curve = cuota.merge(fx, on="date", how="left").sort_values("date")
    curve["usdclp"] = curve["usdclp"].ffill().bfill()
    curve["benchmark_value_usd"] = cuotas * curve["valor_cuota"] / curve["usdclp"]
    curve["invested_usd"] = capital_usd

    final = float(curve.iloc[-1]["benchmark_value_usd"])
    snap = {
        "entry_cuota": entry_cuota,
        "last_cuota": last_cuota,
        "entry_date": str(entry["entry_cuota_date"]),
        "last_date": str(last_cuota_date),
        "invested_usd": capital_usd,
        "final_usd": final,
        "pnl_usd": final - capital_usd,
        "pnl_pct": final / capital_usd - 1.0,
    }
    return curve[["date", "benchmark_value_usd", "invested_usd"]], snap


def money(x: float) -> str:
    return f"USD {x:,.2f}"


def pct(x: float) -> str:
    return f"{x:+.2%}"


def render_professional_table(df: pd.DataFrame, pnl_cols: list[str]) -> None:
    def paint(val: str) -> str:
        try:
            numeric = float(val.replace(",", "").replace("%", ""))
        except Exception:
            return val
        cls = "pos" if numeric >= 0 else "neg"
        return f'<span class="{cls}">{val}</span>'

    html = ['<div class="pro-table-wrap"><table class="pro-table"><thead><tr>']
    html += [f"<th>{col}</th>" for col in df.columns]
    html += ["</tr></thead><tbody>"]
    for _, row in df.iterrows():
        html.append("<tr>")
        for col in df.columns:
            val = str(row[col])
            if col in pnl_cols:
                val = paint(val)
            html.append(f"<td>{val}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.markdown(
    """
<style>
  .stApp { background: #0b0e11; color: #eaecef; }
  .block-container { padding-top: 1rem; max-width: 1400px; }
  h1, h2, h3 { color: #f0b90b !important; }
  [data-testid="stMetricValue"] { color: #eaecef; }
  [data-testid="stMetricDelta"] { color: #8ddf8b; }
  .pro-table-wrap { background:#12161c; border:1px solid #1f2630; border-radius:10px; padding:10px; }
  .pro-table { width:100%; border-collapse:collapse; font-size:14px; }
  .pro-table thead th { text-align:left; background:#161b22; color:#f0b90b; padding:10px; border-bottom:1px solid #2a3441; }
  .pro-table tbody td { padding:9px 10px; border-bottom:1px solid #1f2630; color:#eaecef; }
  .pro-table tbody tr:hover { background:#141a22; }
  .pos { color:#0ecb81; font-weight:600; }
  .neg { color:#f6465d; font-weight:600; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Portfolio Monitor")
st.caption("Modo diario tipo exchange: posiciones, benchmark Fondo A y curva comparativa.")

with st.sidebar:
    st.write(f"Posiciones: `{INPUT_POSITIONS}`")
    if st.button("Actualizar"):
        st.cache_data.clear()
        st.rerun()
    if st.button("Reiniciar benchmark Fondo A"):
        if BENCHMARK_ENTRY.exists():
            BENCHMARK_ENTRY.unlink()
        st.cache_data.clear()
        st.rerun()

try:
    positions = load_positions()
    snap, hist = build_portfolio_snapshot(positions)
    port_curve = build_portfolio_curve(positions, hist)
    invested = float((positions["quantity"] * positions["avg_cost"]).sum())
    bench_curve, bench_snap = build_fondo_a_benchmark(invested)
except Exception as exc:
    st.error(f"Error cargando panel: {exc}")
    st.stop()

final_port = float(snap["market_value_usd"].sum())
port_pnl = final_port - invested
port_pnl_pct = final_port / invested - 1.0

st.subheader("1) Portafolio Actual")
c1, c2, c3 = st.columns(3)
c1.metric("Valor Final Portafolio", money(final_port))
c2.metric("Ganancia/Perdida", money(port_pnl), pct(port_pnl_pct))
c3.metric("Capital Inicial", money(invested))

tbl = snap[
    ["ticker", "weight_now", "avg_cost", "last_price", "pnl_usd", "pnl_pct", "market_value_usd"]
].copy()
tbl.columns = [
    "Accion",
    "% Actual",
    "Precio Compra",
    "Ultimo Precio",
    "Gan/Perd USD",
    "Gan/Perd %",
    "Valor Final USD",
]
tbl_fmt = tbl.copy()
tbl_fmt["% Actual"] = tbl_fmt["% Actual"].map(lambda x: f"{x:.2%}")
tbl_fmt["Precio Compra"] = tbl_fmt["Precio Compra"].map(lambda x: f"{x:,.2f}")
tbl_fmt["Ultimo Precio"] = tbl_fmt["Ultimo Precio"].map(lambda x: f"{x:,.2f}")
tbl_fmt["Gan/Perd USD"] = tbl_fmt["Gan/Perd USD"].map(lambda x: f"{x:+,.2f}")
tbl_fmt["Gan/Perd %"] = tbl_fmt["Gan/Perd %"].map(lambda x: f"{x:+.2%}")
tbl_fmt["Valor Final USD"] = tbl_fmt["Valor Final USD"].map(lambda x: f"{x:,.2f}")
render_professional_table(tbl_fmt, pnl_cols=["Gan/Perd USD", "Gan/Perd %"])

st.subheader("2) Benchmark: 100% Fondo A AFP UNO")
b1, b2, b3 = st.columns(3)
b1.metric("Valor Final Benchmark", money(float(bench_snap["final_usd"])))
b2.metric("Ganancia/Perdida", money(float(bench_snap["pnl_usd"])), pct(float(bench_snap["pnl_pct"])))
b3.metric("Capital Inicial", money(float(bench_snap["invested_usd"])))

bench_tbl = pd.DataFrame(
    [
        {
            "Activo": "Fondo A AFP UNO",
            "Precio Compra (Valor Cuota)": float(bench_snap["entry_cuota"]),
            "Ultimo Precio (Valor Cuota)": float(bench_snap["last_cuota"]),
            "Gan/Perd USD": float(bench_snap["pnl_usd"]),
            "Gan/Perd %": float(bench_snap["pnl_pct"]),
            "Valor Final USD": float(bench_snap["final_usd"]),
            "Fecha Compra": bench_snap["entry_date"],
            "Fecha Ultimo Precio": bench_snap["last_date"],
        }
    ]
)
bench_fmt = bench_tbl.copy()
bench_fmt["Precio Compra (Valor Cuota)"] = bench_fmt["Precio Compra (Valor Cuota)"].map(lambda x: f"{x:,.2f}")
bench_fmt["Ultimo Precio (Valor Cuota)"] = bench_fmt["Ultimo Precio (Valor Cuota)"].map(lambda x: f"{x:,.2f}")
bench_fmt["Gan/Perd USD"] = bench_fmt["Gan/Perd USD"].map(lambda x: f"{x:+,.2f}")
bench_fmt["Gan/Perd %"] = bench_fmt["Gan/Perd %"].map(lambda x: f"{x:+.2%}")
bench_fmt["Valor Final USD"] = bench_fmt["Valor Final USD"].map(lambda x: f"{x:,.2f}")
render_professional_table(bench_fmt, pnl_cols=["Gan/Perd USD", "Gan/Perd %"])

st.subheader("3) Evolucion: Portafolio vs Benchmark")
cmp = port_curve[["date", "portfolio_value_usd"]].merge(
    bench_curve[["date", "benchmark_value_usd"]],
    on="date",
    how="inner",
)
if cmp.empty:
    st.warning("Aun no hay fechas comunes para graficar comparacion.")
else:
    fig = px.line(
        cmp,
        x="date",
        y=["portfolio_value_usd", "benchmark_value_usd"],
        template="plotly_dark",
        labels={"value": "USD", "variable": "Serie"},
    )
    fig.update_layout(
        legend_title_text="",
        xaxis_title="Fecha",
        yaxis_title="Valor (USD)",
        paper_bgcolor="#0b0e11",
        plot_bgcolor="#0b0e11",
        font=dict(color="#eaecef"),
    )
    fig.for_each_trace(
        lambda t: t.update(
            name="Portafolio" if t.name == "portfolio_value_usd" else "Benchmark Fondo A"
        )
    )
    st.plotly_chart(fig, use_container_width=True)
