# Alpha vs Fondo A

Pipeline de inversion independiente para superar al Fondo A de UNO AFP en horizonte de 6 meses.

## Objetivo
- Fondo A se usa solo como benchmark.
- La cartera es independiente y sin restricciones regulatorias del Fondo A.
- No hay ejecucion automatica de ordenes; solo recomendaciones.

## Estructura
- `src/ingestion/`: ingesta y parseo de benchmark + catalogos
- `src/portfolio/`: seleccion de activos y senales
- `src/optimization/`: optimizacion por escenarios
- `src/backtest/`: simulacion, metricas, ordenes
- `src/dashboard/`: app Streamlit

## Setup
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -e .[dev]
```

## Ejecutar pipeline
```bash
python scripts/run_pipeline.py --start 2024-01-01 --end 2025-12-31 --scenario balanceado --rebalance monthly
```

## Correr dashboard
```bash
streamlit run src/dashboard/app.py
```

## Publicar gratis (Streamlit Community Cloud)
1. Sube este proyecto a un repo en GitHub.
2. Entra a https://share.streamlit.io e inicia sesion con GitHub.
3. Crea una app nueva con:
   - Repository: tu repo
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. Deploy.

Notas:
- `requirements.txt` ya esta preparado (`-e .`) para instalar dependencias del proyecto.
- La app obtiene precios en linea (Yahoo/SP), por lo que necesita internet para actualizar datos.

## Tests
```bash
pytest -q
```

## Datos esperados
- Benchmark Fondo A desde SP en `data/raw/fondo_a_valor_cuota.csv`
- Catalogo Racional en `data/catalog/racional_instruments.csv`
- Precios historicos en `data/raw/instrument_prices.csv`
