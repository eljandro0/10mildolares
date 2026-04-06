from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common.io import write_csv


DEFAULT_SP_URL = "https://www.spensiones.cl/apps/valoresCuotaFondo/vcfAFP.php"


@dataclass(frozen=True)
class SPFetchConfig:
    url: str = DEFAULT_SP_URL
    afp_name: str = "UNO"
    timeout_seconds: int = 25


class SPensionesClient:
    """Fetch Fondo A benchmark from SP site.

    The endpoint can return either HTML or CSV-like content depending on params/site changes.
    This client parses both formats and keeps source traceability.
    """

    def __init__(self, config: SPFetchConfig | None = None) -> None:
        self.config = config or SPFetchConfig()

    def fetch_fondo_a(
        self,
        start_date: date,
        end_date: date,
        out_csv: Path | None = None,
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")
        payload = {
            "tf": "1",  # Fondo A
            "afp": self.config.afp_name,
            "fec_inicio": start_date.isoformat(),
            "fec_fin": end_date.isoformat(),
        }
        response = requests.get(
            self.config.url,
            params=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")

        if "text/csv" in content_type or "application/csv" in content_type:
            df = self.parse_csv(response.text)
        else:
            df = self.parse_html(response.text)

        df = self._normalize(df, source_url=response.url)
        mask = (df["date"] >= pd.Timestamp(start_date)) & (
            df["date"] <= pd.Timestamp(end_date)
        )
        df = df.loc[mask].copy()
        if out_csv is not None:
            write_csv(df, out_csv)
        return df

    @staticmethod
    def parse_html(html: str) -> pd.DataFrame:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None:
            raise ValueError("No table found in SP response")

        headers = [
            th.get_text(strip=True).lower().replace(" ", "_")
            for th in table.find_all("th")
        ]
        rows = []
        for tr in table.find_all("tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if tds:
                rows.append(tds)
        if not rows:
            raise ValueError("No data rows found in SP table")
        if headers and len(headers) == len(rows[0]):
            return pd.DataFrame(rows, columns=headers)
        return pd.DataFrame(rows)

    @staticmethod
    def parse_csv(raw: str) -> pd.DataFrame:
        # Detect common separators used in Chilean exports.
        sep = ";" if raw.count(";") > raw.count(",") else ","
        df = pd.read_csv(StringIO(raw), sep=sep)
        if df.empty:
            raise ValueError("Empty CSV from SP")
        return df

    @staticmethod
    def _find_col(columns: Iterable[str], candidates: list[str]) -> str:
        low_map = {c.lower().strip(): c for c in columns}
        for cand in candidates:
            if cand in low_map:
                return low_map[cand]
        raise KeyError(f"Missing expected columns. candidates={candidates}")

    def _normalize(self, raw_df: pd.DataFrame, source_url: str) -> pd.DataFrame:
        df = raw_df.copy()
        cols = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        df.columns = cols

        date_col = self._find_col(
            df.columns,
            ["fecha", "date", "fec", "f"],
        )
        value_col = self._find_col(
            df.columns,
            ["valor_cuota", "valorcuota", "valor", "value", "cuota"],
        )

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        value_txt = (
            df[value_col]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        out["valor_cuota"] = pd.to_numeric(value_txt, errors="coerce")
        out["benchmark"] = "FONDO_A_UNO"
        out["source_url"] = source_url
        out = out.dropna(subset=["date", "valor_cuota"]).sort_values("date")
        if out.empty:
            raise ValueError("No valid benchmark rows after parsing")
        return out

