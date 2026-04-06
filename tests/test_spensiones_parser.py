import pandas as pd

from ingestion.spensiones import SPensionesClient


def test_parse_html_and_normalize():
    html = """
    <html><body>
      <table>
        <tr><th>Fecha</th><th>Valor Cuota</th></tr>
        <tr><td>01-03-2026</td><td>55.123,45</td></tr>
        <tr><td>02-03-2026</td><td>55.200,10</td></tr>
      </table>
    </body></html>
    """
    raw = SPensionesClient.parse_html(html)
    client = SPensionesClient()
    out = client._normalize(raw, source_url="http://example")
    assert len(out) == 2
    assert set(["date", "valor_cuota", "benchmark", "source_url"]).issubset(out.columns)
    assert out["valor_cuota"].iloc[0] > 55000


def test_parse_csv():
    csv = "Fecha;Valor Cuota\n01-03-2026;55.123,45\n"
    raw = SPensionesClient.parse_csv(csv)
    assert isinstance(raw, pd.DataFrame)
    assert not raw.empty

