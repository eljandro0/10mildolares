import pandas as pd

from backtest.metrics import annualized_sharpe, max_drawdown, rolling_alpha, win_rate


def test_annualized_sharpe_positive():
    ret = pd.Series([0.01, 0.0, 0.005, -0.002, 0.004])
    assert annualized_sharpe(ret) > 0


def test_max_drawdown():
    curve = pd.Series([100, 105, 103, 95, 97, 110])
    dd = max_drawdown(curve)
    assert round(dd, 4) == round((95 / 105) - 1, 4)


def test_win_rate():
    p = pd.Series([0.01, -0.01, 0.02, 0.0])
    b = pd.Series([0.0, -0.02, 0.01, 0.01])
    assert win_rate(p, b) == 0.5


def test_rolling_alpha():
    p = pd.Series([0.01] * 40)
    b = pd.Series([0.005] * 40)
    r = rolling_alpha(p, b, window=30)
    assert abs(r.iloc[-1] - 0.005) < 1e-9

