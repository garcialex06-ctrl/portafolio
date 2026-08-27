from __future__ import annotations

import pytest

from src.core.backtest import (
    final_wealth_summary,
    historical_wealth_paths,
    optimized_weights_display_table,
    optimized_weights_table,
)
from src.core.metrics import calcular_rendimientos
from src.core.portfolio import maximize_sharpe
from src.data.loader import DEFAULT_CSV_PATH, load_portfolio_data


def test_optimized_weights_table_zeros_for_unselected() -> None:
    table = optimized_weights_table(
        universe_tickers=["A", "B", "C"],
        selected_tickers=["A", "C"],
        optimized_weights={"A": 0.4, "C": 0.6},
    )
    assert list(table.columns) == ["A", "B", "C"]
    assert table.loc["Peso", "A"] == pytest.approx(0.4)
    assert table.loc["Peso", "B"] == pytest.approx(0.0)
    assert table.loc["Peso", "C"] == pytest.approx(0.6)


def test_optimized_weights_display_table() -> None:
    table = optimized_weights_display_table(
        optimized_weights={"NVDA": 0.316, "IBM": 0.286, "AAPL": 0.15, "CVX": 0.248, "X": 0.0},
        forced_weights={"AAPL": 0.15},
    )
    assert list(table.columns) == ["Activo", "Porcentaje"]
    assert table.iloc[0]["Activo"] == "NVDA"
    assert table.iloc[0]["Porcentaje"] == "31.6%"
    assert any("🔴 AAPL" in row for row in table["Activo"])
    assert "X" not in table["Activo"].values
    assert table.iloc[-1]["Activo"] == "TOTAL"
    assert table.iloc[-1]["Porcentaje"] == "100.0%"


def test_historical_paths_start_at_100() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:6])
    returns = calcular_rendimientos(prepared.prices[tickers])
    opt = maximize_sharpe(returns, rf_annual=0.04)
    paths = historical_wealth_paths(
        prepared.prices,
        selected_tickers=tickers,
        optimized_weights=opt.weights,
        benchmark=prepared.benchmark,
    )
    assert paths.equal_weight.iloc[0] == pytest.approx(100.0)
    assert paths.optimized.iloc[0] == pytest.approx(100.0)
    assert paths.benchmark.iloc[0] == pytest.approx(100.0)
    assert len(paths.equal_weight) == len(paths.benchmark)


def test_forced_weight_appears_in_results_table() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:6])
    forced = {tickers[0]: 0.25}
    returns = calcular_rendimientos(prepared.prices[tickers])
    opt = maximize_sharpe(returns, rf_annual=0.04, forced_weights=forced)
    table = optimized_weights_table(
        universe_tickers=prepared.valid_tickers,
        selected_tickers=tickers,
        optimized_weights=opt.weights,
    )
    assert table.loc["Peso", tickers[0]] == pytest.approx(0.25, abs=1e-5)
    unselected = [t for t in prepared.valid_tickers if t not in tickers]
    assert unselected
    assert table.loc["Peso", unselected[0]] == pytest.approx(0.0)


def test_final_wealth_summary_and_figure() -> None:
    from utils.charts import historical_wealth_figure

    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:5])
    returns = calcular_rendimientos(prepared.prices[tickers])
    opt = maximize_sharpe(returns, rf_annual=0.04)
    paths = historical_wealth_paths(
        prepared.prices,
        selected_tickers=tickers,
        optimized_weights=opt.weights,
        benchmark=prepared.benchmark,
    )
    finals = final_wealth_summary(paths)
    assert finals["Optimizado"] > 0
    fig = historical_wealth_figure(paths)
    assert len(fig.data) == 3
