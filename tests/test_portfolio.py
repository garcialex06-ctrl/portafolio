from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.metrics import PERIODS_PER_YEAR, calcular_rendimientos
from src.core.portfolio import (
    compare_equal_vs_optimized,
    compute_efficient_frontier,
    equal_weight_vector,
    evaluate_portfolio,
    maximize_sharpe,
    portfolio_expected_return_annualized,
)
from src.data.loader import DEFAULT_CSV_PATH, load_portfolio_data


def _toy_returns() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 60
    a = rng.normal(0.01, 0.04, size=n)
    b = 0.6 * a + rng.normal(0.008, 0.03, size=n)
    c = rng.normal(0.005, 0.02, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_equal_weights_sum_to_one() -> None:
    w = equal_weight_vector(["A", "B", "C", "D"])
    assert abs(w.sum() - 1.0) < 1e-12
    assert (w == 0.25).all()


def test_compound_portfolio_return_not_linear() -> None:
    mean_m = np.array([0.01, 0.01])
    w = np.array([0.5, 0.5])
    ann = portfolio_expected_return_annualized(w, mean_m)
    assert ann == pytest.approx((1.01**PERIODS_PER_YEAR) - 1)
    assert ann != pytest.approx(0.12)


def test_optimized_sharpe_at_least_equal_weight() -> None:
    returns = _toy_returns()
    comparison = compare_equal_vs_optimized(returns, rf_annual=0.04)
    assert comparison.optimized.sharpe >= comparison.equal_weight.sharpe - 1e-6
    assert abs(comparison.optimized.weights.sum() - 1.0) < 1e-6
    assert (comparison.optimized.weights >= -1e-8).all()


def test_forced_weight_is_respected() -> None:
    returns = _toy_returns()
    opt = maximize_sharpe(returns, rf_annual=0.04, forced_weights={"C": 0.20})
    assert opt.weights["C"] == pytest.approx(0.20, abs=1e-5)


def test_real_csv_comparison() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:8])
    returns = calcular_rendimientos(prepared.prices[tickers])
    comparison = compare_equal_vs_optimized(returns, rf_annual=0.04)
    assert comparison.equal_weight.expected_return > -1
    assert comparison.optimized.volatility > 0
    assert comparison.optimized.sharpe >= comparison.equal_weight.sharpe - 1e-4


def test_evaluate_rejects_bad_weight_sum() -> None:
    returns = _toy_returns()
    with pytest.raises(Exception, match="sumar 1"):
        evaluate_portfolio({"A": 0.5, "B": 0.2, "C": 0.1}, returns, rf_annual=0.04)


def test_efficient_frontier_has_points() -> None:
    returns = _toy_returns()
    frontier = compute_efficient_frontier(returns, rf_annual=0.04, n_points=15)
    assert len(frontier.volatilities) >= 2
    assert len(frontier.efficient_volatilities) >= 2
    gmv_idx = int(np.argmin(frontier.volatilities))
    assert gmv_idx > 0
    assert gmv_idx < len(frontier.volatilities) - 1


def test_efficient_frontier_monotonic_efficient_branch() -> None:
    """Rama eficiente: rendimiento y volatilidad crecen monótonamente."""
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:12])
    returns = calcular_rendimientos(prepared.prices[tickers])
    frontier = compute_efficient_frontier(returns, rf_annual=0.04, n_points=40)

    assert np.all(np.diff(frontier.efficient_expected_returns) >= -1e-8)
    assert np.all(np.diff(frontier.efficient_volatilities) >= -1e-8)


def test_tangency_near_frontier() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    tickers = list(prepared.valid_tickers[:12])
    returns = calcular_rendimientos(prepared.prices[tickers])
    frontier = compute_efficient_frontier(returns, rf_annual=0.04)
    tg = frontier.tangency
    dist = np.min(
        np.hypot(
            frontier.efficient_volatilities - tg.volatility,
            frontier.efficient_expected_returns - tg.expected_return,
        )
    )
    assert dist < 0.02

def test_cml_passes_through_tangency() -> None:
    returns = _toy_returns()
    frontier = compute_efficient_frontier(returns, rf_annual=0.04)
    tg = frontier.tangency
    rf = frontier.rf_annual
    cml_at_tg = rf + tg.sharpe * tg.volatility
    assert cml_at_tg == pytest.approx(tg.expected_return, rel=1e-4)


def test_efficient_frontier_figure_builds() -> None:
    from utils.charts import efficient_frontier_figure

    returns = _toy_returns()
    frontier = compute_efficient_frontier(returns, rf_annual=0.04)
    fig = efficient_frontier_figure(frontier)
    assert len(fig.data) >= 4

