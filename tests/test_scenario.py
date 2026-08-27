from src.config.scenario import (
    DEFAULT_RF_ANNUAL,
    annual_to_monthly,
    build_scenario,
    resolve_risk_free_rate,
)

AVAILABLE = ("AAPL", "MSFT", "V", "NVDA")


def _ok(**kwargs):
    defaults = {
        "available_tickers": AVAILABLE,
        "selected_tickers": AVAILABLE,
        "forced_weights_percent": {},
        "rf_annual_percent": None,
    }
    defaults.update(kwargs)
    return build_scenario(**defaults)


def test_empty_rf_uses_4_percent_annual_and_monthly() -> None:
    rate, used_default = resolve_risk_free_rate(None)
    assert used_default is True
    assert rate == DEFAULT_RF_ANNUAL
    assert annual_to_monthly(rate) == DEFAULT_RF_ANNUAL / 12

    result = _ok(rf_annual_percent=None)
    assert result.ok
    assert result.scenario is not None
    assert result.scenario.used_default_rf is True
    assert result.scenario.rf_annual == 0.04
    assert result.scenario.rf_monthly == 0.04 / 12


def test_custom_rf_is_converted_to_monthly() -> None:
    result = _ok(rf_annual_percent=5.0)
    assert result.ok
    assert result.scenario is not None
    assert result.scenario.used_default_rf is False
    assert result.scenario.rf_annual == 0.05
    assert result.scenario.rf_monthly == 0.05 / 12


def test_allows_mildly_negative_rf() -> None:
    result = _ok(rf_annual_percent=-2.0)
    assert result.ok
    assert result.scenario is not None
    assert result.scenario.rf_annual == -0.02


def test_rejects_extreme_negative_rf() -> None:
    result = _ok(rf_annual_percent=-20.0)
    assert not result.ok
    assert any("negativos extremos" in error for error in result.errors)


def test_rejects_nonsensical_high_rf() -> None:
    result = _ok(rf_annual_percent=80.0)
    assert not result.ok
    assert any("no puede superar" in error for error in result.errors)


def test_forced_visa_weight_leaves_free_weight() -> None:
    result = _ok(forced_weights_percent={"V": 15.0})
    assert result.ok
    assert result.scenario is not None
    assert result.scenario.forced_weights["V"] == 0.15
    assert abs(result.scenario.free_weight - 0.85) < 1e-12


def test_rejects_forced_weights_above_100_percent() -> None:
    result = _ok(forced_weights_percent={"V": 60.0, "AAPL": 50.0})
    assert not result.ok
    assert any("no puede superar 100%" in error for error in result.errors)


def test_rejects_forced_weight_on_deselected_asset() -> None:
    result = _ok(
        selected_tickers=("AAPL", "MSFT"),
        forced_weights_percent={"V": 15.0},
    )
    assert not result.ok
    assert any("no está incluido" in error for error in result.errors)


def test_rejects_single_asset_universe() -> None:
    result = _ok(selected_tickers=("AAPL",))
    assert not result.ok
    assert any("al menos 2 activos" in error for error in result.errors)


def test_rejects_empty_selection() -> None:
    result = _ok(selected_tickers=())
    assert not result.ok
    assert any("al menos un activo" in error for error in result.errors)


def test_rejects_full_allocation_with_leftover_assets() -> None:
    result = _ok(forced_weights_percent={"V": 100.0})
    assert not result.ok
    assert any("suman 100%" in error for error in result.errors)


def test_allows_fully_specified_forced_portfolio() -> None:
    result = _ok(
        selected_tickers=("V", "AAPL"),
        forced_weights_percent={"V": 40.0, "AAPL": 60.0},
    )
    assert result.ok
    assert result.scenario is not None
    assert abs(result.scenario.free_weight) < 1e-12


def test_rejects_missing_forced_weight_value() -> None:
    result = _ok(forced_weights_percent={"V": None})
    assert not result.ok
    assert any("Indica el peso forzado de V" in error for error in result.errors)


def test_rejects_unknown_selected_ticker() -> None:
    result = _ok(selected_tickers=("AAPL", "FAKE"))
    assert not result.ok
    assert any("no pertenecen al universo válido" in error for error in result.errors)
