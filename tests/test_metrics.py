from __future__ import annotations

import math

import pandas as pd
import pytest

from src.core.metrics import (
    PERIODS_PER_YEAR,
    MetricsError,
    calcular_matriz_correlaciones,
    calcular_matriz_covarianzas,
    calcular_rendimiento_esperado_anualizado,
    calcular_rendimientos,
    calcular_volatilidad_anualizada,
)
from src.data.loader import DEFAULT_CSV_PATH, load_portfolio_data


def test_rendimientos_mensuales_simples() -> None:
    precios = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    rendimientos = calcular_rendimientos(precios)
    assert isinstance(rendimientos, pd.Series)
    assert len(rendimientos) == 2
    assert rendimientos.iloc[0] == pytest.approx(0.10)
    assert rendimientos.iloc[1] == pytest.approx(0.10)


def test_rendimiento_anualizado_multiplicativo_no_lineal() -> None:
    # 1% mensual constante → (1.01)^12 - 1, no 0.01 * 12
    rendimientos = pd.Series([0.01] * 24)
    anual = calcular_rendimiento_esperado_anualizado(rendimientos)
    assert anual == pytest.approx((1.01**PERIODS_PER_YEAR) - 1)
    assert anual != pytest.approx(0.12)


def test_volatilidad_anualizada_multiplicativa() -> None:
    rendimientos = pd.Series([0.02, -0.01, 0.015, 0.005])
    mensual = rendimientos.std(ddof=1)
    anual = calcular_volatilidad_anualizada(rendimientos)
    assert anual == pytest.approx(mensual * math.sqrt(PERIODS_PER_YEAR))


def test_covarianza_y_correlacion() -> None:
    rendimientos = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.005, 0.015],
            "B": [0.008, 0.018, -0.002, 0.012],
        }
    )
    cov = calcular_matriz_covarianzas(rendimientos)
    corr = calcular_matriz_correlaciones(rendimientos)
    assert cov.shape == (2, 2)
    assert corr.shape == (2, 2)
    assert cov.loc["A", "A"] == pytest.approx(rendimientos["A"].var(ddof=1))
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["A", "B"] == pytest.approx(rendimientos["A"].corr(rendimientos["B"]))


def test_dataframe_multiples_activos() -> None:
    precios = pd.DataFrame(
        {
            "AAPL": [100.0, 105.0, 110.0],
            "MSFT": [200.0, 210.0, 199.0],
        },
        index=pd.date_range("2020-01-01", periods=3, freq="MS"),
    )
    rendimientos = calcular_rendimientos(precios)
    assert isinstance(rendimientos, pd.DataFrame)
    assert list(rendimientos.columns) == ["AAPL", "MSFT"]
    assert len(rendimientos) == 2

    esperado = calcular_rendimiento_esperado_anualizado(rendimientos)
    volatilidad = calcular_volatilidad_anualizada(rendimientos)
    assert isinstance(esperado, pd.Series)
    assert isinstance(volatilidad, pd.Series)
    assert set(esperado.index) == {"AAPL", "MSFT"}


def test_csv_real_fase_2_sin_optimizacion() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    subset = prepared.prices[["AAPL", "MSFT"]]
    rendimientos = calcular_rendimientos(subset)
    assert len(rendimientos) == prepared.n_periods - 1
    cov = calcular_matriz_covarianzas(rendimientos)
    corr = calcular_matriz_correlaciones(rendimientos)
    assert cov.shape == (2, 2)
    assert corr.loc["AAPL", "MSFT"] == pytest.approx(corr.loc["MSFT", "AAPL"])


def test_rechaza_precios_con_nan() -> None:
    precios = pd.Series([100.0, None, 110.0])
    with pytest.raises(MetricsError, match="faltantes"):
        calcular_rendimientos(precios)


def test_rechaza_rendimientos_insuficientes() -> None:
    with pytest.raises(MetricsError, match="al menos 2 observaciones"):
        calcular_volatilidad_anualizada(pd.Series([0.01]))
