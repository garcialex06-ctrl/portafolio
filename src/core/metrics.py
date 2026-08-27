"""
Fase 2 · Backend cuantitativo (Funcionalidad 2).

Funciones auxiliares para transformar precios en rendimientos y métricas
anualizadas. La optimización de portafolio (Markowitz, frontera eficiente,
máxima Sharpe) se implementará en la Fase 3; aquí no se calcula.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Frecuencia mensual del dataset (Fase 2). La anualización usa 12 periodos.
PERIODS_PER_YEAR = 12


class MetricsError(ValueError):
    """Error recuperable al calcular métricas de la Fase 2."""


def calcular_rendimientos(precios: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Fase 2: rendimientos históricos mensuales simples a partir de precios.

    Fórmula: r_t = P_t / P_{t-1} - 1

    Sin optimización (Fase 3).
    """
    if isinstance(precios, pd.Series):
        serie = _as_float_series(precios)
        if len(serie) < 2:
            raise MetricsError("Se necesitan al menos 2 precios para calcular rendimientos.")
        return serie.pct_change().dropna()

    frame = _as_float_frame(precios)
    if frame.shape[0] < 2:
        raise MetricsError("Se necesitan al menos 2 observaciones de precios.")
    if frame.shape[1] == 0:
        raise MetricsError("No hay columnas de precios para calcular rendimientos.")
    return frame.pct_change().dropna(how="all")


def calcular_rendimiento_esperado_anualizado(
    rendimientos: pd.DataFrame | pd.Series,
) -> pd.Series | float:
    """
    Fase 2: rendimiento esperado anualizado con compounding multiplicativo.

    Fórmula: (1 + mean(r_mensual))^12 - 1  (no mean(r_mensual) × 12)

    Sin optimización (Fase 3).
    """
    monthly = _validate_returns(rendimientos)
    if isinstance(monthly, pd.Series):
        return _annualize_mean_return(monthly.mean())
    return monthly.mean().apply(_annualize_mean_return)


def calcular_volatilidad_anualizada(
    rendimientos: pd.DataFrame | pd.Series,
) -> pd.Series | float:
    """
    Fase 2: volatilidad anualizada bajo independencia mensual.

    Fórmula: std(r_mensual) × sqrt(12)

    Sin optimización (Fase 3).
    """
    monthly = _validate_returns(rendimientos)
    if isinstance(monthly, pd.Series):
        return _annualize_volatility(monthly.std(ddof=1))
    return monthly.std(ddof=1).apply(_annualize_volatility)


def calcular_matriz_covarianzas(rendimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 2: matriz de covarianzas muestral de rendimientos mensuales.

    Sin optimización (Fase 3).
    """
    frame = _validate_returns_frame(rendimientos)
    return frame.cov()


def calcular_matriz_correlaciones(rendimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Fase 2: matriz de correlaciones de Pearson entre activos.

    Sin optimización (Fase 3).
    """
    frame = _validate_returns_frame(rendimientos)
    return frame.corr()


def _annualize_mean_return(mean_monthly: float) -> float:
    if not math.isfinite(mean_monthly):
        raise MetricsError("El rendimiento mensual promedio no es finito.")
    return float((1.0 + mean_monthly) ** PERIODS_PER_YEAR - 1.0)


def _annualize_volatility(std_monthly: float) -> float:
    if not math.isfinite(std_monthly):
        raise MetricsError("La volatilidad mensual no es finita.")
    if std_monthly < 0:
        raise MetricsError("La volatilidad mensual no puede ser negativa.")
    return float(std_monthly * math.sqrt(PERIODS_PER_YEAR))


def _validate_returns(rendimientos: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(rendimientos, pd.Series):
        serie = _as_float_series(rendimientos).dropna()
        if serie.empty:
            raise MetricsError("No hay rendimientos para calcular la métrica.")
        if len(serie) < 2:
            raise MetricsError("Se necesitan al menos 2 observaciones de rendimientos.")
        return serie
    return _validate_returns_frame(rendimientos)


def _validate_returns_frame(rendimientos: pd.DataFrame) -> pd.DataFrame:
    frame = _as_float_frame(rendimientos).dropna(how="all")
    if frame.empty:
        raise MetricsError("No hay rendimientos para calcular la métrica.")
    if frame.shape[1] < 1:
        raise MetricsError("Se requiere al menos una columna de rendimientos.")
    if frame.shape[0] < 2:
        raise MetricsError("Se necesitan al menos 2 observaciones de rendimientos.")
    return frame


def _as_float_series(precios: pd.Series) -> pd.Series:
    serie = pd.to_numeric(precios, errors="coerce")
    if serie.isna().any():
        raise MetricsError("Los precios contienen valores no numéricos o faltantes.")
    return serie.astype(float)


def _as_float_frame(precios: pd.DataFrame) -> pd.DataFrame:
    frame = precios.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise MetricsError("Los precios contienen valores no numéricos o faltantes.")
    return frame.astype(float)
