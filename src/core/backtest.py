"""
Validación histórica (Funcionalidad 4): pesos óptimos y evolución base 100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.core.portfolio import (
    PortfolioOptimizationError,
    equal_weight_vector,
    maximize_sharpe,
)

INITIAL_WEALTH = 100.0


@dataclass(frozen=True)
class HistoricalWealthPaths:
    """Series de valor acumulado (base 100) para equal, óptimo y benchmark."""

    equal_weight: pd.Series
    optimized: pd.Series
    benchmark: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp


def optimized_weights_table(
    *,
    universe_tickers: Sequence[str],
    selected_tickers: Sequence[str],
    optimized_weights: Mapping[str, float] | pd.Series,
) -> pd.DataFrame:
    """
    Tabla horizontal de pesos del portafolio optimizado.

    Incluye todo el universo del menú de Inputs: seleccionados con su peso óptimo
    (respetando forzados) y no seleccionados explícitamente en 0.
    """
    universe = [str(t) for t in universe_tickers]
    selected = {str(t) for t in selected_tickers}
    weights = pd.Series(optimized_weights, dtype=float)

    row: dict[str, float] = {}
    for ticker in universe:
        if ticker in selected:
            if ticker not in weights.index:
                raise PortfolioOptimizationError(
                    f"Falta el peso optimizado de {ticker}."
                )
            row[ticker] = float(weights[ticker])
        else:
            row[ticker] = 0.0

    table = pd.DataFrame([row], index=["Peso"])
    table.index.name = "Métrica"
    return table


def optimized_weights_display_table(
    *,
    optimized_weights: Mapping[str, float] | pd.Series,
    forced_weights: Mapping[str, float] | None = None,
    weight_threshold: float = 1e-4,
) -> pd.DataFrame:
    """
    Tabla vertical para la UI: solo activos con peso > 0, marcador de forzados y TOTAL.
    """
    forced = {str(t) for t in (forced_weights or {})}
    weights = pd.Series(optimized_weights, dtype=float).sort_values(ascending=False)

    rows: list[dict[str, str]] = []
    for ticker, weight in weights.items():
        w = float(weight)
        if w <= weight_threshold:
            continue
        marker = "🔴 " if str(ticker) in forced else ""
        rows.append(
            {
                "Activo": f"{marker}{ticker}",
                "Porcentaje": f"{w * 100:.1f}%",
            }
        )

    total_pct = float(weights.sum()) * 100.0
    rows.append({"Activo": "TOTAL", "Porcentaje": f"{total_pct:.1f}%"})
    return pd.DataFrame(rows)


def historical_wealth_paths(
    prices: pd.DataFrame,
    *,
    selected_tickers: Sequence[str],
    optimized_weights: Mapping[str, float] | pd.Series,
    benchmark: pd.Series,
    initial_wealth: float = INITIAL_WEALTH,
) -> HistoricalWealthPaths:
    """
    Evolución buy-and-hold desde `initial_wealth` (p. ej. $100 hace ~5 años).

    Pesos fijos al inicio; el valor en t es la revalorización proporcional
    de cada activo respecto al precio inicial.
    """
    tickers = [str(t) for t in selected_tickers]
    if len(tickers) < 1:
        raise PortfolioOptimizationError("Se requiere al menos un activo seleccionado.")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise PortfolioOptimizationError(
            "Faltan precios para: " + ", ".join(missing)
        )
    if benchmark is None or benchmark.empty:
        raise PortfolioOptimizationError(
            "No hay benchmark S&P 500 disponible para la validación histórica."
        )

    asset_prices = prices[tickers].astype(float)
    bench = benchmark.astype(float).rename(str(benchmark.name))

    aligned = asset_prices.join(bench, how="inner").dropna(how="any")
    if len(aligned) < 2:
        raise PortfolioOptimizationError(
            "No hay suficientes observaciones alineadas para la validación histórica."
        )

    asset_aligned = aligned[tickers]
    bench_aligned = aligned[str(benchmark.name)]

    equal_w = equal_weight_vector(tickers)
    opt_w = pd.Series(optimized_weights, dtype=float).reindex(tickers)
    if opt_w.isna().any():
        raise PortfolioOptimizationError(
            "Los pesos optimizados no cubren todos los activos seleccionados."
        )
    if abs(float(opt_w.sum()) - 1.0) > 1e-4:
        raise PortfolioOptimizationError(
            f"Los pesos optimizados deben sumar 1 (suma: {float(opt_w.sum()):.4f})."
        )

    equal_path = _buy_and_hold_wealth(asset_aligned, equal_w, initial_wealth)
    opt_path = _buy_and_hold_wealth(asset_aligned, opt_w, initial_wealth)
    bench_path = initial_wealth * (bench_aligned / float(bench_aligned.iloc[0]))

    return HistoricalWealthPaths(
        equal_weight=equal_path.rename("Pesos iguales"),
        optimized=opt_path.rename("Optimizado"),
        benchmark=bench_path.rename("S&P 500"),
        start=pd.Timestamp(aligned.index[0]),
        end=pd.Timestamp(aligned.index[-1]),
    )


def final_wealth_summary(paths: HistoricalWealthPaths) -> dict[str, float]:
    """Valor final de cada escenario (respuesta a $100 hace 5 años)."""
    return {
        "Pesos iguales": float(paths.equal_weight.iloc[-1]),
        "Optimizado": float(paths.optimized.iloc[-1]),
        "S&P 500": float(paths.benchmark.iloc[-1]),
    }


def _buy_and_hold_wealth(
    prices: pd.DataFrame,
    weights: pd.Series,
    initial_wealth: float,
) -> pd.Series:
    """Valor acumulado con asignación inicial fija (buy-and-hold)."""
    w = weights.reindex(prices.columns).to_numpy(dtype=float)
    base = prices.iloc[0].to_numpy(dtype=float)
    if np.any(base <= 0) or not np.all(np.isfinite(base)):
        raise PortfolioOptimizationError(
            "Hay precios iniciales inválidos para la validación histórica."
        )
    relative = prices.to_numpy(dtype=float) / base
    values = initial_wealth * (relative @ w)
    return pd.Series(values, index=prices.index, dtype=float)
