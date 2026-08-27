"""
Optimización de portafolio en Python puro (NumPy + SciPy).

Sin PyPortfolioOpt. Calcula pesos iguales y máxima razón de Sharpe
con restricciones long-only y pesos forzados opcionales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.core.metrics import PERIODS_PER_YEAR

_WEIGHT_TOL = 1e-8
_SHARPE_EPS = 1e-12

__all__ = [
    "EfficientFrontierResult",
    "PortfolioComparison",
    "PortfolioMetrics",
    "PortfolioOptimizationError",
    "compare_equal_vs_optimized",
    "compute_efficient_frontier",
    "equal_weight_vector",
    "evaluate_portfolio",
    "maximize_sharpe",
    "portfolio_expected_return_annualized",
    "portfolio_sharpe",
    "portfolio_volatility_annualized",
]


class PortfolioOptimizationError(ValueError):
    """Error recuperable al optimizar o evaluar un portafolio."""


@dataclass(frozen=True)
class PortfolioMetrics:
    """Métricas anualizadas de un portafolio."""

    expected_return: float
    volatility: float
    sharpe: float
    weights: pd.Series


@dataclass(frozen=True)
class PortfolioComparison:
    equal_weight: PortfolioMetrics
    optimized: PortfolioMetrics


@dataclass(frozen=True)
class EfficientFrontierResult:
    """Frontera eficiente y puntos clave para el gráfico riesgo–rendimiento."""

    volatilities: np.ndarray
    expected_returns: np.ndarray
    inefficient_volatilities: np.ndarray
    inefficient_expected_returns: np.ndarray
    efficient_volatilities: np.ndarray
    efficient_expected_returns: np.ndarray
    min_variance: PortfolioMetrics
    equal_weight: PortfolioMetrics
    tangency: PortfolioMetrics
    rf_annual: float


def equal_weight_vector(tickers: Sequence[str]) -> pd.Series:
    """Pesos iguales 1/n sobre el universo seleccionado."""
    names = [str(t) for t in tickers]
    if len(names) < 1:
        raise PortfolioOptimizationError("Se requiere al menos un activo.")
    n = len(names)
    return pd.Series(np.full(n, 1.0 / n), index=names, dtype=float)


def portfolio_expected_return_annualized(weights: np.ndarray, mean_monthly: np.ndarray) -> float:
    """Rendimiento esperado anualizado: (1 + w'μ_mensual)^12 - 1."""
    monthly = float(np.dot(weights, mean_monthly))
    return float((1.0 + monthly) ** PERIODS_PER_YEAR - 1.0)


def portfolio_volatility_annualized(weights: np.ndarray, cov_monthly: np.ndarray) -> float:
    """Volatilidad anualizada: sqrt(w'Σw) × sqrt(12)."""
    var_m = float(np.dot(weights, np.dot(cov_monthly, weights)))
    if var_m < 0 and var_m > -1e-14:
        var_m = 0.0
    if var_m < 0:
        raise PortfolioOptimizationError("La varianza del portafolio resultó negativa.")
    return float(np.sqrt(var_m) * np.sqrt(PERIODS_PER_YEAR))


def portfolio_expected_return_linear_annual(
    weights: np.ndarray,
    mean_monthly: np.ndarray,
) -> float:
    """Rendimiento esperado anual lineal (Markowitz): w'μ_mensual × 12."""
    return float(np.dot(weights, mean_monthly) * PERIODS_PER_YEAR)


def evaluate_portfolio_linear(
    weights: pd.Series | Mapping[str, float],
    returns: pd.DataFrame,
    *,
    rf_annual: float,
) -> PortfolioMetrics:
    """Evalúa μ y σ anualizados linealmente (espacio estándar Markowitz / CML)."""
    tickers = list(returns.columns)
    w = _align_weights(weights, tickers)
    mean_m = returns.mean().to_numpy(dtype=float)
    cov_m = returns.cov().to_numpy(dtype=float)
    r_ann = portfolio_expected_return_linear_annual(w, mean_m)
    vol_ann = portfolio_volatility_annualized(w, cov_m)
    sharpe = portfolio_sharpe(r_ann, vol_ann, rf_annual)
    return PortfolioMetrics(
        expected_return=r_ann,
        volatility=vol_ann,
        sharpe=sharpe,
        weights=pd.Series(w, index=tickers, dtype=float),
    )


def maximize_sharpe_linear(
    returns: pd.DataFrame,
    *,
    rf_annual: float,
    forced_weights: Mapping[str, float] | None = None,
) -> PortfolioMetrics:
    """Maximiza Sharpe con retorno anual lineal (coherente con la frontera y la CML)."""
    if returns.shape[1] < 2:
        raise PortfolioOptimizationError("Se necesitan al menos 2 activos para optimizar.")
    tickers = list(returns.columns)
    mean_m = returns.mean().to_numpy(dtype=float)
    cov_m = returns.cov().to_numpy(dtype=float)
    forced = dict(forced_weights or {})
    _validate_forced_weights(forced, tickers)

    n = len(tickers)
    index = {t: i for i, t in enumerate(tickers)}

    def objective(w: np.ndarray) -> float:
        r_ann = portfolio_expected_return_linear_annual(w, mean_m)
        vol_ann = portfolio_volatility_annualized(w, cov_m)
        if vol_ann <= _SHARPE_EPS:
            return 1e6
        return -portfolio_sharpe(r_ann, vol_ann, rf_annual)

    constraints = _weight_constraints(tickers, forced)
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = _initial_weights(n, tickers, forced)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"No se pudo optimizar el portafolio (Sharpe lineal): {result.message}"
        )

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    return evaluate_portfolio_linear(pd.Series(weights, index=tickers), returns, rf_annual=rf_annual)


def portfolio_sharpe(
    expected_return_annual: float,
    volatility_annual: float,
    rf_annual: float,
) -> float:
    """Ratio de Sharpe anual: (R - rf) / σ."""
    if volatility_annual <= _SHARPE_EPS:
        raise PortfolioOptimizationError("La volatilidad es demasiado baja para calcular Sharpe.")
    return float((expected_return_annual - rf_annual) / volatility_annual)


def evaluate_portfolio(
    weights: pd.Series | Mapping[str, float],
    returns: pd.DataFrame,
    *,
    rf_annual: float,
) -> PortfolioMetrics:
    """Evalúa rendimiento, volatilidad y Sharpe anualizados de un vector de pesos."""
    tickers = list(returns.columns)
    w = _align_weights(weights, tickers)
    mean_m = returns.mean().to_numpy(dtype=float)
    cov_m = returns.cov().to_numpy(dtype=float)
    r_ann = portfolio_expected_return_annualized(w, mean_m)
    vol_ann = portfolio_volatility_annualized(w, cov_m)
    sharpe = portfolio_sharpe(r_ann, vol_ann, rf_annual)
    return PortfolioMetrics(
        expected_return=r_ann,
        volatility=vol_ann,
        sharpe=sharpe,
        weights=pd.Series(w, index=tickers, dtype=float),
    )


def maximize_sharpe(
    returns: pd.DataFrame,
    *,
    rf_annual: float,
    forced_weights: Mapping[str, float] | None = None,
) -> PortfolioMetrics:
    """
    Maximiza la razón de Sharpe anual (long-only, suma de pesos = 1).

    Los pesos forzados se imponen como restricciones de igualdad.
    """
    if returns.shape[1] < 2:
        raise PortfolioOptimizationError("Se necesitan al menos 2 activos para optimizar.")
    tickers = list(returns.columns)
    mean_m = returns.mean().to_numpy(dtype=float)
    cov_m = returns.cov().to_numpy(dtype=float)
    forced = dict(forced_weights or {})
    _validate_forced_weights(forced, tickers)

    n = len(tickers)
    index = {t: i for i, t in enumerate(tickers)}
    free_idx = [i for i, t in enumerate(tickers) if t not in forced]
    forced_sum = float(sum(forced.values()))
    if forced_sum >= 1.0 - _WEIGHT_TOL and free_idx:
        raise PortfolioOptimizationError(
            "Los pesos forzados suman 100% y no dejan margen para el resto de activos."
        )

    def objective(w: np.ndarray) -> float:
        r_ann = portfolio_expected_return_annualized(w, mean_m)
        vol_ann = portfolio_volatility_annualized(w, cov_m)
        if vol_ann <= _SHARPE_EPS:
            return 1e6
        return -portfolio_sharpe(r_ann, vol_ann, rf_annual)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for ticker, weight in forced.items():
        i = index[ticker]
        constraints.append({"type": "eq", "fun": lambda w, i=i, wt=weight: w[i] - wt})

    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = _initial_weights(n, tickers, forced)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"No se pudo optimizar el portafolio: {result.message}"
        )

    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    return evaluate_portfolio(pd.Series(weights, index=tickers), returns, rf_annual=rf_annual)


def compare_equal_vs_optimized(
    returns: pd.DataFrame,
    *,
    rf_annual: float,
    forced_weights: Mapping[str, float] | None = None,
) -> PortfolioComparison:
    """Compara portafolio de pesos iguales vs máxima Sharpe."""
    if returns.empty:
        raise PortfolioOptimizationError("No hay rendimientos para comparar portafolios.")
    equal = evaluate_portfolio(
        equal_weight_vector(returns.columns),
        returns,
        rf_annual=rf_annual,
    )
    optimized = maximize_sharpe(
        returns,
        rf_annual=rf_annual,
        forced_weights=forced_weights,
    )
    return PortfolioComparison(equal_weight=equal, optimized=optimized)


def compute_efficient_frontier(
    returns: pd.DataFrame,
    *,
    rf_annual: float,
    forced_weights: Mapping[str, float] | None = None,
    n_points: int = 120,
) -> EfficientFrontierResult:
    """
    Frontera Markowitz completa (hipérbola) en espacio (σ, μ) lineal anual.

    Optimización cuadrática: min w'Σw s.a. w'μ = objetivo mensual, long-only.
    Devuelve ambas ramas (ineficiente + eficiente) y el portafolio de mínima varianza.
    """
    if returns.shape[1] < 2:
        raise PortfolioOptimizationError("Se necesitan al menos 2 activos para la frontera.")
    if n_points < 2:
        raise PortfolioOptimizationError("Se requieren al menos 2 puntos para la frontera.")

    tickers = list(returns.columns)
    mean_m = returns.mean().to_numpy(dtype=float)
    cov_m = returns.cov().to_numpy(dtype=float)
    forced = dict(forced_weights or {})

    gmv_weights = _minimize_global_variance(mean_m, cov_m, tickers, forced)
    mu_min = _optimize_mean_return(mean_m, cov_m, tickers, forced, direction="min")
    mu_max = _optimize_mean_return(mean_m, cov_m, tickers, forced, direction="max")
    targets = np.linspace(mu_min, mu_max, n_points)

    vols: list[float] = []
    rets: list[float] = []
    w_prev = _initial_weights(len(tickers), tickers, forced)

    for target in targets:
        try:
            weights = _minimize_variance_for_target(
                mean_m,
                cov_m,
                tickers,
                forced,
                target_monthly=float(target),
                w0=w_prev,
            )
            w_prev = weights.copy()
            vols.append(portfolio_volatility_annualized(weights, cov_m))
            rets.append(portfolio_expected_return_linear_annual(weights, mean_m))
        except PortfolioOptimizationError:
            continue

    if len(vols) < 2:
        raise PortfolioOptimizationError(
            "No se pudo construir la frontera eficiente con los activos seleccionados."
        )

    vols_arr = np.asarray(vols, dtype=float)
    rets_arr = np.asarray(rets, dtype=float)
    vols_arr, rets_arr = _dedupe_consecutive_frontier_points(vols_arr, rets_arr)

    gmv_idx = int(np.argmin(vols_arr))
    min_variance = evaluate_portfolio_linear(
        pd.Series(gmv_weights, index=tickers), returns, rf_annual=rf_annual
    )

    tangency = maximize_sharpe_linear(
        returns, rf_annual=rf_annual, forced_weights=forced
    )
    equal = evaluate_portfolio_linear(
        equal_weight_vector(tickers), returns, rf_annual=rf_annual
    )

    return EfficientFrontierResult(
        volatilities=vols_arr,
        expected_returns=rets_arr,
        inefficient_volatilities=vols_arr[: gmv_idx + 1],
        inefficient_expected_returns=rets_arr[: gmv_idx + 1],
        efficient_volatilities=vols_arr[gmv_idx:],
        efficient_expected_returns=rets_arr[gmv_idx:],
        min_variance=min_variance,
        equal_weight=equal,
        tangency=tangency,
        rf_annual=rf_annual,
    )


def _minimize_global_variance(
    mean_m: np.ndarray,
    cov_m: np.ndarray,
    tickers: Sequence[str],
    forced: Mapping[str, float],
) -> np.ndarray:
    """Portafolio de mínima varianza global (long-only)."""
    n = len(tickers)

    def objective(w: np.ndarray) -> float:
        return float(0.5 * np.dot(w, np.dot(cov_m, w)))

    constraints = _weight_constraints(tickers, forced)
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = _initial_weights(n, tickers, forced)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"No se pudo calcular el portafolio de mínima varianza: {result.message}"
        )
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def _validate_forced_weights(forced: Mapping[str, float], tickers: Sequence[str]) -> None:
    forced_sum = float(sum(forced.values()))
    if forced_sum > 1.0 + _WEIGHT_TOL:
        raise PortfolioOptimizationError("La suma de pesos forzados supera 100%.")
    for ticker, weight in forced.items():
        if ticker not in tickers:
            raise PortfolioOptimizationError(
                f"El peso forzado de {ticker} no corresponde a un activo seleccionado."
            )
        if weight < 0 or weight > 1:
            raise PortfolioOptimizationError(f"El peso forzado de {ticker} debe estar en [0, 1].")


def _dedupe_consecutive_frontier_points(
    vols: np.ndarray,
    rets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Elimina duplicados consecutivos sin reordenar (preserva la hipérbola)."""
    clean_vols = [float(vols[0])]
    clean_rets = [float(rets[0])]
    for vol, ret in zip(vols[1:], rets[1:]):
        if abs(vol - clean_vols[-1]) < 1e-10 and abs(ret - clean_rets[-1]) < 1e-10:
            continue
        clean_vols.append(float(vol))
        clean_rets.append(float(ret))
    return np.asarray(clean_vols, dtype=float), np.asarray(clean_rets, dtype=float)


def _minimize_variance_for_target(
    mean_m: np.ndarray,
    cov_m: np.ndarray,
    tickers: Sequence[str],
    forced: Mapping[str, float],
    *,
    target_monthly: float,
    w0: np.ndarray | None = None,
) -> np.ndarray:
    """Minimiza w'Σw sujeto a w'μ = objetivo mensual (Markowitz, long-only)."""
    n = len(tickers)

    def objective(w: np.ndarray) -> float:
        return float(0.5 * np.dot(w, np.dot(cov_m, w)))

    constraints = _weight_constraints(tickers, forced)
    constraints.append(
        {"type": "eq", "fun": lambda w, t=target_monthly: np.dot(w, mean_m) - t}
    )
    bounds = [(0.0, 1.0) for _ in range(n)]
    start = w0 if w0 is not None else _initial_weights(n, tickers, forced)

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"Optimización cuadrática fallida: {result.message}"
        )
    weights = np.clip(result.x, 0.0, 1.0)
    return weights / weights.sum()


def _optimize_mean_return(
    mean_m: np.ndarray,
    cov_m: np.ndarray,
    tickers: Sequence[str],
    forced: Mapping[str, float],
    *,
    direction: str,
) -> float:
    """Retorno mensual mínimo o máximo alcanzable con las restricciones."""
    n = len(tickers)
    sign = -1.0 if direction == "max" else 1.0

    def objective(w: np.ndarray) -> float:
        return float(sign * np.dot(w, mean_m))

    constraints = _weight_constraints(tickers, forced)
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = _initial_weights(n, tickers, forced)

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise PortfolioOptimizationError(
            f"No se pudo calcular el retorno {direction}: {result.message}"
        )
    return float(np.dot(result.x, mean_m))


def _weight_constraints(
    tickers: Sequence[str],
    forced: Mapping[str, float],
) -> list[dict]:
    """Restricciones de suma unitaria y pesos forzados."""
    index = {t: i for i, t in enumerate(tickers)}
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    for ticker, weight in forced.items():
        if ticker not in index:
            raise PortfolioOptimizationError(
                f"El peso forzado de {ticker} no corresponde a un activo seleccionado."
            )
        i = index[ticker]
        constraints.append({"type": "eq", "fun": lambda w, i=i, wt=weight: w[i] - wt})
    return constraints


def _align_weights(weights: pd.Series | Mapping[str, float], tickers: Sequence[str]) -> np.ndarray:
    series = pd.Series(weights, dtype=float)
    missing = [t for t in tickers if t not in series.index]
    if missing:
        raise PortfolioOptimizationError(
            "Faltan pesos para: " + ", ".join(missing)
        )
    w = series.reindex(tickers).to_numpy(dtype=float)
    if not np.all(np.isfinite(w)):
        raise PortfolioOptimizationError("Hay pesos no finitos.")
    total = float(w.sum())
    if abs(total - 1.0) > 1e-4:
        raise PortfolioOptimizationError(
            f"Los pesos deben sumar 1 (suma actual: {total:.4f})."
        )
    if np.any(w < -_WEIGHT_TOL):
        raise PortfolioOptimizationError("No se permiten pesos negativos.")
    return w


def _initial_weights(
    n: int,
    tickers: Sequence[str],
    forced: Mapping[str, float],
) -> np.ndarray:
    w0 = np.zeros(n, dtype=float)
    free = [i for i, t in enumerate(tickers) if t not in forced]
    remaining = 1.0 - float(sum(forced.values()))
    for ticker, weight in forced.items():
        w0[list(tickers).index(ticker)] = weight
    if free:
        share = remaining / len(free)
        for i in free:
            w0[i] = share
    return w0
