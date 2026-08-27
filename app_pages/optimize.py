"""
Optimización y frontera eficiente.

Muestra la matriz de correlaciones y la comparación pesos iguales vs
portafolio de máxima Sharpe (Python puro: NumPy + SciPy).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.core.metrics import (
    MetricsError,
    calcular_matriz_correlaciones,
    calcular_rendimientos,
)
from src.core.portfolio import (
    PortfolioOptimizationError,
    compare_equal_vs_optimized,
    compute_efficient_frontier,
)
from utils.charts import correlation_heatmap_figure, efficient_frontier_figure
from utils.state import require_prepared_data, require_scenario


def _comparison_table(comparison) -> pd.DataFrame:
    """Tabla anualizada: pesos iguales vs portafolio de máxima Sharpe."""
    return pd.DataFrame(
        {
            "Métrica": [
                "Rendimiento esperado (anual)",
                "Volatilidad / riesgo (anual)",
                "Ratio de Sharpe",
            ],
            "Pesos iguales": [
                f"{comparison.equal_weight.expected_return:.2%}",
                f"{comparison.equal_weight.volatility:.2%}",
                f"{comparison.equal_weight.sharpe:.3f}",
            ],
            "Optimizado (máx. Sharpe)": [
                f"{comparison.optimized.expected_return:.2%}",
                f"{comparison.optimized.volatility:.2%}",
                f"{comparison.optimized.sharpe:.3f}",
            ],
        }
    )

scenario = require_scenario()
prepared = require_prepared_data()

st.caption(
    f"{len(scenario.selected_tickers)} activos confirmados · "
    f"rf mensual {scenario.rf_monthly:.4%}."
)

tickers = list(scenario.selected_tickers)
missing = [ticker for ticker in tickers if ticker not in prepared.prices.columns]
if missing:
    st.error(
        "Algunos activos del escenario ya no están en el dataset: "
        + ", ".join(missing)
        + ". Vuelve a confirmar el universo en Inputs."
    )
    st.stop()

try:
    returns = calcular_rendimientos(prepared.prices[tickers])
except MetricsError as exc:
    st.error(str(exc))
    st.stop()

with st.container(border=True):
    st.caption(
        "Correlación de Pearson entre rendimientos mensuales. "
        "Rojo = negativa · Azul = positiva."
    )
    try:
        corr = calcular_matriz_correlaciones(returns)
        fig = correlation_heatmap_figure(corr)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except MetricsError as exc:
        st.error(str(exc))

comparison = None

with st.container(border=True):
    st.subheader("Comparación de Portafolios: Pesos Iguales vs Optimizado")
    st.caption(
        "Rendimiento anualizado con capitalización compuesta "
        "$(1 + \\bar{r}_m)^{12} - 1$. Volatilidad $\\sigma_m \\times \\sqrt{12}$. "
        "Optimización de máxima Sharpe en NumPy + SciPy (sin PyPortfolioOpt)."
    )
    try:
        comparison = compare_equal_vs_optimized(
            returns,
            rf_annual=scenario.rf_annual,
            forced_weights=scenario.forced_weights,
        )
        st.dataframe(
            _comparison_table(comparison),
            hide_index=True,
            width="stretch",
        )
        st.session_state.portfolio_comparison = comparison
    except (PortfolioOptimizationError, MetricsError) as exc:
        st.error(str(exc))
        comparison = None

with st.container(border=True):
    st.subheader("Frontera eficiente y línea del mercado de capitales")
    st.caption(
        "Espacio Markowitz estándar: rendimiento anual **lineal** ($\\bar{r}_m \\times 12$), "
        f"volatilidad $\\sigma_m \\sqrt{{12}}$. CML con rf = {scenario.rf_annual:.2%}."
    )
    if comparison is None:
        st.info("Confirma el universo y revisa la tabla comparativa de arriba.")
    else:
        try:
            frontier = compute_efficient_frontier(
                returns,
                rf_annual=scenario.rf_annual,
                forced_weights=scenario.forced_weights,
            )
            st.plotly_chart(
                efficient_frontier_figure(frontier),
                width="stretch",
            )
        except (PortfolioOptimizationError, MetricsError) as exc:
            st.error(str(exc))
