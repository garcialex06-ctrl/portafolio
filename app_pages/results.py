"""
Resultados finales y validación histórica (Funcionalidad 4).

Tabla de pesos del portafolio optimizado y evolución base 100 vs S&P 500.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.core.backtest import (
    final_wealth_summary,
    historical_wealth_paths,
    optimized_weights_display_table,
)
from src.core.metrics import MetricsError, calcular_rendimientos
from src.core.portfolio import PortfolioOptimizationError, maximize_sharpe
from src.config.tickers import ticker_label
from utils.charts import historical_wealth_figure
from utils.state import require_prepared_data, require_scenario

scenario = require_scenario()
prepared = require_prepared_data()

st.caption(
    f"{len(scenario.selected_tickers)} activos confirmados · "
    f"pesos forzados: {len(scenario.forced_weights)} · "
    f"rf anual {scenario.rf_annual:.2%}."
)

tickers = list(scenario.selected_tickers)
universe = list(prepared.valid_tickers)
missing = [ticker for ticker in tickers if ticker not in prepared.prices.columns]
if missing:
    st.error(
        "Algunos activos del escenario ya no están en el dataset: "
        + ", ".join(missing)
        + ". Vuelve a confirmar el universo en Inputs."
    )
    st.stop()

if prepared.benchmark is None:
    st.error(
        "No hay índice S&P 500 en el dataset. No se puede completar la "
        "validación histórica contra el benchmark."
    )
    st.stop()

try:
    returns = calcular_rendimientos(prepared.prices[tickers])
    optimized = maximize_sharpe(
        returns,
        rf_annual=scenario.rf_annual,
        forced_weights=scenario.forced_weights,
    )
except (PortfolioOptimizationError, MetricsError) as exc:
    st.error(str(exc))
    st.stop()

with st.container(border=True):
    weights_title = (
        "📊 Pesos del Portafolio Optimizado con Restricción de Pesos"
        if scenario.forced_weights
        else "📊 Pesos del Portafolio Optimizado"
    )
    st.subheader(weights_title)
    st.caption(
        "Solo activos con peso mayor a 0 % en la solución óptima (máx. Sharpe). "
        "El resto del universo seleccionado queda fuera de la asignación."
    )
    display_df = optimized_weights_display_table(
        optimized_weights=optimized.weights,
        forced_weights=scenario.forced_weights,
    )
    _, table_col, _ = st.columns([1, 2, 1])
    with table_col:
        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Activo": st.column_config.TextColumn("Activo"),
                "Porcentaje": st.column_config.TextColumn("Porcentaje"),
            },
        )
    if scenario.forced_weights:
        st.caption("🔴 = peso forzado definido en Inputs.")

    st.divider()
    st.markdown("**Activos excluidos en Inputs**")
    st.caption(
        "Empresas desmarcadas al confirmar el universo. "
        "No participan en la optimización ni en el backtest histórico."
    )
    selected_set = set(tickers)
    deselected = [ticker for ticker in universe if ticker not in selected_set]
    if deselected:
        st.dataframe(
            pd.DataFrame({"Activo": [ticker_label(ticker) for ticker in deselected]}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No hay activos excluidos: se incluyeron todos los disponibles en Inputs.")

with st.container(border=True):
    st.subheader("Evolución histórica (base $100)")
    st.caption(
        "Buy-and-hold desde el inicio de la muestra (~5 años). "
        "Pesos iguales y optimizado solo sobre activos seleccionados; "
        "benchmark = S&P 500."
    )
    try:
        paths = historical_wealth_paths(
            prepared.prices,
            selected_tickers=tickers,
            optimized_weights=optimized.weights,
            benchmark=prepared.benchmark,
        )
        st.plotly_chart(historical_wealth_figure(paths), width="stretch")

        finals = final_wealth_summary(paths)
        initial = 100.0

        st.subheader("💰 Valores Finales")
        st.info(
            "💡 Este gráfico responde a la pregunta: "
            f"'Si hubieras invertido $100 hace ~5 años "
            f"({paths.start.date()} → {paths.end.date()}), "
            "¿cuánto tendrías hoy en cada opción "
            "(pesos iguales, portafolio optimizado y S&P 500 como benchmark)?'"
        )

        col_eq, col_opt, col_bench = st.columns(3)
        col_eq.metric(
            "Pesos Iguales",
            f"${finals['Pesos iguales']:.2f}",
            delta=f"+${finals['Pesos iguales'] - initial:.2f}",
        )
        col_opt.metric(
            "Portafolio Optimizado",
            f"${finals['Optimizado']:.2f}",
            delta=f"+${finals['Optimizado'] - initial:.2f}",
        )
        col_bench.metric(
            "S&P 500",
            f"${finals['S&P 500']:.2f}",
            delta=f"+${finals['S&P 500'] - initial:.2f}",
        )

        st.markdown("### 📝 Nota Explicativa")
        note_left, note_right = st.columns(2)
        equal_pct = 100.0 / len(tickers)
        with note_left:
            st.markdown(
                f"🟢 **Portafolio de Pesos Iguales:**\n\n"
                f"Se asignó el mismo peso ({equal_pct:.1f}%) a cada una de las "
                f"**{len(tickers)}** empresas seleccionadas:\n\n"
                f"{', '.join(tickers)}"
            )
        with note_right:
            opt_weights = optimized.weights.sort_values(ascending=False)
            weights_txt = ", ".join(
                f"**{t}:** {w:.1%}"
                for t, w in opt_weights.items()
                if float(w) > 1e-4
            )
            st.markdown(
                f"🎯 **Portafolio Optimizado:**\n\n{weights_txt}"
            )
    except PortfolioOptimizationError as exc:
        st.error(str(exc))
