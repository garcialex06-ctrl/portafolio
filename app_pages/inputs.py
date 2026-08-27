"""Inputs y configuración inicial (Funcionalidad 1)."""

import streamlit as st

from utils.state import require_prepared_data
from src.config.scenario import build_scenario
from src.config.tickers import ticker_label

prepared = require_prepared_data()
available = list(prepared.valid_tickers)
scenario = st.session_state.get("scenario")

st.caption("Definir el universo de inversión y los supuestos del modelo a partir de los activos válidos.")

with st.container(border=True):
    st.subheader("Activos disponibles")
    st.caption(
        "Todas las empresas con datos completos entran por defecto. "
        "Desmarca las que no quieras incluir en el portafolio."
    )
    columns = st.columns(4)
    for index, ticker in enumerate(available):
        with columns[index % 4]:
            st.checkbox(
                ticker,
                value=True,
                key=f"include_{ticker}",
                persist_state="session",
                help=ticker_label(ticker),
            )

selected = [ticker for ticker in available if st.session_state.get(f"include_{ticker}")]

with st.container(border=True):
    st.subheader("Pesos forzados")
    st.caption(
        "Opcional. Ejemplo: Visa (`V`) = 15%. El porcentaje restante se optimizará "
        "en las siguientes secciones. La suma no puede superar 100%."
    )
    forced_tickers = st.multiselect(
        "Activos con peso fijo",
        options=selected,
        format_func=ticker_label,
        key="forced_tickers",
        persist_state="session",
        placeholder="Elige uno o varios activos",
        help="Solo puedes forzar pesos de activos que estén incluidos arriba.",
    )
    forced_tickers = [ticker for ticker in forced_tickers if ticker in selected]

    forced_weights_percent: dict[str, float | None] = {}
    if forced_tickers:
        weight_cols = st.columns(min(3, len(forced_tickers)))
        for index, ticker in enumerate(forced_tickers):
            with weight_cols[index % len(weight_cols)]:
                forced_weights_percent[ticker] = st.number_input(
                    f"Peso de {ticker_label(ticker)} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=None,
                    placeholder="15.0",
                    step=0.5,
                    format="%.2f",
                    key=f"forced_w_{ticker}",
                    persist_state="session",
                )
    forced_sum = sum(weight for weight in forced_weights_percent.values() if weight is not None)
    st.metric("Suma de pesos forzados", f"{forced_sum:.2f}%")
    if forced_sum > 100:
        st.error("La suma de los pesos forzados supera 100%. Ajusta los valores antes de confirmar.")

with st.container(border=True):
    st.subheader("Tasa libre de riesgo")
    st.caption(
        "Por defecto se usa 4% anual (aprox. Treasury a 10 años). "
        "Si dejas el campo vacío, se aplica ese 4%. Puedes escribir otra tasa anual; "
        "el modelo la convierte a mensual como $r_{mensual} = r_{anual}/12$."
    )
    rf_annual_percent = st.number_input(
        "Tasa libre de riesgo anual (%)",
        value=None,
        placeholder="4.0",
        step=0.25,
        format="%.2f",
        key="rf_annual_percent",
        persist_state="session",
        help="Vacío = 4% anual. No se aceptan tasas negativas extremas.",
        icon=":material/percent:",
    )
    rf_for_preview = 4.0 if rf_annual_percent is None else rf_annual_percent
    st.write(f"Equivalente mensual de referencia: **{rf_for_preview / 12:.4f}%**")
    if rf_annual_percent is None:
        st.caption("No ingresaste una tasa: al confirmar se usará 4% anual.")

if len(selected) < 2:
    st.warning("Selecciona al menos 2 activos para poder diversificar.")

confirmed = st.button(
    "Confirmar universo y supuestos",
    type="primary",
    icon=":material/lock:",
)

if confirmed:
    validation = build_scenario(
        available_tickers=available,
        selected_tickers=selected,
        forced_weights_percent=forced_weights_percent,
        rf_annual_percent=rf_annual_percent,
    )
    if validation.ok and validation.scenario is not None:
        st.session_state.scenario = validation.scenario
        st.rerun()
    else:
        for error in validation.errors:
            st.error(error)

scenario = st.session_state.get("scenario")
if scenario is not None:
    confirmed_forced = {ticker: round(weight * 100.0, 4) for ticker, weight in scenario.forced_weights.items()}
    draft_forced = {
        ticker: round(weight, 4)
        for ticker, weight in forced_weights_percent.items()
        if weight is not None
    }
    rf_changed = (
        (rf_annual_percent is None) != scenario.used_default_rf
        or (
            rf_annual_percent is not None
            and abs(rf_annual_percent - scenario.rf_annual * 100.0) > 1e-6
        )
    )
    if tuple(selected) != scenario.selected_tickers or draft_forced != confirmed_forced or rf_changed:
        st.warning(
            "Hay cambios sin confirmar. Las demás secciones siguen usando el escenario anterior."
        )
    forced_txt = (
        ", ".join(f"{ticker} {weight:.1%}" for ticker, weight in scenario.forced_weights.items())
        or "ninguno"
    )
    rf_origin = "valor por defecto" if scenario.used_default_rf else "definida por el usuario"
    with st.container(border=True):
        st.subheader("Escenario confirmado")
        st.write(
            f"**Activos:** {', '.join(scenario.selected_tickers)}  \n"
            f"**Pesos forzados:** {forced_txt}  \n"
            f"**Tasa libre de riesgo:** {scenario.rf_annual:.2%} anual "
            f"({scenario.rf_monthly:.4%} mensual, {rf_origin})"
        )
