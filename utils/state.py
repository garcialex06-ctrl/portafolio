"""Estado compartido de la app Streamlit (dataset y escenario)."""

from __future__ import annotations

import streamlit as st

from src.config.scenario import PortfolioScenario
from src.data.loader import PreparedData

_ASSET_WIDGET_PREFIXES = ("include_", "forced_w_")


def require_prepared_data() -> PreparedData:
    prepared = st.session_state.get("prepared_data")
    if prepared is None:
        st.info(
            "Valida el dataset por defecto o carga un CSV en "
            "**Carga y preparación de datos** para continuar."
        )
        st.stop()
    return prepared


def require_scenario() -> PortfolioScenario:
    require_prepared_data()
    scenario = st.session_state.get("scenario")
    if scenario is None:
        st.info(
            "Confirma el universo de activos y los supuestos en "
            "**Inputs y configuración inicial** para continuar."
        )
        st.stop()
    return scenario


def store_prepared_data(
    prepared: PreparedData,
    *,
    source_name: str,
    source_kind: str,
) -> None:
    st.session_state.prepared_data = prepared
    st.session_state.valid_tickers = prepared.valid_tickers
    st.session_state.data_source_name = source_name
    st.session_state.data_source_kind = source_kind
    st.session_state.scenario = None
    _clear_asset_widgets()


def _clear_asset_widgets() -> None:
    for key in list(st.session_state.keys()):
        key_name = str(key)
        if key_name.startswith(_ASSET_WIDGET_PREFIXES):
            del st.session_state[key]
    st.session_state.pop("forced_tickers", None)
