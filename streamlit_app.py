"""Portafolio de inversión con optimización de Markowitz."""

from __future__ import annotations

import streamlit as st

from src.data.loader import (
    DEFAULT_CSV_PATH,
    DataPreparationError,
    PreparedData,
    load_portfolio_data,
)

st.set_page_config(
    page_title="Portafolio de inversión",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_prepared_data(csv_path: str) -> PreparedData:
    return load_portfolio_data(csv_path)


st.session_state.setdefault("scenario", None)
st.session_state.setdefault("data_source_name", DEFAULT_CSV_PATH.name)
st.session_state.setdefault("data_source_kind", "default")

if "prepared_data" not in st.session_state:
    try:
        st.session_state.prepared_data = get_prepared_data(str(DEFAULT_CSV_PATH))
        st.session_state.valid_tickers = st.session_state.prepared_data.valid_tickers
    except DataPreparationError as exc:
        st.session_state.prepared_data = None
        st.error(str(exc))

prepared = st.session_state.get("prepared_data")

scenario = st.session_state.get("scenario")
if prepared is not None and scenario is not None:
    missing = [ticker for ticker in scenario.selected_tickers if ticker not in prepared.valid_tickers]
    if missing:
        st.session_state.scenario = None
        scenario = None

PAGES = [
    st.Page(
        "app_pages/home.py",
        title="Inicio",
        icon=":material/home:",
        default=True,
    ),
    st.Page(
        "app_pages/data.py",
        title="Carga y preparación de datos",
        icon=":material/database:",
    ),
    st.Page(
        "app_pages/inputs.py",
        title="Inputs y configuración inicial",
        icon=":material/tune:",
    ),
    st.Page(
        "app_pages/optimize.py",
        title="Optimización y frontera eficiente",
        icon=":material/show_chart:",
    ),
    st.Page(
        "app_pages/results.py",
        title="Resultados finales y validación histórica",
        icon=":material/fact_check:",
    ),
]
page_by_title = {p.title: p for p in PAGES}
page_titles = list(page_by_title.keys())
st.session_state["_pages_by_title"] = page_by_title

page = st.navigation(PAGES, position="hidden")

with st.sidebar:
    st.markdown("**Menú principal**")
    # Key ligada a la página activa: al navegar desde Inicio el selectbox
    # se remonta y no fuerza el regreso a la sección anterior.
    selected_title = st.selectbox(
        "Sección",
        options=page_titles,
        index=page_titles.index(page.title),
        key=f"sidebar_section_{page.title}",
        label_visibility="collapsed",
    )
    if selected_title != page.title:
        st.switch_page(page_by_title[selected_title])

    st.divider()
    if prepared is None:
        st.caption("Dataset pendiente de validar")
    elif scenario is None:
        source_name = st.session_state.get("data_source_name", DEFAULT_CSV_PATH.name)
        st.caption(f"Dataset: {source_name}")
        st.caption("Universo pendiente de confirmar")
    else:
        st.markdown("**Universo confirmado**")
        st.write(f"{len(scenario.selected_tickers)} activos seleccionados")
        st.write(f"rf anual {scenario.rf_annual:.2%}")
        if scenario.forced_weights:
            st.write(f"{len(scenario.forced_weights)} peso(s) forzado(s)")

st.title(page.title)
page.run()
