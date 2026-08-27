"""Carga y preparación de datos (Funcionalidad 0)."""

import streamlit as st

from utils.state import store_prepared_data
from src.data.loader import (
    DEFAULT_CSV_PATH,
    DataPreparationError,
    PreparedData,
    load_portfolio_data,
    load_uploaded_csv,
)

MAX_UPLOAD_MB = 10

prepared: PreparedData | None = st.session_state.get("prepared_data")
source_name = st.session_state.get("data_source_name", DEFAULT_CSV_PATH.name)
source_kind = st.session_state.get("data_source_kind", "default")

st.caption(
    "Valida el CSV del proyecto o carga uno nuevo. Solo continúan los activos con datos completos."
)

with st.container(border=True):
    st.subheader("Origen del dataset")
    if prepared is None:
        st.warning("Todavía no hay un dataset validado.")
    elif source_kind == "default":
        st.write(f"Dataset activo: **{source_name}** (archivo por defecto).")
    else:
        st.write(f"Dataset activo: **{source_name}** (CSV cargado).")

    if st.button(
        "Validar dataset por defecto",
        type="primary",
        icon=":material/verified:",
        help=f"Lee y valida {DEFAULT_CSV_PATH.name}.",
    ):
        try:
            default_data = load_portfolio_data(DEFAULT_CSV_PATH)
        except DataPreparationError as exc:
            st.error(str(exc))
        else:
            store_prepared_data(
                default_data,
                source_name=DEFAULT_CSV_PATH.name,
                source_kind="default",
            )
            st.rerun()

    uploaded = st.file_uploader(
        "Agregar un nuevo dataset (.csv)",
        type=["csv"],
        accept_multiple_files=False,
        max_upload_size=MAX_UPLOAD_MB,
        help=(
            "El archivo debe tener una columna Date, precios de empresas y, "
            "si existe, el S&P 500 (^GSPC). Máximo "
            f"{MAX_UPLOAD_MB} MB."
        ),
        key="dataset_upload",
    )
    load_clicked = st.button(
        "Cargar y validar CSV",
        icon=":material/upload_file:",
        disabled=uploaded is None,
    )
    if load_clicked:
        if uploaded is None:
            st.error("Selecciona un archivo .csv antes de cargarlo.")
        else:
            try:
                uploaded_data = load_uploaded_csv(uploaded.name, uploaded.getvalue())
            except DataPreparationError as exc:
                st.error(str(exc))
            else:
                store_prepared_data(
                    uploaded_data,
                    source_name=uploaded.name,
                    source_kind="upload",
                )
                st.rerun()

prepared = st.session_state.get("prepared_data")
if prepared is None:
    st.info("Valida el dataset por defecto o carga un CSV para ver el universo de inversión.")
    st.stop()

for message in prepared.messages:
    st.warning(message)

if not prepared.messages:
    st.success("Todos los activos tienen datos completos. No se omitió ninguna empresa.")

with st.container(horizontal=True):
    st.metric("Observaciones", prepared.n_periods, border=True)
    st.metric("Activos válidos", len(prepared.valid_tickers), border=True)
    st.metric("Omitidos", len(prepared.omitted_tickers), border=True)
    st.metric("Benchmark", prepared.benchmark_ticker or "No disponible", border=True)

with st.container(border=True):
    st.subheader("Universo válido")
    st.write(
        "Estos activos tienen historial completo y estarán disponibles en "
        "Inputs y configuración inicial. El S&P 500 se reserva como benchmark "
        "y no se incluye como activo a optimizar."
    )
    st.write(", ".join(prepared.valid_tickers))

with st.expander("Vista previa de precios"):
    preview = prepared.prices.copy()
    if prepared.benchmark is not None:
        preview[str(prepared.benchmark.name)] = prepared.benchmark
    st.dataframe(preview.head(12), width="stretch")
