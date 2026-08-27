"""Página de inicio: descripción general de la aplicación."""

from __future__ import annotations

import streamlit as st

st.caption(
    "Análisis y optimización de portafolios con la teoría de Markowitz, "
    "implementada en Python puro (NumPy + SciPy)."
)

st.info(
    "💡 Bienvenido al **Portafolio de Inversión con Markowitz**. "
    "Esta app te guía paso a paso: desde la carga de datos hasta la "
    "validación histórica de tu portafolio optimizado."
)

with st.container(border=True):
    st.subheader("📊 ¿Qué hace esta aplicación?")
    st.markdown(
        """
        Herramienta interactiva para estudiantes y analistas que quieren **entender**
        cómo se construye un portafolio eficiente a partir de precios históricos de
        **20 empresas** y del índice **S&P 500**.

        A diferencia de enfoques basados en librerías como PyPortfolioOpt, aquí los
        cálculos de riesgo, rendimiento y optimización están implementados
        directamente, para ver la lógica detrás de cada resultado.
        """
    )

with st.container(border=True):
    st.subheader("🧭 Flujo recomendado")
    st.markdown(
        """
        1. 📁 **Carga y preparación de datos** — validar el CSV y filtrar activos incompletos.
        2. ⚙️ **Inputs y configuración inicial** — elegir activos, pesos forzados y tasa libre de riesgo.
        3. 📈 **Optimización y frontera eficiente** — correlaciones, comparación y gráfico Markowitz + CML.
        4. ✅ **Resultados finales y validación histórica** — pesos óptimos y evolución base $100 vs S&P 500.
        """
    )

with st.container(border=True):
    st.subheader("📌 En un vistazo")
    col_s, col_a, col_b, col_r = st.columns(4)
    col_s.metric("Secciones", "4")
    col_a.metric("Activos", "~20")
    col_b.metric("Benchmark", "S&P 500")
    col_r.metric("rf default", "4%")

with st.container(border=True):
    st.subheader("Secciones del menú")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Carga y preparación de datos")
        st.markdown(
            "Carga automática o manual del CSV. Excluye activos con datos faltantes "
            "e informa qué empresas se omitieron. El S&P 500 se usa como benchmark, "
            "no como activo optimizable."
        )
        st.page_link(
            "app_pages/data.py",
            label="Ir a Carga de datos",
            icon=":material/database:",
        )

        st.markdown("#### Inputs y configuración inicial")
        st.markdown(
            "Selecciona el universo de inversión, define **pesos forzados** "
            "(por ejemplo, Visa = 15 %) y la **tasa libre de riesgo** "
            "(por defecto ~4 % anual). Confirma el escenario para continuar."
        )
        st.page_link(
            "app_pages/inputs.py",
            label="Ir a Inputs",
            icon=":material/tune:",
        )

    with col2:
        st.markdown("#### Optimización y frontera eficiente")
        st.markdown(
            "Mapa de calor de correlaciones, tabla comparativa "
            "(pesos iguales vs optimizado) con rendimiento, volatilidad y Sharpe anualizados, "
            "y gráfico de la **frontera eficiente** con la **línea del mercado de capitales (CML)**."
        )
        st.page_link(
            "app_pages/optimize.py",
            label="Ir a Optimización",
            icon=":material/show_chart:",
        )

        st.markdown("#### Resultados finales y validación histórica")
        st.markdown(
            "Tabla horizontal con los **pesos del portafolio optimizado** (máx. Sharpe) "
            "y gráfico de evolución histórica base **$100** para pesos iguales, "
            "portafolio optimizado y S&P 500."
        )
        st.page_link(
            "app_pages/results.py",
            label="Ir a Resultados",
            icon=":material/fact_check:",
        )

with st.container(border=True):
    st.markdown("### 📝 Decisiones de diseño")
    design_left, design_right = st.columns(2)
    with design_left:
        st.markdown(
            "🎨 **Tema oscuro** inspirado en terminales financieras (Bloomberg).\n\n"
            "📐 **Retornos mensuales simples**; métricas anualizadas en tablas y gráficos."
        )
    with design_right:
        st.markdown(
            "🔒 **Long-only**: pesos entre 0 % y 100 %, suma igual a 100 %.\n\n"
            "⚙️ **Optimización** con SciPy (SLSQP): frontera eficiente y máximo Sharpe."
        )
