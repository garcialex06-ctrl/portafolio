"""
Visualizaciones del portafolio (correlaciones, frontera y validación histórica).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# EfficientFrontierResult / HistoricalWealthPaths: anotaciones diferidas.


def correlation_heatmap_figure(corr: pd.DataFrame) -> go.Figure:
    """Mapa de calor: rojo negativo, azul positivo; diagonal 1.00 de abajo-izq a arriba-der."""
    labels = [str(col) for col in corr.columns]
    values = corr.to_numpy(dtype=float)
    text = [[f"{v:.3f}" for v in row] for row in values]

    n = len(labels)
    tick_size = 11 if n <= 12 else 9
    text_size = 9 if n <= 14 else 7
    # Rectangular: altura contenida para que el ancho del layout domine.
    height = max(360, min(520, 180 + n * 16))

    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            textfont={"size": text_size, "color": "#f5f5f5"},
            colorscale=[
                [0.0, "#ff4d4f"],
                [0.5, "#0d0d0d"],
                [1.0, "#3d8bfd"],
            ],
            zmid=0.0,
            zmin=-1.0,
            zmax=1.0,
            colorbar={
                "title": {
                    "text": "Correlación",
                    "side": "right",
                    "font": {"color": "#f5f5f5", "size": 12},
                },
                "tickfont": {"color": "#f5f5f5", "size": 11},
                "thickness": 14,
                "len": 0.85,
                "outlinewidth": 0,
            },
            hovertemplate="%{y} × %{x}<br>correlación=%{z:.3f}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    fig.update_layout(
        title={
            "text": "Matriz de correlaciones entre activos",
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": "#f5f5f5", "size": 16},
        },
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"color": "#f5f5f5", "size": 12},
        margin={"l": 80, "r": 90, "t": 56, "b": 88},
        height=height,
        autosize=True,
        xaxis={
            "title": {"text": "Activos", "font": {"color": "#f5f5f5", "size": 12}},
            "side": "bottom",
            "tickangle": -45,
            "tickfont": {"size": tick_size, "color": "#f5f5f5"},
            "showgrid": False,
            "constrain": "domain",
        },
        yaxis={
            # go.Heatmap coloca la primera fila arriba; invertir el eje deja
            # el primer activo abajo → diagonal 1.00 de abajo-izq a arriba-der.
            "autorange": "reversed",
            "title": {"text": "Activos", "font": {"color": "#f5f5f5", "size": 12}},
            "tickfont": {"size": tick_size, "color": "#f5f5f5"},
            "showgrid": False,
            "constrain": "domain",
        },
    )
    return fig


def efficient_frontier_figure(result) -> go.Figure:
    """
    Frontera Markowitz (hipérbola completa) y CML en espacio (σ, μ) lineal anual.
    """
    eq = result.equal_weight
    tg = result.tangency
    gmv = result.min_variance
    rf = result.rf_annual

    fig = go.Figure()

    if len(result.inefficient_volatilities) >= 2:
        fig.add_trace(
            go.Scatter(
                x=result.inefficient_volatilities,
                y=result.inefficient_expected_returns,
                mode="lines",
                name="Rama ineficiente",
                line={"color": "#555555", "width": 1.5, "dash": "dot"},
                hovertemplate="Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=result.efficient_volatilities,
            y=result.efficient_expected_returns,
            mode="lines",
            name="Frontera eficiente",
            line={"color": "#3d8bfd", "width": 2.5},
            hovertemplate="Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}<extra></extra>",
        )
    )

    x_lo, x_hi, y_lo, y_hi = _frontier_axis_bounds(result, rf)
    sharpe_t = tg.sharpe
    cml_x_end = min(x_hi, tg.volatility * 1.35)
    cml_vol = np.linspace(0.0, cml_x_end, 100)
    cml_ret = rf + sharpe_t * cml_vol
    fig.add_trace(
        go.Scatter(
            x=cml_vol,
            y=cml_ret,
            mode="lines",
            name="CML",
            line={"color": "#00c853", "width": 2, "dash": "dash"},
            hovertemplate=(
                "CML · Línea del Mercado de Capitales<br>"
                "Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[gmv.volatility],
            y=[gmv.expected_return],
            mode="markers",
            name="Mín. varianza",
            marker={"color": "#00e5ff", "size": 12, "symbol": "square"},
            hovertemplate=(
                "Mínima varianza<br>Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[eq.volatility],
            y=[eq.expected_return],
            mode="markers",
            name="Pesos iguales",
            marker={"color": "#ffab00", "size": 14, "symbol": "circle"},
            hovertemplate=(
                "Pesos iguales<br>Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[tg.volatility],
            y=[tg.expected_return],
            mode="markers",
            name="Máx. Sharpe",
            marker={"color": "#b388ff", "size": 14, "symbol": "diamond"},
            hovertemplate=(
                "Máxima Sharpe<br>Riesgo=%{x:.2%}<br>Rendimiento=%{y:.2%}"
                "<br>Sharpe=%{customdata:.3f}<extra></extra>"
            ),
            customdata=[tg.sharpe],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[rf],
            mode="markers",
            name=f"rf ({rf:.2%})",
            marker={"color": "#ffffff", "size": 10, "symbol": "circle-open"},
            hovertemplate=f"Tasa libre de riesgo<br>{rf:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=None,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"color": "#f5f5f5", "size": 12},
        margin={"l": 64, "r": 130, "t": 16, "b": 64},
        height=560,
        autosize=True,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"color": "#f5f5f5", "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
            "tracegroupgap": 6,
        },
        xaxis={
            "title": {"text": "Volatilidad anual (riesgo)", "font": {"color": "#f5f5f5"}},
            "tickformat": ".0%",
            "tickfont": {"color": "#f5f5f5"},
            "gridcolor": "#222222",
            "zerolinecolor": "#444444",
            "range": [x_lo, x_hi],
            "constrain": "domain",
        },
        yaxis={
            "title": {"text": "Rendimiento esperado anual (lineal)", "font": {"color": "#f5f5f5"}},
            "tickformat": ".0%",
            "tickfont": {"color": "#f5f5f5"},
            "gridcolor": "#222222",
            "zerolinecolor": "#444444",
            "range": [y_lo, y_hi],
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
        },
    )
    return fig


def _frontier_axis_bounds(result, rf: float) -> tuple[float, float, float, float]:
    """Encuadre 1:1 centrado en la hipérbola y los puntos clave."""
    xs = np.concatenate(
        [
            result.volatilities,
            np.array([0.0, result.equal_weight.volatility, result.tangency.volatility]),
            np.array([result.min_variance.volatility]),
        ]
    )
    ys = np.concatenate(
        [
            result.expected_returns,
            np.array([rf, result.equal_weight.expected_return, result.tangency.expected_return]),
            np.array([result.min_variance.expected_return]),
        ]
    )
    pad = 0.10 * max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 0.05)
    x_lo = max(0.0, float(xs.min()) - pad)
    y_lo = max(0.0, float(ys.min()) - pad)
    span = max(float(xs.max()) - x_lo, float(ys.max()) - y_lo) + pad
    return x_lo, x_lo + span, y_lo, y_lo + span


def historical_wealth_figure(paths) -> go.Figure:
    """Evolución histórica base 100: pesos iguales, optimizado y S&P 500."""
    fig = go.Figure()

    series = [
        (paths.equal_weight, "Pesos iguales", "#ffab00", "solid"),
        (paths.optimized, "Optimizado (máx. Sharpe)", "#b388ff", "solid"),
        (paths.benchmark, "S&P 500", "#00c853", "dash"),
    ]
    for data, name, color, dash in series:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.to_numpy(dtype=float),
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5, "dash": dash},
                hovertemplate=(
                    f"{name}<br>Fecha=%{{x|%Y-%m}}"
                    "<br>Valor=$%{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=None,
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"color": "#f5f5f5", "size": 12},
        margin={"l": 64, "r": 24, "t": 24, "b": 56},
        height=480,
        autosize=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": "#f5f5f5", "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "title": {"text": "Tiempo", "font": {"color": "#f5f5f5"}},
            "tickfont": {"color": "#f5f5f5"},
            "gridcolor": "#222222",
            "zerolinecolor": "#444444",
        },
        yaxis={
            "title": {"text": "Valor acumulado ($)", "font": {"color": "#f5f5f5"}},
            "tickfont": {"color": "#f5f5f5"},
            "gridcolor": "#222222",
            "zerolinecolor": "#444444",
            "tickprefix": "$",
        },
        hovermode="x unified",
    )
    return fig
