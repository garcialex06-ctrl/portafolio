"""
Motor cuantitativo del portafolio (NumPy + SciPy, sin PyPortfolioOpt).

Importar submódulos directamente (`src.core.metrics`, `src.core.portfolio`)
para evitar cargas circulares al inicializar el paquete.
"""

from src.core.metrics import (
    PERIODS_PER_YEAR,
    calcular_matriz_correlaciones,
    calcular_matriz_covarianzas,
    calcular_rendimiento_esperado_anualizado,
    calcular_rendimientos,
    calcular_volatilidad_anualizada,
)

__all__ = [
    "PERIODS_PER_YEAR",
    "calcular_rendimientos",
    "calcular_rendimiento_esperado_anualizado",
    "calcular_volatilidad_anualizada",
    "calcular_matriz_covarianzas",
    "calcular_matriz_correlaciones",
]
