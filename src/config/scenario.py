"""Funcionalidad 1: universo de inversión, pesos forzados y tasa libre de riesgo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

DEFAULT_RF_ANNUAL = 0.04
RF_ANNUAL_MIN = -0.05
RF_ANNUAL_MAX = 0.25
MIN_SELECTED_ASSETS = 2
PERIODS_PER_YEAR = 12
_WEIGHT_TOLERANCE = 1e-8


class ScenarioConfigError(ValueError):
    """Error recuperable al validar la configuración del portafolio."""


@dataclass(frozen=True)
class PortfolioScenario:
    """Universo y supuestos confirmados para las funcionalidades 2-4."""

    selected_tickers: tuple[str, ...]
    forced_weights: dict[str, float]
    rf_annual: float
    rf_monthly: float
    used_default_rf: bool

    @property
    def forced_weight_sum(self) -> float:
        return float(sum(self.forced_weights.values()))

    @property
    def free_weight(self) -> float:
        return 1.0 - self.forced_weight_sum


@dataclass(frozen=True)
class ScenarioValidation:
    scenario: PortfolioScenario | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.scenario is not None and not self.errors


def annual_to_monthly(rf_annual: float) -> float:
    """Convierte una tasa anual a mensual (r / 12), alineada a precios mensuales."""
    return rf_annual / PERIODS_PER_YEAR


def resolve_risk_free_rate(rf_annual_percent: float | None) -> tuple[float, bool]:
    """Interpreta el input en % anual. Vacío o None usa 4%."""
    if rf_annual_percent is None:
        return DEFAULT_RF_ANNUAL, True
    if not _is_finite_number(rf_annual_percent):
        raise ScenarioConfigError("La tasa libre de riesgo debe ser un número.")
    return float(rf_annual_percent) / 100.0, False


def build_scenario(
    *,
    available_tickers: Sequence[str],
    selected_tickers: Sequence[str],
    forced_weights_percent: Mapping[str, float | None],
    rf_annual_percent: float | None,
) -> ScenarioValidation:
    """Valida inputs de usuario y, si son coherentes, construye el escenario."""
    errors: list[str] = []
    available = tuple(str(t) for t in available_tickers)
    available_set = set(available)

    selected = _unique_keep_order(str(ticker) for ticker in selected_tickers)
    unknown_selected = [t for t in selected if t not in available_set]
    if unknown_selected:
        errors.append(
            "Hay activos seleccionados que no pertenecen al universo válido: "
            + ", ".join(unknown_selected)
            + "."
        )

    if not selected:
        errors.append("Selecciona al menos un activo para el portafolio.")
    elif len(selected) < MIN_SELECTED_ASSETS:
        errors.append(
            f"Selecciona al menos {MIN_SELECTED_ASSETS} activos para poder "
            "diversificar y optimizar el portafolio."
        )

    selected_set = set(selected)
    forced_weights, weight_errors = _parse_forced_weights(
        forced_weights_percent, selected_set, available_set
    )
    errors.extend(weight_errors)

    try:
        rf_annual, used_default = resolve_risk_free_rate(rf_annual_percent)
        errors.extend(_validate_risk_free_rate(rf_annual, used_default))
    except ScenarioConfigError as exc:
        rf_annual, used_default = DEFAULT_RF_ANNUAL, False
        errors.append(str(exc))

    if errors:
        return ScenarioValidation(scenario=None, errors=tuple(errors))

    scenario = PortfolioScenario(
        selected_tickers=selected,
        forced_weights=forced_weights,
        rf_annual=rf_annual,
        rf_monthly=annual_to_monthly(rf_annual),
        used_default_rf=used_default,
    )
    return ScenarioValidation(scenario=scenario, errors=())


def _parse_forced_weights(
    forced_weights_percent: Mapping[str, float | None],
    selected_set: set[str],
    available_set: set[str],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    parsed: dict[str, float] = {}

    for ticker, percent in forced_weights_percent.items():
        name = str(ticker)
        if name not in available_set:
            errors.append(f"No se puede forzar un peso sobre {name}: no está en el universo válido.")
            continue
        if name not in selected_set:
            errors.append(
                f"No se puede forzar un peso sobre {name} porque no está incluido en el portafolio."
            )
            continue
        if percent is None:
            errors.append(f"Indica el peso forzado de {name} o quítalo de la lista.")
            continue
        if not _is_finite_number(percent):
            errors.append(f"El peso forzado de {name} debe ser un número.")
            continue
        if percent <= 0:
            errors.append(f"El peso forzado de {name} debe ser mayor que 0%.")
            continue
        if percent > 100:
            errors.append(f"El peso forzado de {name} no puede superar 100%.")
            continue
        parsed[name] = float(percent) / 100.0

    weight_sum = sum(parsed.values())
    if weight_sum > 1.0 + _WEIGHT_TOLERANCE:
        errors.append(
            f"La suma de los pesos forzados es {weight_sum:.1%} y no puede superar 100%."
        )
    leftover_selected = [t for t in selected_set if t not in parsed]
    if weight_sum >= 1.0 - _WEIGHT_TOLERANCE and leftover_selected:
        errors.append(
            "Los pesos forzados suman 100%, pero hay activos seleccionados sin peso. "
            "Deselecciónalos o deja margen inferior a 100% para optimizar el resto."
        )

    return parsed, errors


def _validate_risk_free_rate(rf_annual: float, used_default: bool) -> list[str]:
    if used_default:
        return []
    if rf_annual < RF_ANNUAL_MIN:
        return [
            "La tasa libre de riesgo anual no puede ser inferior a "
            f"{RF_ANNUAL_MIN:.0%}. No se aceptan valores negativos extremos."
        ]
    if rf_annual > RF_ANNUAL_MAX:
        return [
            "La tasa libre de riesgo anual no puede superar "
            f"{RF_ANNUAL_MAX:.0%}. Revisa el valor; el modelo espera una tasa tipo Treasury."
        ]
    return []


def _unique_keep_order(tickers: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            ordered.append(ticker)
    return tuple(ordered)


def _is_finite_number(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))
