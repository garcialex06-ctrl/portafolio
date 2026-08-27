"""Funcionalidad 0: carga, validación y exclusión de activos incompletos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "datos" / "portafolio_21_activos.csv"
DATE_COLUMN = "Date"
BENCHMARK_TICKERS = ("^GSPC", "GSPC", "SPX", "SP500", "S&P500")
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_CSV_MB = MAX_CSV_BYTES // (1024 * 1024)


class DataPreparationError(ValueError):
    """Error recuperable al preparar el CSV histórico."""


@dataclass(frozen=True)
class PreparedData:
    """Universo listo para la Funcionalidad 1.

    `prices` solo incluye activos de inversión con historial completo.
    El S&P 500 queda en `benchmark` y no forma parte del universo invertible.
    """

    prices: pd.DataFrame
    benchmark: pd.Series | None
    valid_tickers: tuple[str, ...]
    omitted_tickers: tuple[str, ...]
    messages: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def n_periods(self) -> int:
        return int(len(self.prices.index))

    @property
    def benchmark_ticker(self) -> str | None:
        if self.benchmark is None:
            return None
        return str(self.benchmark.name)


def load_portfolio_data(csv_path: str | Path | None = None) -> PreparedData:
    """Carga el CSV por defecto y deja solo activos con datos completos."""
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if not path.is_file():
        raise DataPreparationError(f"No se encontró el archivo de datos: {path}")
    raw = pd.read_csv(path)
    return prepare_portfolio_data(raw)


def load_uploaded_csv(filename: str, data: bytes) -> PreparedData:
    """Valida un CSV subido por el usuario y prepara el universo invertible."""
    _assert_csv_filename(filename)
    if len(data) > MAX_CSV_BYTES:
        raise DataPreparationError(
            f"El archivo supera el tamaño máximo de {MAX_CSV_MB} MB."
        )
    if not data.strip():
        raise DataPreparationError("El archivo está vacío.")
    raw = _read_csv_bytes(data)
    return prepare_portfolio_data(raw)


def prepare_portfolio_data(
    raw: pd.DataFrame,
    *,
    date_column: str = DATE_COLUMN,
    benchmark_tickers: tuple[str, ...] = BENCHMARK_TICKERS,
) -> PreparedData:
    """Valida el panel de precios y excluye columnas con huecos o no finitos."""
    if raw.empty:
        raise DataPreparationError("El archivo de datos está vacío.")

    date_col = _resolve_date_column(raw, date_column)
    frame = raw.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    invalid_dates = int(frame[date_col].isna().sum())
    if invalid_dates:
        raise DataPreparationError(
            f"Hay {invalid_dates} fecha(s) inválida(s) en la columna '{date_col}'."
        )

    frame = frame.set_index(date_col).sort_index()
    if frame.index.has_duplicates:
        dupes = int(frame.index.duplicated().sum())
        raise DataPreparationError(f"Hay {dupes} fecha(s) duplicada(s) en el archivo.")

    prices = frame.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(how="all")
    if prices.empty:
        raise DataPreparationError("No hay observaciones de precios utilizables.")

    incomplete = _incomplete_columns(prices)
    complete_cols = [col for col in prices.columns if col not in incomplete]
    omitted_tickers = tuple(str(col) for col in incomplete)
    messages = tuple(_omission_message(col, benchmark_tickers) for col in omitted_tickers)

    complete = prices[complete_cols]
    benchmark_name = _find_benchmark(complete.columns, benchmark_tickers)
    investable_cols = [col for col in complete.columns if col != benchmark_name]
    if not investable_cols:
        raise DataPreparationError(
            "No quedan activos de inversión con datos completos. "
            "Revisa el CSV e intenta de nuevo."
        )

    investable = complete[investable_cols].astype(float)
    benchmark = (
        complete[benchmark_name].astype(float).rename(benchmark_name)
        if benchmark_name is not None
        else None
    )

    return PreparedData(
        prices=investable,
        benchmark=benchmark,
        valid_tickers=tuple(str(col) for col in investable.columns),
        omitted_tickers=omitted_tickers,
        messages=messages,
        start=pd.Timestamp(investable.index.min()),
        end=pd.Timestamp(investable.index.max()),
    )


def _assert_csv_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    if Path(name).suffix.lower() != ".csv":
        raise DataPreparationError("Solo se aceptan archivos .csv.")
    if not name.strip("."):
        raise DataPreparationError("El nombre del archivo no es válido.")
    return name


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataPreparationError(
                "No se pudo interpretar el CSV. Revisa delimitadores y la fila de encabezados."
            ) from exc
    raise DataPreparationError("No se pudo leer el CSV (codificación no reconocida).") from last_error


def _resolve_date_column(raw: pd.DataFrame, date_column: str) -> str:
    if date_column in raw.columns:
        return date_column
    lowered = {str(col).strip().lower(): col for col in raw.columns}
    if "date" in lowered:
        return lowered["date"]
    raise DataPreparationError(
        f"El archivo debe incluir una columna de fechas llamada '{date_column}'."
    )


def _incomplete_columns(prices: pd.DataFrame) -> list[str]:
    finite = np.isfinite(prices.to_numpy(dtype=float, copy=False))
    n_invalid = (~finite).sum(axis=0)
    return [str(col) for col, n_missing in zip(prices.columns, n_invalid) if n_missing > 0]


def _find_benchmark(columns: pd.Index, benchmark_tickers: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().upper(): str(col) for col in columns}
    for candidate in benchmark_tickers:
        key = candidate.strip().upper()
        if key in normalized:
            return normalized[key]
    return None


def _omission_message(ticker: str, benchmark_tickers: tuple[str, ...]) -> str:
    is_benchmark = ticker.strip().upper() in {t.strip().upper() for t in benchmark_tickers}
    label = "el índice" if is_benchmark else "la empresa"
    return f"Se omitió {label} {ticker} por datos incompletos."
