from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import (
    DEFAULT_CSV_PATH,
    DataPreparationError,
    load_portfolio_data,
    load_uploaded_csv,
    prepare_portfolio_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2020-10-01", "2020-11-01", "2020-12-01"],
            "AAPL": [100.0, 110.0, 120.0],
            "MSFT": [200.0, 210.0, 220.0],
            "^GSPC": [3000.0, 3100.0, 3200.0],
        }
    )


def test_real_csv_is_complete() -> None:
    prepared = load_portfolio_data(DEFAULT_CSV_PATH)
    assert prepared.n_periods == 60
    assert len(prepared.valid_tickers) == 20
    assert prepared.omitted_tickers == ()
    assert prepared.messages == ()
    assert prepared.benchmark_ticker == "^GSPC"
    assert "^GSPC" not in prepared.valid_tickers
    assert "^GSPC" not in prepared.prices.columns
    assert prepared.start == pd.Timestamp("2020-10-01")
    assert prepared.end == pd.Timestamp("2025-09-01")


def test_omits_company_with_missing_prices() -> None:
    raw = _sample_frame()
    raw.loc[1, "MSFT"] = None
    prepared = prepare_portfolio_data(raw)
    assert prepared.valid_tickers == ("AAPL",)
    assert prepared.omitted_tickers == ("MSFT",)
    assert prepared.messages == ("Se omitió la empresa MSFT por datos incompletos.",)
    assert prepared.benchmark_ticker == "^GSPC"


def test_omits_incomplete_benchmark() -> None:
    raw = _sample_frame()
    raw.loc[0, "^GSPC"] = float("nan")
    prepared = prepare_portfolio_data(raw)
    assert prepared.benchmark is None
    assert prepared.valid_tickers == ("AAPL", "MSFT")
    assert prepared.omitted_tickers == ("^GSPC",)
    assert prepared.messages == ("Se omitió el índice ^GSPC por datos incompletos.",)


def test_omits_non_finite_prices() -> None:
    raw = _sample_frame()
    raw.loc[2, "AAPL"] = float("inf")
    prepared = prepare_portfolio_data(raw)
    assert "AAPL" not in prepared.valid_tickers
    assert "Se omitió la empresa AAPL por datos incompletos." in prepared.messages


def test_raises_when_no_investable_assets_remain() -> None:
    raw = _sample_frame()
    raw.loc[0, "AAPL"] = None
    raw.loc[1, "MSFT"] = None
    with pytest.raises(DataPreparationError, match="No quedan activos"):
        prepare_portfolio_data(raw)


def test_raises_when_date_column_is_missing() -> None:
    raw = _sample_frame().drop(columns=["Date"])
    with pytest.raises(DataPreparationError, match="columna de fechas"):
        prepare_portfolio_data(raw)


def test_raises_when_csv_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no_existe.csv"
    with pytest.raises(DataPreparationError, match="No se encontró"):
        load_portfolio_data(missing)


def test_project_csv_exists() -> None:
    assert DEFAULT_CSV_PATH.is_file()
    assert DEFAULT_CSV_PATH.parent == PROJECT_ROOT / "datos"


def test_load_uploaded_csv_from_default_file() -> None:
    data = DEFAULT_CSV_PATH.read_bytes()
    prepared = load_uploaded_csv("portafolio_21_activos.csv", data)
    assert prepared.n_periods == 60
    assert len(prepared.valid_tickers) == 20


def test_rejects_non_csv_upload_name() -> None:
    with pytest.raises(DataPreparationError, match="Solo se aceptan archivos"):
        load_uploaded_csv("precios.xlsx", b"Date,AAPL\n2020-01-01,1")


def test_rejects_empty_upload() -> None:
    with pytest.raises(DataPreparationError, match="vacío"):
        load_uploaded_csv("nuevo.csv", b"   ")


def test_rejects_path_traversal_but_accepts_basename() -> None:
    payload = _sample_frame().to_csv(index=False).encode("utf-8")
    prepared = load_uploaded_csv("..\\secret.csv", payload)
    assert prepared.valid_tickers == ("AAPL", "MSFT")
