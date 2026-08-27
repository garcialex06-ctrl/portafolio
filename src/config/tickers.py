"""Nombres de empresas para la interfaz de configuración."""

TICKER_NAMES: dict[str, str] = {
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "BAC": "Bank of America",
    "CVX": "Chevron",
    "DIS": "Disney",
    "GOOGL": "Alphabet",
    "IBM": "IBM",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase",
    "KO": "Coca-Cola",
    "MA": "Mastercard",
    "META": "Meta",
    "MSFT": "Microsoft",
    "NFLX": "Netflix",
    "NVDA": "NVIDIA",
    "PFE": "Pfizer",
    "TSLA": "Tesla",
    "V": "Visa",
    "WMT": "Walmart",
    "XOM": "ExxonMobil",
}


def ticker_label(ticker: str) -> str:
    name = TICKER_NAMES.get(ticker)
    return f"{ticker} · {name}" if name else ticker
