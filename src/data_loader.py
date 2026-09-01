"""Download and prepare historical financial market data."""

from collections.abc import Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf


DateInput = str | date | datetime


def _validate_tickers(
    tickers: Sequence[str],
) -> list[str]:
    """Validate and normalize ticker symbols."""
    if isinstance(tickers, str):
        raise TypeError(
            "Tickers must be provided as a sequence of strings."
        )

    normalized_tickers = [
        ticker.strip().upper()
        for ticker in tickers
        if isinstance(ticker, str) and ticker.strip()
    ]

    if not normalized_tickers:
        raise ValueError(
            "At least one valid ticker must be provided."
        )

    if len(normalized_tickers) != len(tickers):
        raise ValueError(
            "Every ticker must be a non-empty string."
        )

    if len(set(normalized_tickers)) != len(
        normalized_tickers
    ):
        raise ValueError("Ticker symbols must be unique.")

    return normalized_tickers


def _validate_dates(
    start_date: DateInput,
    end_date: DateInput,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate the requested historical period."""
    try:
        normalized_start = pd.Timestamp(start_date)
        normalized_end = pd.Timestamp(end_date)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Start and end dates must be valid dates."
        ) from error

    if pd.isna(normalized_start) or pd.isna(normalized_end):
        raise ValueError(
            "Start and end dates must be valid dates."
        )

    if normalized_start >= normalized_end:
        raise ValueError(
            "Start date must be earlier than end date."
        )

    return normalized_start, normalized_end


def download_adjusted_prices(
    tickers: Sequence[str],
    start_date: DateInput,
    end_date: DateInput,
) -> pd.DataFrame:
    """Download aligned adjusted closing prices from Yahoo Finance."""
    normalized_tickers = _validate_tickers(tickers)

    normalized_start, normalized_end = _validate_dates(
        start_date,
        end_date,
    )

    downloaded_data = yf.download(
        tickers=normalized_tickers,
        start=normalized_start.strftime("%Y-%m-%d"),
        end=normalized_end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
        multi_level_index=True,
    )

    if downloaded_data is None or downloaded_data.empty:
        raise RuntimeError(
            "No market data was downloaded."
        )

    if not isinstance(
        downloaded_data.columns,
        pd.MultiIndex,
    ):
        raise RuntimeError(
            "Downloaded market data has an unexpected format."
        )

    if "Close" not in downloaded_data.columns.get_level_values(0):
        raise RuntimeError(
            "Downloaded market data does not contain closing prices."
        )

    prices = downloaded_data["Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(
            name=normalized_tickers[0]
        )

    prices.columns = [
        str(column).upper()
        for column in prices.columns
    ]

    missing_tickers = [
        ticker
        for ticker in normalized_tickers
        if ticker not in prices.columns
        or prices[ticker].dropna().empty
    ]

    if missing_tickers:
        missing_symbols = ", ".join(missing_tickers)

        raise RuntimeError(
            f"No price data was found for: {missing_symbols}."
        )

    prices = prices.loc[:, normalized_tickers]
    prices = prices.dropna(axis=0, how="any")
    prices = prices.sort_index()

    if len(prices) < 2:
        raise RuntimeError(
            "At least two aligned price observations are required."
        )

    if not np.all(np.isfinite(prices.to_numpy())):
        raise RuntimeError(
            "Downloaded prices must contain only finite values."
        )

    if np.any(prices.to_numpy() <= 0):
        raise RuntimeError(
            "Downloaded prices must be greater than zero."
        )

    prices.index.name = "Date"
    prices.columns.name = "Ticker"

    return prices.astype(float)


def calculate_simple_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate simple periodic returns from historical prices."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("Prices must be a pandas DataFrame.")

    if prices.empty:
        raise ValueError("Prices cannot be empty.")

    if prices.shape[0] < 2:
        raise ValueError(
            "Prices must contain at least two observations."
        )

    if prices.shape[1] < 1:
        raise ValueError(
            "Prices must contain at least one asset."
        )

    try:
        price_values = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Prices must contain numeric values."
        ) from error

    if not np.all(np.isfinite(price_values)):
        raise ValueError(
            "Prices must contain only finite values."
        )

    if np.any(price_values <= 0):
        raise ValueError(
            "Prices must be greater than zero."
        )

    returns = prices.pct_change(
        fill_method=None
    ).dropna(axis=0, how="any")

    if returns.empty:
        raise ValueError(
            "Returns could not be calculated."
        )

    returns.index.name = prices.index.name
    returns.columns.name = prices.columns.name

    return returns.astype(float)


def main() -> None:
    """Download an educational diversified ETF portfolio."""
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

    prices = download_adjusted_prices(
        tickers=tickers,
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    returns = calculate_simple_returns(prices)

    print("Historical market data")
    print(f"Assets: {', '.join(prices.columns)}")
    print(
        f"Period: {prices.index.min().date()} "
        f"to {prices.index.max().date()}"
    )
    print(f"Price observations: {len(prices):,}")
    print(f"Return observations: {len(returns):,}")

    print("\nLatest adjusted prices")
    print(prices.tail())

    print("\nAverage daily returns")
    print(returns.mean().map(lambda value: f"{value:.4%}"))


if __name__ == "__main__":
    main()