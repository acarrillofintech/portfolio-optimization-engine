"""Tests for historical market data loading."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    calculate_simple_returns,
    download_adjusted_prices,
)


def test_download_adjusted_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloaded prices should be normalized and aligned."""
    dates = pd.date_range("2024-01-01", periods=4)

    columns = pd.MultiIndex.from_product(
        [["Close"], ["SPY", "QQQ"]]
    )

    downloaded_data = pd.DataFrame(
        [
            [100.0, 200.0],
            [101.0, 202.0],
            [np.nan, 204.0],
            [103.0, 206.0],
        ],
        index=dates,
        columns=columns,
    )

    received_arguments: dict[str, object] = {}

    def fake_download(**kwargs):
        received_arguments.update(kwargs)
        return downloaded_data

    monkeypatch.setattr(
        "src.data_loader.yf.download",
        fake_download,
    )

    result = download_adjusted_prices(
        tickers=[" spy ", "qqq"],
        start_date="2024-01-01",
        end_date="2024-02-01",
    )

    assert list(result.columns) == ["SPY", "QQQ"]
    assert len(result) == 3
    assert result.index.name == "Date"
    assert result.columns.name == "Ticker"
    assert received_arguments["auto_adjust"] is True
    assert received_arguments["threads"] is False
    assert received_arguments["progress"] is False


def test_empty_download_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty market response must be rejected."""
    monkeypatch.setattr(
        "src.data_loader.yf.download",
        lambda **kwargs: pd.DataFrame(),
    )

    with pytest.raises(
        RuntimeError,
        match="No market data was downloaded",
    ):
        download_adjusted_prices(
            ["SPY"],
            "2024-01-01",
            "2024-02-01",
        )


def test_missing_ticker_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every requested asset must contain price data."""
    dates = pd.date_range("2024-01-01", periods=2)

    columns = pd.MultiIndex.from_product(
        [["Close"], ["SPY"]]
    )

    downloaded_data = pd.DataFrame(
        [[100.0], [101.0]],
        index=dates,
        columns=columns,
    )

    monkeypatch.setattr(
        "src.data_loader.yf.download",
        lambda **kwargs: downloaded_data,
    )

    with pytest.raises(
        RuntimeError,
        match="No price data was found for: TLT",
    ):
        download_adjusted_prices(
            ["SPY", "TLT"],
            "2024-01-01",
            "2024-02-01",
        )


def test_unexpected_download_format_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloaded data must use the expected column structure."""
    downloaded_data = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )

    monkeypatch.setattr(
        "src.data_loader.yf.download",
        lambda **kwargs: downloaded_data,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected format",
    ):
        download_adjusted_prices(
            ["SPY"],
            "2024-01-01",
            "2024-02-01",
        )


def test_missing_close_column_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Downloaded data must contain closing prices."""
    columns = pd.MultiIndex.from_product(
        [["Open"], ["SPY"]]
    )

    downloaded_data = pd.DataFrame(
        [[100.0], [101.0]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=columns,
    )

    monkeypatch.setattr(
        "src.data_loader.yf.download",
        lambda **kwargs: downloaded_data,
    )

    with pytest.raises(
        RuntimeError,
        match="does not contain closing prices",
    ):
        download_adjusted_prices(
            ["SPY"],
            "2024-01-01",
            "2024-02-01",
        )


def test_tickers_cannot_be_a_single_string() -> None:
    """Ticker input must be a sequence, not one string."""
    with pytest.raises(
        TypeError,
        match="sequence of strings",
    ):
        download_adjusted_prices(
            "SPY",  # type: ignore[arg-type]
            "2024-01-01",
            "2024-02-01",
        )


def test_duplicate_tickers_raise_value_error() -> None:
    """Ticker symbols must be unique."""
    with pytest.raises(
        ValueError,
        match="Ticker symbols must be unique",
    ):
        download_adjusted_prices(
            ["SPY", "spy"],
            "2024-01-01",
            "2024-02-01",
        )


def test_invalid_date_order_raises_value_error() -> None:
    """Start date must occur before the end date."""
    with pytest.raises(
        ValueError,
        match="Start date must be earlier than end date",
    ):
        download_adjusted_prices(
            ["SPY"],
            "2024-02-01",
            "2024-01-01",
        )


def test_calculate_simple_returns() -> None:
    """Simple returns should be calculated correctly."""
    prices = pd.DataFrame(
        {
            "ASSET_A": [100.0, 110.0, 121.0],
            "ASSET_B": [200.0, 180.0, 198.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )

    result = calculate_simple_returns(prices)

    expected = np.array(
        [
            [0.10, -0.10],
            [0.10, 0.10],
        ]
    )

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
    )


def test_prices_must_be_dataframe() -> None:
    """Prices must use a pandas DataFrame."""
    with pytest.raises(
        TypeError,
        match="Prices must be a pandas DataFrame",
    ):
        calculate_simple_returns(
            np.array([100.0, 101.0])  # type: ignore[arg-type]
        )


def test_empty_prices_raise_value_error() -> None:
    """Price data cannot be empty."""
    with pytest.raises(
        ValueError,
        match="Prices cannot be empty",
    ):
        calculate_simple_returns(pd.DataFrame())


def test_single_observation_raises_value_error() -> None:
    """At least two prices are needed to calculate returns."""
    prices = pd.DataFrame({"SPY": [100.0]})

    with pytest.raises(
        ValueError,
        match="at least two observations",
    ):
        calculate_simple_returns(prices)


def test_non_finite_prices_raise_value_error() -> None:
    """Prices cannot contain missing or infinite values."""
    prices = pd.DataFrame(
        {
            "SPY": [100.0, np.nan],
            "QQQ": [200.0, 201.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Prices must contain only finite values",
    ):
        calculate_simple_returns(prices)


def test_non_positive_prices_raise_value_error() -> None:
    """Prices must be strictly positive."""
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 0.0],
            "QQQ": [200.0, 201.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Prices must be greater than zero",
    ):
        calculate_simple_returns(prices)


def test_non_numeric_prices_raise_value_error() -> None:
    """Price values must be numeric."""
    prices = pd.DataFrame(
        {
            "SPY": [100.0, "invalid"],
            "QQQ": [200.0, 201.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="Prices must contain numeric values",
    ):
        calculate_simple_returns(prices)