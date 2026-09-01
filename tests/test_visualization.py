"""Tests for portfolio visualizations."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    PortfolioOptimizationResult,
)
from src.portfolio_metrics import PortfolioMetrics
from src.simulation import PortfolioSimulationResult
from src.visualization import (
    create_all_visualizations,
    plot_correlation_matrix,
    plot_normalized_prices,
    plot_portfolio_simulation,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Provide sample historical prices."""
    return pd.DataFrame(
        {
            "ASSET_A": [100.0, 102.0, 104.0, 103.0],
            "ASSET_B": [100.0, 99.0, 101.0, 105.0],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )


@pytest.fixture
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Provide sample asset returns."""
    return prices.pct_change().dropna()


@pytest.fixture
def simulation() -> PortfolioSimulationResult:
    """Provide a small portfolio simulation."""
    weights = np.array(
        [
            [0.20, 0.80],
            [0.50, 0.50],
            [0.80, 0.20],
        ]
    )

    return PortfolioSimulationResult(
        weights=weights,
        expected_returns=np.array([0.08, 0.10, 0.12]),
        volatilities=np.array([0.10, 0.13, 0.18]),
        sharpe_ratios=np.array([0.50, 0.5385, 0.50]),
    )


@pytest.fixture
def minimum_variance() -> PortfolioOptimizationResult:
    """Provide a minimum-variance portfolio."""
    return PortfolioOptimizationResult(
        weights=np.array([0.20, 0.80]),
        metrics=PortfolioMetrics(
            expected_return=0.08,
            volatility=0.10,
            sharpe_ratio=0.50,
        ),
    )


@pytest.fixture
def maximum_sharpe() -> PortfolioOptimizationResult:
    """Provide a maximum-Sharpe portfolio."""
    return PortfolioOptimizationResult(
        weights=np.array([0.50, 0.50]),
        metrics=PortfolioMetrics(
            expected_return=0.10,
            volatility=0.13,
            sharpe_ratio=0.5385,
        ),
    )


def _assert_valid_image(path: Path) -> None:
    """Verify that an image file exists and is not empty."""
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0


def test_plot_normalized_prices_creates_image(
    prices: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Normalized price chart should be saved."""
    result = plot_normalized_prices(
        prices,
        output_directory=tmp_path,
    )

    _assert_valid_image(result)


def test_plot_correlation_matrix_creates_image(
    returns: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Correlation heatmap should be saved."""
    result = plot_correlation_matrix(
        returns,
        output_directory=tmp_path,
    )

    _assert_valid_image(result)


def test_plot_portfolio_simulation_creates_image(
    simulation: PortfolioSimulationResult,
    minimum_variance: PortfolioOptimizationResult,
    maximum_sharpe: PortfolioOptimizationResult,
    tmp_path: Path,
) -> None:
    """Risk-return chart should be saved."""
    result = plot_portfolio_simulation(
        simulation=simulation,
        minimum_variance=minimum_variance,
        maximum_sharpe=maximum_sharpe,
        output_directory=tmp_path,
    )

    _assert_valid_image(result)


def test_create_all_visualizations(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    simulation: PortfolioSimulationResult,
    minimum_variance: PortfolioOptimizationResult,
    maximum_sharpe: PortfolioOptimizationResult,
    tmp_path: Path,
) -> None:
    """The complete visualization pipeline should create three images."""
    generated_files = create_all_visualizations(
        prices=prices,
        returns=returns,
        simulation=simulation,
        minimum_variance=minimum_variance,
        maximum_sharpe=maximum_sharpe,
        output_directory=tmp_path,
    )

    assert len(generated_files) == 3

    for generated_file in generated_files:
        _assert_valid_image(generated_file)


def test_empty_prices_raise_value_error(
    tmp_path: Path,
) -> None:
    """Price visualization requires historical data."""
    with pytest.raises(
        ValueError,
        match="Prices must be a non-empty pandas DataFrame",
    ):
        plot_normalized_prices(
            pd.DataFrame(),
            output_directory=tmp_path,
        )


def test_empty_returns_raise_value_error(
    tmp_path: Path,
) -> None:
    """Correlation visualization requires return data."""
    with pytest.raises(
        ValueError,
        match="Returns must be a non-empty pandas DataFrame",
    ):
        plot_correlation_matrix(
            pd.DataFrame(),
            output_directory=tmp_path,
        )


def test_empty_simulation_raises_value_error(
    minimum_variance: PortfolioOptimizationResult,
    maximum_sharpe: PortfolioOptimizationResult,
    tmp_path: Path,
) -> None:
    """Portfolio chart requires simulated portfolios."""
    empty_simulation = PortfolioSimulationResult(
        weights=np.empty((0, 2)),
        expected_returns=np.array([]),
        volatilities=np.array([]),
        sharpe_ratios=np.array([]),
    )

    with pytest.raises(
        ValueError,
        match="Simulation must contain at least one portfolio",
    ):
        plot_portfolio_simulation(
            simulation=empty_simulation,
            minimum_variance=minimum_variance,
            maximum_sharpe=maximum_sharpe,
            output_directory=tmp_path,
        )