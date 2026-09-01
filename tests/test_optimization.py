"""Tests for portfolio optimization."""

import numpy as np
import pytest

from src.optimization import (
    PortfolioOptimizationResult,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
)
from src.portfolio_metrics import (
    calculate_portfolio_metrics,
)


@pytest.fixture
def expected_returns() -> np.ndarray:
    """Provide annual expected returns."""
    return np.array([0.10, 0.14, 0.08])


@pytest.fixture
def covariance_matrix() -> np.ndarray:
    """Provide an annual covariance matrix."""
    return np.array(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0900, 0.0100],
            [0.0040, 0.0100, 0.0225],
        ]
    )


def test_minimum_variance_weights_match_known_solution(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Minimum-variance weights should match the known solution."""
    result = minimum_variance_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    expected_weights = np.array(
        [0.31670462, 0.08548955, 0.59780584]
    )

    np.testing.assert_allclose(
        result.weights,
        expected_weights,
        atol=1e-4,
    )


def test_maximum_sharpe_weights_match_known_solution(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Maximum-Sharpe weights should match the known solution."""
    result = maximum_sharpe_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    expected_weights = np.array(
        [0.36810023, 0.24177416, 0.39012562]
    )

    np.testing.assert_allclose(
        result.weights,
        expected_weights,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "optimizer",
    [
        minimum_variance_portfolio,
        maximum_sharpe_portfolio,
    ],
)
def test_optimized_weights_sum_to_one(
    optimizer,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Every optimized portfolio must invest all available capital."""
    result = optimizer(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    assert result.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "optimizer",
    [
        minimum_variance_portfolio,
        maximum_sharpe_portfolio,
    ],
)
def test_optimized_portfolios_are_long_only(
    optimizer,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Long-only portfolios cannot contain negative weights."""
    result = optimizer(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    assert np.all(result.weights >= 0.0)
    assert np.all(result.weights <= 1.0)


def test_minimum_variance_beats_equal_weights(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Minimum-variance portfolio should reduce volatility."""
    optimized = minimum_variance_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    equal_weight_metrics = calculate_portfolio_metrics(
        weights=np.full(3, 1.0 / 3.0),
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    assert (
        optimized.metrics.volatility
        <= equal_weight_metrics.volatility
    )


def test_maximum_sharpe_beats_equal_weights(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Maximum-Sharpe portfolio should improve risk-adjusted return."""
    optimized = maximum_sharpe_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    equal_weight_metrics = calculate_portfolio_metrics(
        weights=np.full(3, 1.0 / 3.0),
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    assert (
        optimized.metrics.sharpe_ratio
        >= equal_weight_metrics.sharpe_ratio
    )


def test_result_uses_optimization_dataclass(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Optimization should return a structured result."""
    result = minimum_variance_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    assert isinstance(result, PortfolioOptimizationResult)


def test_one_dimensional_covariance_raises_value_error(
    expected_returns: np.ndarray,
) -> None:
    """Covariance input must be a matrix."""
    covariance_matrix = np.array([0.04, 0.09, 0.0225])

    with pytest.raises(
        ValueError,
        match="Covariance matrix must be two-dimensional",
    ):
        minimum_variance_portfolio(
            expected_returns,
            covariance_matrix,
        )


def test_mismatched_dimensions_raise_value_error(
    expected_returns: np.ndarray,
) -> None:
    """Covariance dimensions must match the assets."""
    covariance_matrix = np.eye(2)

    with pytest.raises(
        ValueError,
        match=(
            "Covariance matrix dimensions must match "
            "the number of assets"
        ),
    ):
        minimum_variance_portfolio(
            expected_returns,
            covariance_matrix,
        )


def test_non_symmetric_covariance_raises_value_error(
    expected_returns: np.ndarray,
) -> None:
    """A covariance matrix must be symmetric."""
    covariance_matrix = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.02, 0.09, 0.01],
            [0.00, 0.01, 0.03],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Covariance matrix must be symmetric",
    ):
        minimum_variance_portfolio(
            expected_returns,
            covariance_matrix,
        )


def test_non_positive_semidefinite_covariance_raises_error(
    expected_returns: np.ndarray,
) -> None:
    """A covariance matrix cannot contain impossible risk."""
    covariance_matrix = np.array(
        [
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Covariance matrix must be "
            "positive semidefinite"
        ),
    ):
        minimum_variance_portfolio(
            expected_returns,
            covariance_matrix,
        )


def test_zero_covariance_matrix_raises_value_error(
    expected_returns: np.ndarray,
) -> None:
    """Optimization requires assets with measurable risk."""
    covariance_matrix = np.zeros((3, 3))

    with pytest.raises(
        ValueError,
        match="Covariance matrix must contain positive risk",
    ):
        minimum_variance_portfolio(
            expected_returns,
            covariance_matrix,
        )


def test_non_finite_risk_free_rate_raises_value_error(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Risk-free rate must be a valid finite number."""
    with pytest.raises(
        ValueError,
        match="Risk-free rate must be finite",
    ):
        maximum_sharpe_portfolio(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            risk_free_rate=np.nan,
        )