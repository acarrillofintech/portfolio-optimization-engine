"""Tests for portfolio metrics."""

import numpy as np
import pytest

from src.portfolio_metrics import (
    PortfolioMetrics,
    annualized_covariance_matrix,
    annualized_expected_returns,
    calculate_portfolio_metrics,
    calculate_sharpe_ratio,
    portfolio_expected_return,
    portfolio_volatility,
)


@pytest.fixture
def expected_returns() -> np.ndarray:
    """Provide annual expected returns for three assets."""
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


@pytest.fixture
def weights() -> np.ndarray:
    """Provide valid portfolio weights."""
    return np.array([0.40, 0.35, 0.25])


def test_annualized_expected_returns() -> None:
    """Daily arithmetic means should be annualized correctly."""
    returns = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.08],
        ]
    )

    result = annualized_expected_returns(
        returns,
        periods_per_year=252,
    )

    expected = np.array([7.56, 11.76])

    np.testing.assert_allclose(result, expected)


def test_annualized_covariance_matrix() -> None:
    """The sample covariance matrix should be annualized correctly."""
    returns = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.08],
        ]
    )

    result = annualized_covariance_matrix(
        returns,
        periods_per_year=252,
    )

    expected = np.array(
        [
            [0.1008, 0.1512],
            [0.1512, 0.2352],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_portfolio_expected_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> None:
    """Portfolio return should equal the weighted asset returns."""
    result = portfolio_expected_return(
        weights,
        expected_returns,
    )

    assert result == pytest.approx(0.109)


def test_portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Portfolio volatility should follow the quadratic formula."""
    result = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    expected_volatility = np.sqrt(0.02306125)

    assert result == pytest.approx(expected_volatility)


def test_calculate_sharpe_ratio() -> None:
    """Sharpe ratio should measure excess return per unit of risk."""
    volatility = np.sqrt(0.02306125)

    result = calculate_sharpe_ratio(
        expected_return=0.109,
        volatility=volatility,
        risk_free_rate=0.03,
    )

    expected_sharpe_ratio = (
        (0.109 - 0.03) / volatility
    )

    assert result == pytest.approx(expected_sharpe_ratio)


def test_calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """The combined function should return all portfolio metrics."""
    result = calculate_portfolio_metrics(
        weights=weights,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    expected_volatility = np.sqrt(0.02306125)
    expected_sharpe_ratio = (
        (0.109 - 0.03) / expected_volatility
    )

    assert isinstance(result, PortfolioMetrics)
    assert result.expected_return == pytest.approx(0.109)
    assert result.volatility == pytest.approx(
        expected_volatility
    )
    assert result.sharpe_ratio == pytest.approx(
        expected_sharpe_ratio
    )


@pytest.mark.parametrize("periods_per_year", [0, -1])
def test_invalid_periods_per_year_raise_value_error(
    periods_per_year: int,
) -> None:
    """Annualization periods must be positive."""
    returns = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Periods per year must be greater than zero",
    ):
        annualized_expected_returns(
            returns,
            periods_per_year=periods_per_year,
        )


def test_one_dimensional_returns_raise_value_error() -> None:
    """Historical returns must be represented as a matrix."""
    returns = np.array([0.01, 0.02, 0.03])

    with pytest.raises(
        ValueError,
        match="Returns must be a two-dimensional array",
    ):
        annualized_expected_returns(returns)


def test_non_finite_returns_raise_value_error() -> None:
    """Returns containing missing values must be rejected."""
    returns = np.array(
        [
            [0.01, np.nan],
            [0.02, 0.03],
        ]
    )

    with pytest.raises(
        ValueError,
        match="Returns must contain only finite values",
    ):
        annualized_expected_returns(returns)


def test_weights_must_sum_to_one(
    expected_returns: np.ndarray,
) -> None:
    """Portfolio weights must represent all invested capital."""
    invalid_weights = np.array([0.40, 0.30, 0.20])

    with pytest.raises(
        ValueError,
        match="Portfolio weights must sum to 1",
    ):
        portfolio_expected_return(
            invalid_weights,
            expected_returns,
        )


def test_weight_count_must_match_assets(
    expected_returns: np.ndarray,
) -> None:
    """Every asset must have one corresponding weight."""
    invalid_weights = np.array([0.50, 0.50])

    with pytest.raises(
        ValueError,
        match=(
            "The number of weights must match "
            "the number of assets"
        ),
    ):
        portfolio_expected_return(
            invalid_weights,
            expected_returns,
        )


def test_covariance_matrix_must_be_square(
    weights: np.ndarray,
) -> None:
    """A covariance matrix must have equal dimensions."""
    invalid_covariance = np.ones((3, 2))

    with pytest.raises(
        ValueError,
        match="Covariance matrix must be square",
    ):
        portfolio_volatility(
            weights,
            invalid_covariance,
        )


def test_negative_variance_raises_value_error() -> None:
    """A materially negative portfolio variance is invalid."""
    weights = np.array([1.0])
    invalid_covariance = np.array([[-0.04]])

    with pytest.raises(
        ValueError,
        match="Portfolio variance cannot be negative",
    ):
        portfolio_volatility(
            weights,
            invalid_covariance,
        )


@pytest.mark.parametrize("volatility", [0.0, -0.10])
def test_non_positive_volatility_raises_value_error(
    volatility: float,
) -> None:
    """Sharpe ratio requires strictly positive volatility."""
    with pytest.raises(
        ValueError,
        match="Volatility must be greater than zero",
    ):
        calculate_sharpe_ratio(
            expected_return=0.10,
            volatility=volatility,
            risk_free_rate=0.03,
        )