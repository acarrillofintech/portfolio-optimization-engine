"""Core metrics for quantitative portfolio analysis."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioMetrics:
    """Annualized return, volatility, and Sharpe ratio of a portfolio."""

    expected_return: float
    volatility: float
    sharpe_ratio: float


def _validate_returns(returns: FloatArray) -> FloatArray:
    """Validate and normalize a matrix of historical asset returns."""
    normalized_returns = np.asarray(returns, dtype=float)

    if normalized_returns.ndim != 2:
        raise ValueError("Returns must be a two-dimensional array.")

    observations, assets = normalized_returns.shape

    if observations < 2:
        raise ValueError("Returns must contain at least two observations.")

    if assets < 1:
        raise ValueError("Returns must contain at least one asset.")

    if not np.all(np.isfinite(normalized_returns)):
        raise ValueError("Returns must contain only finite values.")

    return normalized_returns


def _validate_weights(
    weights: FloatArray,
    number_of_assets: int,
) -> FloatArray:
    """Validate portfolio weights."""
    normalized_weights = np.asarray(weights, dtype=float)

    if normalized_weights.ndim != 1:
        raise ValueError("Weights must be a one-dimensional array.")

    if len(normalized_weights) != number_of_assets:
        raise ValueError(
            "The number of weights must match the number of assets."
        )

    if not np.all(np.isfinite(normalized_weights)):
        raise ValueError("Weights must contain only finite values.")

    if not np.isclose(normalized_weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return normalized_weights


def annualized_expected_returns(
    returns: FloatArray,
    periods_per_year: int = 252,
) -> FloatArray:
    """Calculate annualized arithmetic expected returns for every asset."""
    validated_returns = _validate_returns(returns)

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be greater than zero.")

    return np.mean(validated_returns, axis=0) * periods_per_year


def annualized_covariance_matrix(
    returns: FloatArray,
    periods_per_year: int = 252,
) -> FloatArray:
    """Calculate the annualized covariance matrix of asset returns."""
    validated_returns = _validate_returns(returns)

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be greater than zero.")

    covariance_matrix = np.cov(
        validated_returns,
        rowvar=False,
        ddof=1,
    )

    return np.atleast_2d(covariance_matrix) * periods_per_year


def portfolio_expected_return(
    weights: FloatArray,
    expected_returns: FloatArray,
) -> float:
    """Calculate the expected return of a portfolio."""
    normalized_expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    )

    if normalized_expected_returns.ndim != 1:
        raise ValueError(
            "Expected returns must be a one-dimensional array."
        )

    if not np.all(np.isfinite(normalized_expected_returns)):
        raise ValueError(
            "Expected returns must contain only finite values."
        )

    validated_weights = _validate_weights(
        weights,
        len(normalized_expected_returns),
    )

    return float(validated_weights @ normalized_expected_returns)


def portfolio_volatility(
    weights: FloatArray,
    covariance_matrix: FloatArray,
) -> float:
    """Calculate portfolio volatility from an annualized covariance matrix."""
    normalized_covariance = np.asarray(covariance_matrix, dtype=float)

    if normalized_covariance.ndim != 2:
        raise ValueError("Covariance matrix must be two-dimensional.")

    rows, columns = normalized_covariance.shape

    if rows == 0 or rows != columns:
        raise ValueError("Covariance matrix must be square.")

    if not np.all(np.isfinite(normalized_covariance)):
        raise ValueError(
            "Covariance matrix must contain only finite values."
        )

    validated_weights = _validate_weights(weights, rows)

    portfolio_variance = float(
        validated_weights @ normalized_covariance @ validated_weights
    )

    if portfolio_variance < -1e-12:
        raise ValueError("Portfolio variance cannot be negative.")

    portfolio_variance = max(portfolio_variance, 0.0)

    return float(np.sqrt(portfolio_variance))


def calculate_sharpe_ratio(
    expected_return: float,
    volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate the Sharpe ratio of a portfolio."""
    if not np.isfinite(expected_return):
        raise ValueError("Expected return must be finite.")

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    if not np.isfinite(volatility) or volatility <= 0:
        raise ValueError("Volatility must be greater than zero.")

    return float(
        (expected_return - risk_free_rate) / volatility
    )


def calculate_portfolio_metrics(
    weights: FloatArray,
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
    risk_free_rate: float = 0.0,
) -> PortfolioMetrics:
    """Calculate the principal annualized portfolio metrics."""
    expected_portfolio_return = portfolio_expected_return(
        weights,
        expected_returns,
    )

    volatility = portfolio_volatility(
        weights,
        covariance_matrix,
    )

    sharpe_ratio = calculate_sharpe_ratio(
        expected_return=expected_portfolio_return,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
    )

    return PortfolioMetrics(
        expected_return=expected_portfolio_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
    )


def main() -> None:
    """Run a simple portfolio metrics demonstration."""
    expected_returns = np.array([0.10, 0.14, 0.08])
    covariance_matrix = np.array(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0900, 0.0100],
            [0.0040, 0.0100, 0.0225],
        ]
    )
    weights = np.array([0.40, 0.35, 0.25])

    metrics = calculate_portfolio_metrics(
        weights=weights,
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.03,
    )

    print("Portfolio metrics")
    print(f"Expected annual return: {metrics.expected_return:.2%}")
    print(f"Annual volatility: {metrics.volatility:.2%}")
    print(f"Sharpe ratio: {metrics.sharpe_ratio:.4f}")


if __name__ == "__main__":
    main()