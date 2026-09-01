"""Portfolio optimization using Modern Portfolio Theory."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.portfolio_metrics import (
    PortfolioMetrics,
    calculate_portfolio_metrics,
    portfolio_volatility,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Result returned by a portfolio optimization process."""

    weights: FloatArray
    metrics: PortfolioMetrics


def _validate_optimization_inputs(
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Validate expected returns and the covariance matrix."""
    normalized_returns = np.asarray(
        expected_returns,
        dtype=float,
    )
    normalized_covariance = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    if normalized_returns.ndim != 1:
        raise ValueError(
            "Expected returns must be a one-dimensional array."
        )

    if len(normalized_returns) == 0:
        raise ValueError(
            "Expected returns must contain at least one asset."
        )

    if not np.all(np.isfinite(normalized_returns)):
        raise ValueError(
            "Expected returns must contain only finite values."
        )

    if normalized_covariance.ndim != 2:
        raise ValueError(
            "Covariance matrix must be two-dimensional."
        )

    rows, columns = normalized_covariance.shape

    if rows != columns:
        raise ValueError("Covariance matrix must be square.")

    if rows != len(normalized_returns):
        raise ValueError(
            "Covariance matrix dimensions must match "
            "the number of assets."
        )

    if not np.all(np.isfinite(normalized_covariance)):
        raise ValueError(
            "Covariance matrix must contain only finite values."
        )

    if not np.allclose(
        normalized_covariance,
        normalized_covariance.T,
    ):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(normalized_covariance)

    if np.min(eigenvalues) < -1e-10:
        raise ValueError(
            "Covariance matrix must be positive semidefinite."
        )

    if np.allclose(normalized_covariance, 0.0):
        raise ValueError(
            "Covariance matrix must contain positive risk."
        )

    return normalized_returns, normalized_covariance


def _normalize_optimized_weights(
    weights: FloatArray,
) -> FloatArray:
    """Remove numerical noise and ensure weights sum exactly to one."""
    normalized_weights = np.clip(
        np.asarray(weights, dtype=float),
        0.0,
        1.0,
    )

    total_weight = normalized_weights.sum()

    if total_weight <= 0:
        raise RuntimeError(
            "Optimization produced invalid portfolio weights."
        )

    return normalized_weights / total_weight


def _optimize_portfolio(
    objective_function: Callable[[FloatArray], float],
    number_of_assets: int,
) -> FloatArray:
    """Run a long-only constrained portfolio optimization."""
    initial_weights = np.full(
        number_of_assets,
        1.0 / number_of_assets,
    )

    bounds = tuple(
        (0.0, 1.0)
        for _ in range(number_of_assets)
    )

    constraints = {
        "type": "eq",
        "fun": lambda weights: float(
            np.sum(weights) - 1.0
        ),
    }

    result = minimize(
        fun=objective_function,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-12,
            "maxiter": 1_000,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Portfolio optimization failed: {result.message}"
        )

    return _normalize_optimized_weights(result.x)


def minimum_variance_portfolio(
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
    risk_free_rate: float = 0.0,
) -> PortfolioOptimizationResult:
    """Find the long-only portfolio with minimum volatility."""
    normalized_returns, normalized_covariance = (
        _validate_optimization_inputs(
            expected_returns,
            covariance_matrix,
        )
    )

    def objective(weights: FloatArray) -> float:
        return portfolio_volatility(
            weights,
            normalized_covariance,
        )

    optimized_weights = _optimize_portfolio(
        objective_function=objective,
        number_of_assets=len(normalized_returns),
    )

    metrics = calculate_portfolio_metrics(
        weights=optimized_weights,
        expected_returns=normalized_returns,
        covariance_matrix=normalized_covariance,
        risk_free_rate=risk_free_rate,
    )

    return PortfolioOptimizationResult(
        weights=optimized_weights,
        metrics=metrics,
    )


def maximum_sharpe_portfolio(
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
    risk_free_rate: float = 0.0,
) -> PortfolioOptimizationResult:
    """Find the long-only portfolio with maximum Sharpe ratio."""
    normalized_returns, normalized_covariance = (
        _validate_optimization_inputs(
            expected_returns,
            covariance_matrix,
        )
    )

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    def objective(weights: FloatArray) -> float:
        metrics = calculate_portfolio_metrics(
            weights=weights,
            expected_returns=normalized_returns,
            covariance_matrix=normalized_covariance,
            risk_free_rate=risk_free_rate,
        )

        return -metrics.sharpe_ratio

    optimized_weights = _optimize_portfolio(
        objective_function=objective,
        number_of_assets=len(normalized_returns),
    )

    metrics = calculate_portfolio_metrics(
        weights=optimized_weights,
        expected_returns=normalized_returns,
        covariance_matrix=normalized_covariance,
        risk_free_rate=risk_free_rate,
    )

    return PortfolioOptimizationResult(
        weights=optimized_weights,
        metrics=metrics,
    )


def _print_result(
    title: str,
    result: PortfolioOptimizationResult,
) -> None:
    """Print an optimization result."""
    print(f"\n{title}")

    for index, weight in enumerate(
        result.weights,
        start=1,
    ):
        print(f"Asset {index}: {weight:.2%}")

    print(
        "Expected annual return: "
        f"{result.metrics.expected_return:.2%}"
    )
    print(
        "Annual volatility: "
        f"{result.metrics.volatility:.2%}"
    )
    print(
        "Sharpe ratio: "
        f"{result.metrics.sharpe_ratio:.4f}"
    )


def main() -> None:
    """Run a portfolio optimization demonstration."""
    expected_returns = np.array([0.10, 0.14, 0.08])

    covariance_matrix = np.array(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0900, 0.0100],
            [0.0040, 0.0100, 0.0225],
        ]
    )

    risk_free_rate = 0.03

    minimum_variance = minimum_variance_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )

    maximum_sharpe = maximum_sharpe_portfolio(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=risk_free_rate,
    )

    print("Modern Portfolio Theory optimization")

    _print_result(
        "Minimum-variance portfolio",
        minimum_variance,
    )

    _print_result(
        "Maximum-Sharpe portfolio",
        maximum_sharpe,
    )


if __name__ == "__main__":
    main()