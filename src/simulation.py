"""Random portfolio simulation for Modern Portfolio Theory."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PortfolioSimulationResult:
    """Collection of simulated portfolio weights and metrics."""

    weights: FloatArray
    expected_returns: FloatArray
    volatilities: FloatArray
    sharpe_ratios: FloatArray

    @property
    def minimum_volatility_index(self) -> int:
        """Return the index of the least volatile portfolio."""
        return int(np.argmin(self.volatilities))

    @property
    def maximum_sharpe_index(self) -> int:
        """Return the index of the portfolio with maximum Sharpe."""
        return int(np.argmax(self.sharpe_ratios))


def _validate_simulation_inputs(
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
    simulations: int,
    risk_free_rate: float,
    seed: int | None,
) -> tuple[FloatArray, FloatArray]:
    """Validate portfolio simulation inputs."""
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

    if rows == 0 or rows != columns:
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

    if isinstance(simulations, bool) or not isinstance(
        simulations,
        (int, np.integer),
    ):
        raise TypeError("Simulations must be an integer.")

    if simulations <= 0:
        raise ValueError(
            "Simulations must be greater than zero."
        )

    if not np.isfinite(risk_free_rate):
        raise ValueError("Risk-free rate must be finite.")

    if seed is not None and (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("Seed must be an integer or None.")

    return normalized_returns, normalized_covariance


def simulate_random_portfolios(
    expected_returns: FloatArray,
    covariance_matrix: FloatArray,
    simulations: int = 100_000,
    risk_free_rate: float = 0.0,
    seed: int | None = None,
) -> PortfolioSimulationResult:
    """Generate long-only portfolios with random weights."""
    normalized_returns, normalized_covariance = (
        _validate_simulation_inputs(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            simulations=simulations,
            risk_free_rate=risk_free_rate,
            seed=seed,
        )
    )

    number_of_assets = len(normalized_returns)
    random_generator = np.random.default_rng(seed)

    weights = random_generator.dirichlet(
        alpha=np.ones(number_of_assets),
        size=simulations,
    )

    portfolio_returns = weights @ normalized_returns

    portfolio_variances = np.einsum(
        "ij,jk,ik->i",
        weights,
        normalized_covariance,
        weights,
    )

    portfolio_variances = np.maximum(
        portfolio_variances,
        0.0,
    )

    portfolio_volatilities = np.sqrt(
        portfolio_variances
    )

    if np.any(portfolio_volatilities <= 0):
        raise ValueError(
            "Simulated portfolio volatility must be positive."
        )

    sharpe_ratios = (
        portfolio_returns - risk_free_rate
    ) / portfolio_volatilities

    return PortfolioSimulationResult(
        weights=weights,
        expected_returns=portfolio_returns,
        volatilities=portfolio_volatilities,
        sharpe_ratios=sharpe_ratios,
    )


def _print_portfolio(
    title: str,
    result: PortfolioSimulationResult,
    index: int,
) -> None:
    """Print one portfolio selected from a simulation."""
    print(f"\n{title}")

    for asset_number, weight in enumerate(
        result.weights[index],
        start=1,
    ):
        print(f"Asset {asset_number}: {weight:.2%}")

    print(
        "Expected annual return: "
        f"{result.expected_returns[index]:.2%}"
    )
    print(
        "Annual volatility: "
        f"{result.volatilities[index]:.2%}"
    )
    print(
        "Sharpe ratio: "
        f"{result.sharpe_ratios[index]:.4f}"
    )


def main() -> None:
    """Run a random portfolio simulation demonstration."""
    expected_returns = np.array([0.10, 0.14, 0.08])

    covariance_matrix = np.array(
        [
            [0.0400, 0.0060, 0.0040],
            [0.0060, 0.0900, 0.0100],
            [0.0040, 0.0100, 0.0225],
        ]
    )

    result = simulate_random_portfolios(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        simulations=100_000,
        risk_free_rate=0.03,
        seed=42,
    )

    print("Random portfolio simulation")
    print(
        f"Number of portfolios: "
        f"{len(result.weights):,}"
    )

    _print_portfolio(
        title="Simulated minimum-volatility portfolio",
        result=result,
        index=result.minimum_volatility_index,
    )

    _print_portfolio(
        title="Simulated maximum-Sharpe portfolio",
        result=result,
        index=result.maximum_sharpe_index,
    )


if __name__ == "__main__":
    main()