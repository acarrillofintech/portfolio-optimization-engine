"""Tests for random portfolio simulation."""

import numpy as np
import pytest

from src.simulation import (
    PortfolioSimulationResult,
    simulate_random_portfolios,
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


@pytest.fixture
def simulation_result(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> PortfolioSimulationResult:
    """Generate a reproducible portfolio simulation."""
    return simulate_random_portfolios(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        simulations=5_000,
        risk_free_rate=0.03,
        seed=42,
    )


def test_simulation_returns_expected_dimensions(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Simulation arrays should have consistent dimensions."""
    assert simulation_result.weights.shape == (5_000, 3)
    assert simulation_result.expected_returns.shape == (5_000,)
    assert simulation_result.volatilities.shape == (5_000,)
    assert simulation_result.sharpe_ratios.shape == (5_000,)


def test_simulated_weights_sum_to_one(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Every simulated portfolio must invest all capital."""
    weight_sums = simulation_result.weights.sum(axis=1)

    np.testing.assert_allclose(weight_sums, 1.0)


def test_simulated_weights_are_long_only(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Random portfolios cannot contain negative weights."""
    assert np.all(simulation_result.weights >= 0.0)
    assert np.all(simulation_result.weights <= 1.0)


def test_simulated_metrics_are_finite(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """All simulated portfolio metrics must be finite."""
    assert np.all(
        np.isfinite(simulation_result.expected_returns)
    )
    assert np.all(
        np.isfinite(simulation_result.volatilities)
    )
    assert np.all(
        np.isfinite(simulation_result.sharpe_ratios)
    )


def test_simulation_is_reproducible(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """The same seed should generate identical results."""
    first_result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=1_000,
        risk_free_rate=0.03,
        seed=123,
    )

    second_result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=1_000,
        risk_free_rate=0.03,
        seed=123,
    )

    np.testing.assert_array_equal(
        first_result.weights,
        second_result.weights,
    )
    np.testing.assert_array_equal(
        first_result.expected_returns,
        second_result.expected_returns,
    )


def test_different_seeds_produce_different_weights(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Different seeds should produce different portfolios."""
    first_result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=100,
        seed=1,
    )

    second_result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=100,
        seed=2,
    )

    assert not np.array_equal(
        first_result.weights,
        second_result.weights,
    )


def test_result_is_simulation_dataclass(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Simulation should return a structured result."""
    assert isinstance(
        simulation_result,
        PortfolioSimulationResult,
    )


def test_minimum_volatility_index_is_correct(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Minimum-volatility property should locate the minimum."""
    expected_index = int(
        np.argmin(simulation_result.volatilities)
    )

    assert (
        simulation_result.minimum_volatility_index
        == expected_index
    )


def test_maximum_sharpe_index_is_correct(
    simulation_result: PortfolioSimulationResult,
) -> None:
    """Maximum-Sharpe property should locate the maximum."""
    expected_index = int(
        np.argmax(simulation_result.sharpe_ratios)
    )

    assert (
        simulation_result.maximum_sharpe_index
        == expected_index
    )


def test_simulated_minimum_volatility_is_near_optimum(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Large simulation should approach minimum volatility."""
    result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=50_000,
        risk_free_rate=0.03,
        seed=42,
    )

    simulated_volatility = result.volatilities[
        result.minimum_volatility_index
    ]

    assert simulated_volatility == pytest.approx(
        0.124789,
        abs=5e-4,
    )


def test_simulated_maximum_sharpe_is_near_optimum(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Large simulation should approach maximum Sharpe."""
    result = simulate_random_portfolios(
        expected_returns,
        covariance_matrix,
        simulations=50_000,
        risk_free_rate=0.03,
        seed=42,
    )

    simulated_sharpe = result.sharpe_ratios[
        result.maximum_sharpe_index
    ]

    assert simulated_sharpe == pytest.approx(
        0.532599,
        abs=5e-4,
    )


@pytest.mark.parametrize("simulations", [0, -1])
def test_invalid_simulation_count_raises_value_error(
    simulations: int,
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Simulation count must be positive."""
    with pytest.raises(
        ValueError,
        match="Simulations must be greater than zero",
    ):
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=simulations,
        )


def test_non_integer_simulations_raise_type_error(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Simulation count must be an integer."""
    with pytest.raises(
        TypeError,
        match="Simulations must be an integer",
    ):
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100.5,  # type: ignore[arg-type]
        )


def test_invalid_seed_raises_type_error(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Random seed must be an integer or None."""
    with pytest.raises(
        TypeError,
        match="Seed must be an integer or None",
    ):
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100,
            seed=42.5,  # type: ignore[arg-type]
        )


def test_non_finite_risk_free_rate_raises_value_error(
    expected_returns: np.ndarray,
    covariance_matrix: np.ndarray,
) -> None:
    """Risk-free rate must contain a finite value."""
    with pytest.raises(
        ValueError,
        match="Risk-free rate must be finite",
    ):
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100,
            risk_free_rate=np.nan,
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
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100,
        )


def test_non_symmetric_covariance_raises_value_error(
    expected_returns: np.ndarray,
) -> None:
    """Covariance matrix must be symmetric."""
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
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100,
        )


def test_zero_covariance_matrix_raises_value_error(
    expected_returns: np.ndarray,
) -> None:
    """Simulation requires assets with measurable risk."""
    covariance_matrix = np.zeros((3, 3))

    with pytest.raises(
        ValueError,
        match="Covariance matrix must contain positive risk",
    ):
        simulate_random_portfolios(
            expected_returns,
            covariance_matrix,
            simulations=100,
        )