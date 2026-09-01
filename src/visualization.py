"""Professional visualizations for portfolio analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import (
    calculate_simple_returns,
    download_adjusted_prices,
)
from src.optimization import (
    PortfolioOptimizationResult,
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
)
from src.portfolio_metrics import (
    annualized_covariance_matrix,
    annualized_expected_returns,
)
from src.simulation import (
    PortfolioSimulationResult,
    simulate_random_portfolios,
)


DEFAULT_FIGURES_DIRECTORY = Path("results/figures")


def _prepare_output_directory(
    output_directory: str | Path,
) -> Path:
    """Create and return the figures directory."""
    normalized_directory = Path(output_directory)

    normalized_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return normalized_directory


def plot_normalized_prices(
    prices: pd.DataFrame,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
) -> Path:
    """Plot the growth of one hundred invested in every asset."""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError(
            "Prices must be a non-empty pandas DataFrame."
        )

    normalized_prices = prices.divide(
        prices.iloc[0]
    ) * 100.0

    figures_directory = _prepare_output_directory(
        output_directory
    )

    output_path = (
        figures_directory / "normalized_prices.png"
    )

    figure, axis = plt.subplots(figsize=(14, 8))

    normalized_prices.plot(
        ax=axis,
        linewidth=2.0,
    )

    axis.set_title(
        "Growth of $100 Invested in Each Asset",
        fontsize=16,
        fontweight="bold",
    )
    axis.set_xlabel("Date")
    axis.set_ylabel("Portfolio value ($)")
    axis.axhline(
        y=100.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
    )
    axis.legend(
        title="Asset",
        frameon=True,
    )
    axis.grid(alpha=0.30)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def plot_correlation_matrix(
    returns: pd.DataFrame,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
) -> Path:
    """Plot the correlation matrix of asset returns."""
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise ValueError(
            "Returns must be a non-empty pandas DataFrame."
        )

    correlation_matrix = returns.corr()

    figures_directory = _prepare_output_directory(
        output_directory
    )

    output_path = (
        figures_directory / "correlation_matrix.png"
    )

    figure, axis = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        cbar_kws={
            "label": "Correlation",
            "shrink": 0.85,
        },
        ax=axis,
    )

    axis.set_title(
        "Correlation Matrix of Daily Returns",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    axis.set_xlabel("Asset")
    axis.set_ylabel("Asset")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def plot_portfolio_simulation(
    simulation: PortfolioSimulationResult,
    minimum_variance: PortfolioOptimizationResult,
    maximum_sharpe: PortfolioOptimizationResult,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
) -> Path:
    """Plot simulated portfolios and optimized solutions."""
    if len(simulation.weights) == 0:
        raise ValueError(
            "Simulation must contain at least one portfolio."
        )

    figures_directory = _prepare_output_directory(
        output_directory
    )

    output_path = (
        figures_directory / "portfolio_optimization.png"
    )

    figure, axis = plt.subplots(figsize=(14, 9))

    scatter = axis.scatter(
        simulation.volatilities,
        simulation.expected_returns,
        c=simulation.sharpe_ratios,
        cmap="viridis",
        s=12,
        alpha=0.55,
        edgecolors="none",
    )

    axis.scatter(
        minimum_variance.metrics.volatility,
        minimum_variance.metrics.expected_return,
        marker="*",
        s=450,
        color="red",
        edgecolor="black",
        linewidth=1.2,
        label="Minimum variance",
        zorder=5,
    )

    axis.scatter(
        maximum_sharpe.metrics.volatility,
        maximum_sharpe.metrics.expected_return,
        marker="*",
        s=450,
        color="gold",
        edgecolor="black",
        linewidth=1.2,
        label="Maximum Sharpe",
        zorder=5,
    )

    color_bar = figure.colorbar(
        scatter,
        ax=axis,
        pad=0.02,
    )
    color_bar.set_label(
        "Sharpe ratio",
        fontsize=11,
    )

    axis.set_title(
        "Portfolio Risk–Return Optimization",
        fontsize=17,
        fontweight="bold",
    )
    axis.set_xlabel(
        "Annualized volatility",
        fontsize=12,
    )
    axis.set_ylabel(
        "Expected annual return",
        fontsize=12,
    )

    axis.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, position: f"{value:.0%}"
        )
    )
    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, position: f"{value:.0%}"
        )
    )

    axis.legend(
        loc="best",
        frameon=True,
        fontsize=11,
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def create_all_visualizations(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    simulation: PortfolioSimulationResult,
    minimum_variance: PortfolioOptimizationResult,
    maximum_sharpe: PortfolioOptimizationResult,
    output_directory: str | Path = DEFAULT_FIGURES_DIRECTORY,
) -> list[Path]:
    """Generate and save every portfolio visualization."""
    return [
        plot_normalized_prices(
            prices,
            output_directory,
        ),
        plot_correlation_matrix(
            returns,
            output_directory,
        ),
        plot_portfolio_simulation(
            simulation,
            minimum_variance,
            maximum_sharpe,
            output_directory,
        ),
    ]


def main() -> None:
    """Run the complete historical portfolio analysis."""
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD"]

    prices = download_adjusted_prices(
        tickers=tickers,
        start_date="2021-01-01",
        end_date="2026-01-01",
    )

    returns = calculate_simple_returns(prices)

    expected_returns = annualized_expected_returns(
        returns.to_numpy(),
        periods_per_year=252,
    )

    covariance_matrix = annualized_covariance_matrix(
        returns.to_numpy(),
        periods_per_year=252,
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

    simulation = simulate_random_portfolios(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        simulations=100_000,
        risk_free_rate=risk_free_rate,
        seed=42,
    )

    generated_figures = create_all_visualizations(
        prices=prices,
        returns=returns,
        simulation=simulation,
        minimum_variance=minimum_variance,
        maximum_sharpe=maximum_sharpe,
    )

    print("Portfolio visualizations generated successfully")

    for figure_path in generated_figures:
        print(f"- {figure_path}")

    print("\nMinimum-variance portfolio")

    for ticker, weight in zip(
        tickers,
        minimum_variance.weights,
        strict=True,
    ):
        print(f"{ticker}: {weight:.2%}")

    print(
        "Expected return: "
        f"{minimum_variance.metrics.expected_return:.2%}"
    )
    print(
        "Volatility: "
        f"{minimum_variance.metrics.volatility:.2%}"
    )
    print(
        "Sharpe ratio: "
        f"{minimum_variance.metrics.sharpe_ratio:.4f}"
    )

    print("\nMaximum-Sharpe portfolio")

    for ticker, weight in zip(
        tickers,
        maximum_sharpe.weights,
        strict=True,
    ):
        print(f"{ticker}: {weight:.2%}")

    print(
        "Expected return: "
        f"{maximum_sharpe.metrics.expected_return:.2%}"
    )
    print(
        "Volatility: "
        f"{maximum_sharpe.metrics.volatility:.2%}"
    )
    print(
        "Sharpe ratio: "
        f"{maximum_sharpe.metrics.sharpe_ratio:.4f}"
    )


if __name__ == "__main__":
    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )

    main()