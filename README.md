# Quantitative Portfolio Optimization Engine

A professional quantitative-finance project for portfolio construction using Modern Portfolio Theory, historical market data, numerical optimization, and Monte Carlo-style random portfolio simulation.

The engine identifies:

- The minimum-variance portfolio
- The maximum-Sharpe portfolio
- The risk–return opportunity set
- Diversification opportunities through correlation analysis

> This project is intended for educational and analytical purposes. It does not constitute financial advice.

---

## English

### Project Overview

This project implements a complete portfolio optimization workflow in Python.

It downloads adjusted historical prices, calculates asset returns and risk estimates, simulates thousands of possible portfolios, and uses constrained numerical optimization to identify efficient allocations.

The reference analysis uses:

| Ticker | Market exposure |
|---|---|
| SPY | Large-cap U.S. equities |
| QQQ | Nasdaq and technology-oriented equities |
| IWM | U.S. small-cap equities |
| TLT | Long-term U.S. Treasury bonds |
| GLD | Gold |

### Main Features

- Historical adjusted-price downloading
- Market-data validation and cleaning
- Daily simple-return calculation
- Annualized expected returns
- Annualized covariance matrices
- Asset correlation analysis
- Portfolio return calculation
- Portfolio volatility calculation
- Sharpe ratio calculation
- Minimum-variance optimization
- Maximum-Sharpe optimization
- Long-only allocation constraints
- Random portfolio simulation
- Reproducible random seeds
- Professional financial visualizations
- Complete analytical notebook
- Mathematical methodology
- Automated test suite

### Mathematical Foundation

For a portfolio-weight vector \(w\), expected-return vector \(\mu\), and covariance matrix \(\Sigma\):

#### Expected portfolio return

\[
E(R_p)=w^T\mu
\]

#### Portfolio volatility

\[
\sigma_p=\sqrt{w^T\Sigma w}
\]

#### Sharpe ratio

\[
S_p=\frac{E(R_p)-r_f}{\sigma_p}
\]

The optimization uses the following constraints:

\[
\sum_{i=1}^{n}w_i=1
\]

\[
0\leq w_i\leq1
\]

These constraints represent a fully invested, long-only portfolio without leverage.

### Minimum-Variance Portfolio

The minimum-variance portfolio solves:

\[
\min_w \quad w^T\Sigma w
\]

subject to the portfolio constraints.

Its objective is to find the asset combination with the lowest estimated annual volatility.

### Maximum-Sharpe Portfolio

The maximum-Sharpe portfolio solves:

\[
\max_w
\quad
\frac{w^T\mu-r_f}
{\sqrt{w^T\Sigma w}}
\]

Its objective is to find the portfolio with the highest estimated excess return per unit of risk.

### Random Portfolio Simulation

The engine generates 100,000 valid long-only portfolios using weights sampled from a Dirichlet distribution.

For every simulated portfolio, it calculates:

- Expected annual return
- Annual volatility
- Sharpe ratio

The resulting risk–return cloud approximates the investment opportunity set and visually identifies its efficient region.

### Project Structure

```text
portfolio-optimization-engine/
├── docs/
│   └── methodology.md
├── notebooks/
│   └── portfolio_optimization_analysis.ipynb
├── results/
│   └── figures/
│       ├── correlation_matrix.png
│       ├── normalized_prices.png
│       └── portfolio_optimization.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── optimization.py
│   ├── portfolio_metrics.py
│   ├── simulation.py
│   └── visualization.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_optimization.py
│   ├── test_portfolio_metrics.py
│   ├── test_simulation.py
│   └── test_visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Installation

Clone the repository:

```powershell
git clone https://github.com/acarrillofintech/portfolio-optimization-engine.git
cd portfolio-optimization-engine
```

Create a virtual environment:

```powershell
py -m venv .venv
```

Activate it in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python -m pip check
```

### Usage

Run the portfolio metrics demonstration:

```powershell
python -m src.portfolio_metrics
```

Run the numerical optimization:

```powershell
python -m src.optimization
```

Run the random portfolio simulation:

```powershell
python -m src.simulation
```

Download and process historical data:

```powershell
python -m src.data_loader
```

Generate all visualizations and the real-data analysis:

```powershell
python -m src.visualization
```

### Notebook

Open:

```text
notebooks/portfolio_optimization_analysis.ipynb
```

Select the project virtual environment as the notebook kernel and run all cells.

The notebook includes:

1. Portfolio configuration
2. Historical adjusted prices
3. Normalized asset performance
4. Annualized risk and return
5. Correlation analysis
6. Portfolio optimization
7. Random portfolio simulation
8. Financial interpretation
9. Conclusions

### Testing

Run the complete automated test suite:

```powershell
python -m pytest -v
```

Current validation:

```text
72 passed
```

The tests cover:

- Financial formulas
- Annualization
- Covariance calculations
- Weight constraints
- Optimization results
- Simulation reproducibility
- Market-data validation
- Invalid inputs
- Visualization generation

Network requests are replaced with controlled test data when testing the data loader. This keeps the suite fast and reproducible.

### Generated Figures

#### Normalized Historical Prices

Compares the historical evolution of USD 100 invested in every selected asset.

#### Correlation Matrix

Displays the relationships between daily asset returns and helps identify diversification opportunities.

#### Portfolio Optimization

Displays:

- Simulated portfolios
- Expected annual returns
- Annualized volatility
- Sharpe ratios
- Minimum-variance portfolio
- Maximum-Sharpe portfolio

### Technologies

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- seaborn
- yfinance
- Jupyter
- pytest

### Data Source

Historical adjusted prices are obtained through the open-source `yfinance` package.

Yahoo Finance data is intended for personal, educational, and research use. Users should review the corresponding provider terms before using the data for other purposes.

### Limitations

The model:

- Depends on historical estimates
- Assumes historical relationships provide useful information
- Uses volatility as the principal risk measure
- Ignores taxes and transaction costs
- Does not model market impact
- Does not permit short selling or leverage
- Does not guarantee future performance

### Disclaimer

This repository is an educational quantitative-finance project.

It does not provide financial, investment, legal, or tax advice. The results should not be interpreted as recommendations to purchase or sell securities.

---

## Español

### Descripción del proyecto

Este proyecto implementa un proceso completo de optimización cuantitativa de portafolios utilizando Python.

El motor descarga precios históricos ajustados, calcula rendimientos y medidas de riesgo, simula miles de portafolios y aplica optimización numérica restringida para encontrar asignaciones eficientes.

### Funcionalidades principales

- Descarga de precios históricos ajustados
- Validación y limpieza de datos financieros
- Cálculo de rendimientos diarios
- Rendimientos esperados anualizados
- Matrices de covarianza anualizadas
- Análisis de correlaciones
- Rendimiento esperado del portafolio
- Volatilidad del portafolio
- Ratio de Sharpe
- Portafolio de mínima varianza
- Portafolio de máximo Sharpe
- Restricciones de posiciones largas
- Simulación de portafolios aleatorios
- Semillas reproducibles
- Visualizaciones financieras
- Notebook analítico
- Metodología matemática
- Pruebas automatizadas

### Objetivo financiero

El proyecto responde dos preguntas principales:

1. ¿Cómo distribuir el capital para reducir la volatilidad?
2. ¿Cómo distribuir el capital para obtener la mayor recompensa estimada por unidad de riesgo?

### Portafolio de mínima varianza

Busca la combinación de activos con la menor volatilidad anual estimada.

Este portafolio prioriza la reducción del riesgo total.

### Portafolio de máximo Sharpe

Busca la combinación con el mayor rendimiento excedente esperado por unidad de volatilidad.

Este portafolio prioriza la eficiencia entre rendimiento y riesgo.

### Diversificación

La diversificación depende de las relaciones entre los activos.

Cuando los activos no se mueven exactamente de la misma manera, su combinación puede reducir el riesgo total del portafolio.

Por esta razón, el motor utiliza la matriz de covarianza completa y no solamente la volatilidad individual de cada activo.

### Instalación

Clona el repositorio:

```powershell
git clone https://github.com/acarrillofintech/portfolio-optimization-engine.git
cd portfolio-optimization-engine
```

Crea y activa el entorno virtual:

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Instala las dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### Ejecución

Ejecuta el análisis completo:

```powershell
python -m src.visualization
```

Ejecuta todas las pruebas:

```powershell
python -m pytest -v
```

Resultado esperado:

```text
72 passed
```

### Documentación

La metodología matemática completa se encuentra en:

```text
docs/methodology.md
```

El análisis interactivo se encuentra en:

```text
notebooks/portfolio_optimization_analysis.ipynb
```

### Resultados visuales

El proyecto genera:

```text
results/figures/normalized_prices.png
results/figures/correlation_matrix.png
results/figures/portfolio_optimization.png
```

### Advertencia

Este proyecto fue desarrollado con fines educativos, analíticos y de demostración profesional.

Los resultados no constituyen asesoría financiera ni garantizan rendimientos futuros.

---

## Author

**Alex Carrillo**

Quantitative finance, financial mathematics, Python, risk analytics, and software engineering.

## License

This project is distributed under the MIT License.