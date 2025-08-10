# Finance Toolkit

`finance_toolkit` is a comprehensive Python library designed to provide essential financial tools for analyzing financial instruments and portfolios. It includes functions to calculate key metrics such as beta, Sharpe ratio, maximum drawdown, risk contributions, Monte Carlo simulations, and more.

## Features

### Core Metrics
- **Beta**: Measures an asset's sensitivity to market movements (standard and advanced versions)
- **Theta**: Calculates the annualized average return of a financial instrument over a specified period
- **Sigma**: Calculates the annualized volatility of a financial instrument over a specified period
- **Max Drawdown**: Measures the largest loss from a peak to a trough in a financial instrument's historical performance
- **Jensen Alpha**: Measures an asset's performance relative to the market, adjusted for systematic risk using CAPM
- **Alpha**: Measures an asset's simple excess return relative to the market
- **Sharpe Ratio**: Evaluates the risk-adjusted return of a financial instrument
- **Calmar Ratio**: Evaluates the risk-adjusted return using maximum drawdown as the measure of risk
- **Historical VaR**: Calculates Value at Risk (VaR) based on historical performance at specified confidence levels

### Portfolio Analytics
- **Relative Volatility Contribution (RVC)**: Calculates each asset's relative contribution to total portfolio volatility
- **Marginal Contribution to Risk (MCR)**: Measures the marginal impact of each asset on portfolio risk

### Technical Analysis
- **Momentum**: Calculates momentum indicators with multiple methods (normal, ROC, log ROC)
- **Indexing**: Creates indexed time series based on percentage changes with optional weighting

### Simulation Tools
- **Multi-Asset GBM**: Geometric Brownian Motion simulation for correlated multi-asset portfolios using Monte Carlo methods

## Recent Updates (v0.1.7)

### Bug Fixes
- **Fixed FutureWarning**: Resolved pandas deprecation warning in `theta()` function by explicitly handling DataFrame.prod() operations
- **Improved Jensen Alpha**: Enhanced calculation with proper data alignment and beta computation
- **Enhanced Calmar Ratio**: Fixed DataFrame handling for multi-asset calculations

### Enhancements
- **Better Error Handling**: Improved robustness across all functions
- **Code Consistency**: Enhanced variable naming and documentation
- **Data Alignment**: Improved handling of misaligned time series data

## Installation

To install `finance_toolkit`, you can use pip:

```bash
pip install git+https://github.com/your_username/finance_toolkit.git
```

## Requirements

- pandas >= 1.3.0
- numpy >= 1.20.0
- typing (for Python < 3.8)

## Usage

### Basic Risk Metrics

```python
import pandas as pd
import numpy as np
from finance_toolkit import beta, theta, sigma, max_drawdown, sharpe, jensen_alpha

# Sample price data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
asset_prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)), index=dates)
market_prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.0008, 0.015, 100)), index=dates)

# Calculate beta
asset_df = pd.DataFrame({'asset': asset_prices})
beta_value = beta(history=asset_df, benchmark=market_prices)
print(f"Beta: {beta_value['asset']:.3f}")

# Calculate annualized return and volatility
annual_return = theta(history=asset_prices, timeperiod=252)
annual_volatility = sigma(history=asset_prices, timeperiod=252)
print(f"Annualized Return: {annual_return:.2%}")
print(f"Annualized Volatility: {annual_volatility:.2%}")

# Calculate maximum drawdown
max_dd = max_drawdown(history=asset_prices)
print(f"Max Drawdown: {max_dd:.2%}")

# Calculate Jensen's Alpha
risk_free_rate = 0.02
jensen_alpha_value = jensen_alpha(history=asset_prices, benchmark=market_prices, 
                                  riskfree=risk_free_rate, timeperiod=252)
print(f"Jensen Alpha: {jensen_alpha_value:.4f}")

# Calculate Sharpe ratio
sharpe_ratio = sharpe(history=asset_prices, riskfree=risk_free_rate, timeperiod=252)
print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
```

### Advanced Beta Analysis

```python
from finance_toolkit import beta_advanced

# Calculate conditional beta for different market conditions
upside_beta = beta_advanced(asset_prices, market_prices, way='+')
downside_beta = beta_advanced(asset_prices, market_prices, way='-')
overall_beta = beta_advanced(asset_prices, market_prices, way='all')

print(f"Upside Beta: {upside_beta:.3f}")
print(f"Downside Beta: {downside_beta:.3f}")
print(f"Overall Beta: {overall_beta:.3f}")
```

### Portfolio Risk Analysis

```python
from finance_toolkit import relative_volatility_contribution, marginal_contribution_to_risk

# Multi-asset portfolio
assets = pd.DataFrame({
    'Asset_A': asset_prices,
    'Asset_B': market_prices,
    'Asset_C': pd.Series(100 * np.cumprod(1 + np.random.normal(0.0005, 0.018, 100)), index=dates)
})

weights = pd.Series([0.4, 0.35, 0.25], index=['Asset_A', 'Asset_B', 'Asset_C'])

# Calculate risk contributions
vol_contrib = relative_volatility_contribution(history=assets, weights=weights)
mcr = marginal_contribution_to_risk(history=assets, weights=weights)

print("Relative Volatility Contributions:")
for asset, contrib in vol_contrib.items():
    print(f"  {asset}: {contrib:.2%}")

print("\nMarginal Contribution to Risk:")
for asset, risk in mcr.items():
    print(f"  {asset}: {risk:.4f}")
```

### Monte Carlo Portfolio Simulation

```python
from finance_toolkit import gbm_multi

# Portfolio simulation parameters
n_steps = 252  # One year of trading days
n_scenarios = 1000
mu = pd.Series([0.08, 0.06, 0.10], index=['Asset_A', 'Asset_B', 'Asset_C'])  # Expected returns
sigma_vec = pd.Series([0.20, 0.15, 0.25], index=['Asset_A', 'Asset_B', 'Asset_C'])  # Volatilities
correlation_matrix = pd.DataFrame([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]], 
                                  index=mu.index, columns=mu.index)
initial_value = 100000  # $100k portfolio
allocation = pd.Series([0.4, 0.35, 0.25], index=['Asset_A', 'Asset_B', 'Asset_C'])

# Run simulation
portfolio_paths = gbm_multi(n_step=n_steps, n_scenario=n_scenarios, mu=mu, 
                           sigma=sigma_vec, corr_matrix=correlation_matrix,
                           p_0=initial_value, allocation=allocation)

# Analyze results
final_values = portfolio_paths[-1, :]
print(f"Expected Portfolio Value after 1 year: ${final_values.mean():,.0f}")
print(f"Portfolio VaR (5%): ${np.percentile(final_values, 5):,.0f}")
print(f"Portfolio CVaR (5%): ${final_values[final_values <= np.percentile(final_values, 5)].mean():,.0f}")
```

### Technical Analysis

```python
from finance_toolkit import momentum, historical_var

# Calculate momentum indicators
price_momentum = momentum(asset_prices, period=20, differential='last', method='normal')
roc_momentum = momentum(asset_prices, period=20, differential='last', method='roc')

# Calculate Historical VaR
var_5 = historical_var(asset_prices, freq='D', conf_level=0.05)
var_1 = historical_var(asset_prices, freq='D', conf_level=0.01)

print(f"5% VaR: {var_5:.2%}")
print(f"1% VaR: {var_1:.2%}")
```

## Function Parameters

### Key Parameters Across Functions:
- `timeperiod`: Number of periods for annualization (default: 252 for daily data)
- `history`: Time series of prices or returns (pd.Series or pd.DataFrame)
- `benchmark`: Market or benchmark time series (pd.Series)
- `riskfree`: Risk-free rate for risk-adjusted metrics
- `weights`: Portfolio weights (pd.Series) for portfolio analytics

## Contributing

Contributions are welcome! If you would like to contribute to `FinanceToolkit`, please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Write tests for your new feature
4. Ensure all tests pass and code follows PEP 8
5. Commit your changes (`git commit -m 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.

## Contact

For questions, suggestions, or bug reports, please:
- Open an issue on GitHub
- Contact: [matteo.bernard@outlook.fr](mailto:matteo.bernard@outlook.fr)

## Acknowledgments

- Built with pandas and numpy
- Inspired by quantitative finance best practices
- Thanks to all contributors and users for feedback and improvements