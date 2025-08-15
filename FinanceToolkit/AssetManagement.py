import pandas as pd
import numpy as np
from typing import Literal

def beta(asset: pd.Series, benchmark: pd.Series, way: Literal['+', '-', 'all']='all') -> float:
    """
    #### Description:
    Beta is a measure of a financial instrument's sensitivity to benchmark movements. A beta of 1 indicates the history tends
    to move in line with the benchmark, a beta greater than 1 suggests higher volatility, and a beta less than 1 indicates
    lower volatility compared to the benchmark.

    #### Parameters:
    - asset (pd.Series): Time series data representing the returns of the financial instrument.
    - benchmark (pd.Series): Time series data representing the returns of the benchmark.
    - way (Literal['+', '-', 'all']): Specifies which type of data points should be considered for the beta calculation:
        - '+' (positive): Only considers periods where the history's returns are positive. This is useful for measuring
        the beta when the history is performing well.
        - '-' (negative): Only considers periods where the history's returns are negative. This is useful for measuring
        the beta when the history is underperforming.
        - 'all': Considers all periods without any filtering, giving the traditional beta measurement.

    #### Returns:
    - float: Beta coefficient, which measures the history's sensitivity to benchmark movements based on the specified filter.
    """
    # Combine history and benchmark returns, calculate covariance and benchmark variance
    df = pd.concat([asset, benchmark], axis=1).dropna().pct_change().dropna()

    # Filter data based on the 'way' parameter, focusing on history variations
    if way == '+':
        df = df[df.iloc[:, 0] > 0]  # Filter where history returns are positive
    elif way == '-':
        df = df[df.iloc[:, 0] < 0]  # Filter where history returns are negative
    elif way == 'all':
        pass  # No filtering needed

    # Calculate covariance and benchmark variance
    covariance = df.cov().iloc[1, 0]
    benchmark_variance = df.iloc[:, 1].var()

    # Calculate beta as the ratio of covariance to benchmark variance
    return covariance / benchmark_variance


def theta(asset: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the average return of a financial instrument over a specified time period.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average return. Default is 252.

    #### Returns:
    - float: Average return over the specified time period.
    """
    returns = asset.pct_change().dropna()
    return (1 + returns).prod() ** (timeperiod / len(returns)) - 1


def sigma(asset: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the average volatility of a financial instrument over a specified time period.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average volatility. Default is 252.

    #### Returns:
    - float: Average volatility over the specified time period.
    """
    returns = asset.pct_change().dropna()
    return returns.std() * np.sqrt(timeperiod)


def max_drawdown(asset: pd.Series) -> float:
    """
    #### Description:
    Max Drawdown is a measure of the largest loss from a peak to a trough in a financial instrument's historical performance.
    It is calculated as the percentage decline relative to the highest cumulative value.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical prices or returns of the financial instrument.

    #### Returns:
    - float: Maximum drawdown, representing the largest percentage decline from a peak to a trough.
    """
    # Calculate the cumulative percentage change in price or returns
    index = 100 * (1 + asset.pct_change()).cumprod()

    # Identify the peaks and calculate drawdowns
    peaks = index.cummax()
    drawdowns = (index - peaks) / peaks

    # Return the maximum drawdown
    return drawdowns.min()


def jensen_alpha(asset: pd.Series, benchmark: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Alpha measures the history's performance relative to the benchmark. A positive alpha indicates that the history has outperformed
    the benchmark, while a negative alpha suggests underperformance.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical performance of the history.
    - benchmark (pd.Series): Time series data representing the historical performance of the benchmark.
    - riskfree (float): Risk-free rate of return.
    - timeperiod (int, optional): Time period for annualization. Default is 252.

    #### Returns:
    - float: Jensen's Alpha, representing the excess return of the history over the benchmark adjusted for systematic risk.
    """
    asset = asset.dropna()
    benchmark = benchmark.dropna()

    # Align dates
    asset = asset.align(benchmark, join='inner')[0]

    # Calculate returns
    asset_returns = (asset.iloc[-1] - asset.iloc[0]) / asset.iloc[0]
    benchmark_returns = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]
    
    # Calculate beta using the beta function
    history_beta = beta(asset, benchmark)

    # Calculate Jensen Alpha
    return asset_returns - (riskfree + history_beta * (benchmark_returns - riskfree))


def alpha(asset: pd.Series, benchmark: pd.Series) -> float:
    """
    #### Description:
    Calculate the alpha of a financial instrument relative to a benchmark. 
    A positive alpha indicates that the financial instrument has outperformed 
    the benchmark, while a negative alpha suggests underperformance.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical performance of the history.
    - benchmark (pd.Series): Time series data representing the historical performance of the benchmark.

    #### Returns:
    - float: Alpha, representing the excess return of the history over the benchmark.
    """
    asset = asset.dropna()
    benchmark = benchmark.dropna()

    asset_returns = (asset[-1] - asset[0]) / asset[0]
    benchmark_returns = (benchmark[-1] - benchmark[0]) / benchmark[0]
    return asset_returns - benchmark_returns


def omega(asset: pd.Series, threshold: float = 0.0) -> float:
    """
    #### Description:
    Calculate the Omega ratio of a financial instrument over the given history.
    The Omega ratio is the probability-weighted ratio of returns above a threshold
    to returns below that threshold.

    #### Parameters:
    - history (pd.Series): Time series data representing historical prices or returns.
    - threshold (float, optional): Minimum acceptable return (MAR). Default is 0.0.

    #### Returns:
    - float: Omega ratio value.
    """
    returns = asset.dropna().pct_change().dropna()

    # Excess returns above and below the threshold
    gains = (returns - threshold).clip(lower=0).sum()
    losses = (threshold - returns).clip(lower=0).sum()

    # Avoid division by zero
    if losses == 0:
        return float('inf')

    return gains / losses


def tracking_error(asset: pd.Series, benchmark: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the tracking error between a financial instrument and its benchmark over a specified time period.
    Tracking error measures the standard deviation of the difference in returns.

    #### Parameters:
    - asset (pd.Series): Historical prices or returns of the portfolio.
    - benchmark (pd.Series): Historical prices or returns of the benchmark.
    - timeperiod (int, optional): Number of periods to annualize the tracking error. Default is 252.

    #### Returns:
    - float: Annualized tracking error.
    """
    asset_returns = asset.dropna().pct_change().dropna()
    benchmark_returns = benchmark.dropna().pct_change().dropna()

    # Align dates
    returns_diff = asset_returns.align(benchmark_returns, join='inner')[0] - benchmark_returns

    return returns_diff.std() * (timeperiod ** 0.5)


def information_ratio(asset: pd.Series, benchmark: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Information Ratio between a financial instrument and its benchmark.
    The Information Ratio measures the active return per unit of active risk (tracking error).

    #### Parameters:
    - asset (pd.Series): Historical prices or returns of the asset.
    - benchmark (pd.Series): Historical prices or returns of the benchmark.
    - timeperiod (int, optional): Number of periods to annualize the return. Default is 252.

    #### Returns:
    - float: Information Ratio.
    """
    asset_returns = asset.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    # Align dates
    asset_returns, benchmark_returns = asset_returns.align(benchmark_returns, join='inner')

    active_return = asset_returns - benchmark_returns
    tracking_err = active_return.std() * (timeperiod ** 0.5)

    if tracking_err == 0:
        return 1.0

    annualized_active_return = active_return.mean() * timeperiod
    return annualized_active_return / tracking_err


def sharpe(asset: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Sharpe Ratio for a financial instrument based on its historical performance.
    The Sharpe Ratio helps assess the investment's return per unit of risk taken.

    #### Parameters:
    - asset (pd.Series): Historical price or return data of the asset.
    - riskfree (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = theta(asset=asset, timeperiod=timeperiod)
    volatility = sigma(asset=asset, timeperiod=timeperiod)
    
    # Calculate Sharpe Ratio using the formula
    return (returns - riskfree) / volatility


def calmar(asset: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Calmar Ratio for a given financial instrument based on its historical performance.
    The Calmar ratio uses a financial instrument's maximum drawdown as its sole measure of risk

    #### Parameters:
    - asset (pd.Series): Historical price or return data of the financial instrument.
    - riskfree (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Calmar Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and max drawdown using helper functions
    returns = theta(asset=asset, timeperiod=timeperiod)
    maxdrawdown = max_drawdown(asset=asset)
    
    if abs(maxdrawdown) != 0:
        # Calculate Calmar Ratio using the formula
        return (returns - riskfree) / abs(maxdrawdown)
    else:
        return 1.0


def sortino(asset: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Sortino Ratio for a financial instrument based on its historical performance.
    The Sortino Ratio measures the return per unit of downside risk, focusing only on negative volatility.

    #### Parameters:
    - asset (pd.Series): Historical price or return data of the asset.
    - riskfree (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and downside deviation. Default is 252.

    #### Returns:
    - float: The Sortino Ratio, a measure of risk-adjusted performance considering only downside risk.
    """
    # Calculate average return
    returns = theta(asset=asset, timeperiod=timeperiod)

    # Calculate downside deviation (only negative returns relative to risk-free rate)
    daily_returns = asset.pct_change().dropna()
    downside_returns = daily_returns[daily_returns < riskfree / timeperiod]
    downside_deviation = ( ( (downside_returns - riskfree / timeperiod) ** 2 ).mean() ** 0.5 ) * (timeperiod ** 0.5)

    # Avoid division by zero
    if downside_deviation == 0:
        return float('inf')

    # Calculate Sortino Ratio
    return (returns - riskfree) / downside_deviation


def indexing(data: pd.Series, base: int = 100, weight: pd.Series = None) -> pd.Series:
    """
    #### Description:
    The function calculates the indexed values based on the percentage change and cumulative product of the input data.
    Optionally, it supports weighting the components if a weight Series is provided.

    #### Parameters:
    - data (pd.Series or pd.DataFrame): Time series data representing the value of a financial instrument.
    - base (float, optional): Initial base value for indexing. Default is 100.
    - weight (pd.Series, optional): Weighting factor for different components in the data. Default is None.

    #### Returns:
    - pd.Series: Indexed time series data.
    """
    # Calculate percentage change and cumulative product for indexing
    indexed_data = base * (data.pct_change() + 1).cumprod()

    # Set the initial base value for indexing
    indexed_data.iloc[0] = base

    # Apply weights if provided
    if isinstance(weight, pd.Series):
        indexed_data = (indexed_data * weight).sum(axis=1)
    return indexed_data


def historical_var(asset, freq: str = 'B', conf_level: float = 0.05) -> float:
    """
    #### Description:
    Calculate the Value at Risk (VaR) for a given financial instrument based on its historical performance.
    VaR measures the maximum potential loss at a specified confidence level over a given time horizon.

    #### Parameters:
    - asset (pd.Series or pd.DataFrame): Historical price or return data of the financial instrument.
    - freq (str): The frequency of the data, e.g., 'D' for daily, 'M' for monthly.
    - conf_level (float): The confidence level for VaR calculation, typically between 0 and 1.

    #### Returns:
    - float: The Value at Risk (VaR), representing the maximum potential loss at the specified confidence level.
    """
    # Resample the historical data based on the specified frequency
    resampled_history = asset.resample(freq).ffill()

    # Calculate the percentage change and drop any NaN values
    returns = resampled_history.pct_change().dropna()

    # Calculate VaR at the specified confidence level
    sorted_returns = returns.dropna().sort_values()
    var = sorted_returns.iloc[round(len(sorted_returns) * conf_level)]

    return abs(var)


def momentum(asset: pd.Series, period: int, differential: str ='last', method: str ='normal') -> pd.Series:
    """
    #### Description:
    Calculates the momentum of a time series data over a specified period.

    Momentum measures the rate of change in the price of a security over a specific period.
    It is commonly used by traders and analysts to identify trends and potential buying or selling signals.

    #### Parameters:
    - asset (pd.Series or pd.DataFrame): Time series data representing the price or value of a security.
    - period (int): The number of data points over which momentum is calculated.
    - differential (str, default='last'): Determines how the reference value is calculated within the period.
        - 'last': Uses the last value within the rolling window as the reference value.
        - 'mean': Uses the mean (average) value within the rolling window as the reference value.
    - method (str, default='normal'): Determines the method of calculating momentum.
        - 'normal': Calculates the difference between the current value and the reference value.
        - 'roc': Calculates the Rate of Change (ROC) as a percentage.
        - 'roclog': Calculates the logarithmic Rate of Change (ROC) as a percentage.

    #### Returns:
    - pd.Series: Series containing the calculated momentum values.
    """
    # Calculate the reference values based on the selected method
    if differential == 'last':
        ctx = asset.rolling(window=period).apply(lambda x: x[0], raw=True).dropna()
    elif differential == 'mean':
        ctx = asset.rolling(window=period).mean().dropna()

    # Reindex the original history to align with the reference values
    ct = asset.reindex(ctx.index)

    # Calculate momentum based on the selected method
    if method == 'normal':
        mo = ct - ctx
    elif method == 'roc':
        mo = 100 * ct / ctx
    elif method == 'roclog':
        mo = 100 * np.log(np.array(ct / ctx))
        mo = pd.Series(mo, index=asset.index[-len(ct):])
    return mo


def gbm_multi(n_step: int, n_scenario: int, mu: pd.Series, sigma: pd.Series, corr_matrix: pd.DataFrame, p_0: float, allocation: pd.Series) -> np.ndarray:
    """
    Simulates a multi-asset Geometric Brownian Motion (GBM) for a portfolio.

    Parameters
    ----------
    n_step : int
        Number of time steps (e.g. trading days).
    n_scenario : int
        Number of Monte Carlo simulation paths.
    mu : pd.Series
        Annualized expected returns of each asset.
    sigma : pd.Series
        Annualized volatilities of each asset.
    corr_matrix : pd.DataFrame
        Correlation matrix between assets.
    p_0 : float
        Initial total portfolio value.
    allocation : pd.Series
        Portfolio weights for each asset (should sum to 1).

    Returns
    -------
    ndarray
        Simulated total portfolio values of shape (n_step, n_scenario).
    """

    # Ensure mu, sigma, and allocation are NumPy arrays
    mu, sigma, allocation = map(np.array, (mu, sigma, allocation))

    # Compute the covariance matrix using the volatilities and correlation matrix
    cov_matrix = np.outer(sigma, sigma) * corr_matrix

    # Cholesky decomposition to generate correlated standard normal variables
    chol_matrix = np.linalg.cholesky(cov_matrix)

    # Generate uncorrelated standard normal random variables
    Z = np.random.normal(size=(n_step, n_scenario, len(mu)))

    # Apply Cholesky transform to induce correlation structure
    correlated_Z = Z @ chol_matrix.T

    # Time increment assuming 252 trading days
    dt = 1 / 252

    # Compute drift term (discretized compounding)
    drift = (1 + mu) ** dt

    # Compute diffusion scale
    diffusion = np.sqrt(dt)

    # Initialize return array
    returns = np.empty_like(correlated_Z)
    returns[0] = 1  # Initial value set to 1 for all assets and scenarios

    # Compute GBM returns
    returns[1:] = drift * np.exp(correlated_Z[1:] * diffusion)

    # Compute cumulative portfolio value across time steps and scenarios
    portfolio = p_0 * allocation * np.cumprod(returns, axis=0)

    # Sum across assets to get total portfolio value
    return portfolio.sum(axis=2)


def relative_volatility_contribution(asset: pd.DataFrame, weight: pd.Series) -> pd.Series:
    """
    Calculate the relative contribution of each asset to the total portfolio volatility.

    Parameters
    ----------
    asset : pd.DataFrame
        Time series of asset prices (columns = asset names, rows = dates).
    weight : pd.Series
        Portfolio weights (index = asset names).

    Returns
    -------
    pd.Series
        Relative contribution of each asset to total portfolio volatility (sum = 1.0).
    """
    # Calculate returns
    returns = asset.pct_change().dropna()
    
    # Covariance matrix
    cov_matrix = returns.cov()

    # Total portfolio volatility
    port_vol = np.sqrt(weight.T @ cov_matrix @ weight)

    # Absolute contribution to volatility
    marginal_contrib = cov_matrix @ weight / port_vol
    contrib_abs = weight * marginal_contrib

    # Relative contribution
    cvr = contrib_abs / port_vol
    cvr.name = 'relative_vol_contribution'

    return cvr


def marginal_contribution_to_risk(asset: pd.DataFrame, weight: pd.Series) -> pd.Series:
    """
    Calculate the marginal contribution to risk (MCR) of each asset in a portfolio.

    Parameters
    ----------
    asset : pd.DataFrame
        Time series of asset prices or NAVs (columns = asset names, rows = dates).
    weight : pd.Series
        Portfolio weights (index = asset names, same names as columns of `asset`).

    Returns
    -------
    pd.Series
        Marginal contribution to risk (MCR) of each asset (same unit as volatility, does not sum to 1).
    """
    # Calculate returns
    returns = asset.pct_change().dropna()

    # Covariance matrix
    cov_matrix = returns.cov()

    # Total portfolio volatility
    port_vol = np.sqrt(weight.T @ cov_matrix @ weight)

    # MCR = (Σ w)_i / σ_p
    mcr = cov_matrix @ weight / port_vol
    mcr.name = 'marginal_contribution_to_risk'

    return mcr

def upside_capture(asset: pd.Series, benchmark: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Upside Capture Ratio of an asset relative to a benchmark.
    The Upside Capture Ratio measures how well the asset performs relative to the benchmark
    during periods when the benchmark has positive returns.

    #### Parameters:
    - asset (pd.Series): Historical price or return data of the asset.
    - benchmark (pd.Series): Historical price or return data of the benchmark.
    - timeperiod (int, optional): The number of periods per year for annualization. Default is 252.

    #### Returns:
    - float: Upside Capture Ratio.
    """
    asset_returns = asset.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    # Align series on dates
    asset_returns, benchmark_returns = asset_returns.align(benchmark_returns, join='inner')

    # Filter periods when benchmark return is positive
    positive_mask = benchmark_returns > 0
    asset_positive = asset_returns[positive_mask]
    benchmark_positive = benchmark_returns[positive_mask]

    if benchmark_positive.mean() == 0:
        return float('inf')

    return (asset_positive.mean() * timeperiod) / (benchmark_positive.mean() * timeperiod)


def downside_capture(asset: pd.Series, benchmark: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Downside Capture Ratio of a financial instrument relative to a benchmark.
    The Downside Capture Ratio measures how well the asset performs relative to the benchmark
    during periods when the benchmark has negative returns.

    #### Parameters:
    - asset (pd.Series): Historical price or return data of the asset.
    - benchmark (pd.Series): Historical price or return data of the benchmark.
    - timeperiod (int, optional): The number of periods per year for annualization. Default is 252.

    #### Returns:
    - float: Downside Capture Ratio.
    """
    asset_returns = asset.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    # Align series on dates
    asset_returns, benchmark_returns = asset_returns.align(benchmark_returns, join='inner')

    # Filter periods when benchmark return is negative
    negative_mask = benchmark_returns < 0
    asset_negative = asset_returns[negative_mask]
    benchmark_negative = benchmark_returns[negative_mask]

    if benchmark_negative.mean() == 0:
        return float('inf')

    return (asset_negative.mean() * timeperiod) / (benchmark_negative.mean() * timeperiod)
