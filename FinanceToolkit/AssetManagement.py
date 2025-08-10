import pandas as pd
import numpy as np
from typing import Literal

def beta(history: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """
    Calculate the beta of each asset in the history DataFrame relative to the benchmark.

    Parameters:
    - history (pd.DataFrame): DataFrame where each column represents the returns of an asset.
    - benchmark (pd.Series): Series representing the returns of the benchmark.

    Returns:
    - pd.Series: Series where each element is the beta of an asset relative to the benchmark.
    """
    # Calculate percentage changes and drop NA values
    history_returns = history.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    # Combine history and benchmark returns into a DataFrame
    df = pd.concat([history_returns, benchmark_returns], axis=1).dropna()
    history_aligned = df.iloc[:,:-1] 
    benchmark_aligned = df.iloc[:,-1]

    # Calculate the covariance and benchmark variance
    benchmark_variance = benchmark_aligned.var()

    # Calculate the covariance of each asset with the benchmark
    covariances = history_aligned.apply(lambda x: x.cov(benchmark_aligned))

    # Calculate beta for each asset using vectorized operations
    betas = covariances / benchmark_variance

    return betas.rename('Beta')

def beta_advanced(history: pd.Series, benchmark: pd.Series, way: Literal['+', '-', 'all']='all') -> float:
    """
    #### Description:
    Beta is a measure of a financial instrument's sensitivity to benchmark movements. A beta of 1 indicates the history tends
    to move in line with the benchmark, a beta greater than 1 suggests higher volatility, and a beta less than 1 indicates
    lower volatility compared to the benchmark.

    #### Parameters:
    - history (pd.Series): Time series data representing the returns of the history.
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
    df = pd.concat([history, benchmark], axis=1).dropna().pct_change().dropna()

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


def theta(history: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the average return of a financial instrument over a specified time period.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average return. Default is 252.

    #### Returns:
    - float: Average return over the specified time period.
    """
    returns = history.pct_change().dropna()
    # Fix for FutureWarning: explicitly specify axis=0 for prod()
    if isinstance(returns, pd.DataFrame):
        cumulative_return = (1 + returns).prod(axis=0) ** (timeperiod / len(returns)) - 1
    else:
        cumulative_return = (1 + returns).prod() ** (timeperiod / len(returns)) - 1
    return cumulative_return

def sigma(history: pd.Series, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the average volatility of a financial instrument over a specified time period.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.
    - timeperiod (int, optional): The number of periods to consider for calculating the average volatility. Default is 252.

    #### Returns:
    - float: Average volatility over the specified time period.
    """
    returns = history.pct_change().dropna()
    return returns.std() * np.sqrt(timeperiod)

def max_drawdown(history: pd.Series) -> float:
    """
    #### Description:
    Max Drawdown is a measure of the largest loss from a peak to a trough in a financial instrument's historical performance.
    It is calculated as the percentage decline relative to the highest cumulative value.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical prices or returns of the financial instrument.

    #### Returns:
    - float: Maximum drawdown, representing the largest percentage decline from a peak to a trough.
    """
    # Calculate the cumulative percentage change in price or returns
    index = 100 * (1 + history.pct_change()).cumprod()

    # Identify the peaks and calculate drawdowns
    peaks = index.cummax()
    drawdowns = (index - peaks) / peaks

    # Return the maximum drawdown
    return drawdowns.min()

def jensen_alpha(history: pd.Series, benchmark: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
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
    # Calculate percentage change in history and benchmark
    history_clean = history.dropna()
    benchmark_clean = benchmark.dropna()
    
    # Align the series
    aligned_data = pd.concat([history_clean, benchmark_clean], axis=1).dropna()
    history_aligned = aligned_data.iloc[:, 0]
    benchmark_aligned = aligned_data.iloc[:, 1]
    
    # Calculate returns
    history_return = (history_aligned.iloc[-1] - history_aligned.iloc[0]) / history_aligned.iloc[0]
    benchmark_return = (benchmark_aligned.iloc[-1] - benchmark_aligned.iloc[0]) / benchmark_aligned.iloc[0]
    
    # Calculate beta using the beta function
    history_beta = beta(pd.DataFrame({'asset': history_aligned}), benchmark_aligned).iloc[0]

    # Calculate Jensen Alpha
    return history_return - (riskfree + history_beta * (benchmark_return - riskfree))


def alpha(history: pd.Series, benchmark: pd.Series) -> float:
    """
    #### Description:
    Alpha measures the history's performance relative to the benchmark. A positive alpha indicates that the history has outperformed
    the benchmark, while a negative alpha suggests underperformance.

    #### Parameters:
    - history (pd.Series): Time series data representing the historical performance of the history.
    - benchmark (pd.Series): Time series data representing the historical performance of the benchmark.

    #### Returns:
    - float: Alpha, representing the excess return of the history over the benchmark.
    """
    # Calculate percentage change in history and benchmark
    history_return = (history.iloc[-1] - history.iloc[0]) / history.iloc[0]
    benchmark_return = (benchmark.iloc[-1] - benchmark.iloc[0]) / benchmark.iloc[0]

    # Calculate alpha as the difference in returns
    return history_return - benchmark_return


def sharpe(history: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Sharpe Ratio for a given financial instrument based on its historical performance.
    The Sharpe Ratio helps assess the investment's return per unit of risk taken.

    #### Parameters:
    - history (pd.Series): Historical price or return data of the financial instrument.
    - riskfree (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = theta(history=history, timeperiod=timeperiod)
    volatility = sigma(history=history, timeperiod=timeperiod)
    
    # Calculate Sharpe Ratio using the formula
    return (returns - riskfree) / volatility

def calmar(history: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Calmar Ratio for a given financial instrument based on its historical performance.
    The Calmar ratio uses a financial instrument's maximum drawdown as its sole measure of risk

    #### Parameters:
    - history (pd.Series): Historical price or return data of the financial instrument.
    - riskfree (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Calmar Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and max drawdown using helper functions
    returns = theta(history=history, timeperiod=timeperiod)
    maxdrawdown = max_drawdown(history=history)
    
    if isinstance(history, pd.Series):
        if abs(maxdrawdown) != 0:
            # Calculate Calmar Ratio using the formula
            return (returns - riskfree) / abs(maxdrawdown)
        else:
            return np.nan
    elif isinstance(history, pd.DataFrame):
        calmar_ratios = pd.Series(index=history.columns)
        for column in history.columns:
            column_returns = theta(history[column], timeperiod)
            column_drawdown = max_drawdown(history[column])
            if abs(column_drawdown) != 0:
                # Calculate Calmar Ratio for each column in the DataFrame
                calmar_ratios[column] = (column_returns - riskfree) / abs(column_drawdown)
            else:
                calmar_ratios[column] = np.nan
        return calmar_ratios

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

def historical_var(history, freq: str = 'B', conf_level: float = 0.05) -> float:
    """
    #### Description:
    Calculate the Value at Risk (VaR) for a given financial instrument based on its historical performance.
    VaR measures the maximum potential loss at a specified confidence level over a given time horizon.

    #### Parameters:
    - history (pd.Series or pd.DataFrame): Historical price or return data of the financial instrument.
    - freq (str): The frequency of the data, e.g., 'D' for daily, 'M' for monthly.
    - conf_level (float): The confidence level for VaR calculation, typically between 0 and 1.

    #### Returns:
    - float: The Value at Risk (VaR), representing the maximum potential loss at the specified confidence level.
    """
    # Resample the historical data based on the specified frequency
    resampled_history = history.resample(freq).ffill()

    # Calculate the percentage change and drop any NaN values
    returns = resampled_history.pct_change().dropna()

    # Calculate VaR at the specified confidence level
    if isinstance(returns, pd.Series):
        sorted_returns = returns.dropna().sort_values()
        var = sorted_returns.iloc[round(len(sorted_returns) * conf_level)]
    elif isinstance(returns, pd.DataFrame):
        var = pd.Series(index=returns.columns)
        for ticker in returns.columns:
            data = returns[ticker].dropna().sort_values()
            var[ticker] = data.iloc[round(len(data) * conf_level)]

    return abs(var)

def momentum(history: pd.Series, period: int, differential: str ='last', method: str ='normal') -> pd.Series:
    """
    #### Description:
    Calculates the momentum of a time series data over a specified period.

    Momentum measures the rate of change in the price of a security over a specific period.
    It is commonly used by traders and analysts to identify trends and potential buying or selling signals.

    #### Parameters:
    - history (pd.Series or pd.DataFrame): Time series data representing the price or value of a security.
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
        ctx = history.rolling(window=period).apply(lambda x: x[0], raw=True).dropna()
    elif differential == 'mean':
        ctx = history.rolling(window=period).mean().dropna()

    # Reindex the original history to align with the reference values
    ct = history.reindex(ctx.index)

    # Calculate momentum based on the selected method
    if method == 'normal':
        mo = ct - ctx
    elif method == 'roc':
        mo = 100 * ct / ctx
    elif method == 'roclog':
        mo = 100 * np.log(np.array(ct / ctx))
        mo = pd.Series(mo, index=history.index[-len(ct):])
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

def relative_volatility_contribution(history: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Calculate the relative contribution of each asset to the total portfolio volatility.

    Parameters
    ----------
    history : pd.DataFrame
        Time series of asset prices (columns = asset names, rows = dates).
    weights : pd.Series
        Portfolio weights (index = asset names).

    Returns
    -------
    pd.Series
        Relative contribution of each asset to total portfolio volatility (sum = 1.0).
    """
    # Calculate returns
    returns = history.pct_change().dropna()
    
    # Covariance matrix
    cov_matrix = returns.cov()

    # Total portfolio volatility
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    # Absolute contribution to volatility
    marginal_contrib = cov_matrix @ weights / port_vol
    contrib_abs = weights * marginal_contrib

    # Relative contribution
    cvr = contrib_abs / port_vol
    cvr.name = 'relative_vol_contribution'

    return cvr

def marginal_contribution_to_risk(history: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Calculate the marginal contribution to risk (MCR) of each asset in a portfolio.

    Parameters
    ----------
    history : pd.DataFrame
        Time series of asset prices or NAVs (columns = asset names, rows = dates).
    weights : pd.Series
        Portfolio weights (index = asset names, same names as columns of `history`).

    Returns
    -------
    pd.Series
        Marginal contribution to risk (MCR) of each asset (same unit as volatility, does not sum to 1).
    """
    # Calculate returns
    returns = history.pct_change().dropna()

    # Covariance matrix
    cov_matrix = returns.cov()

    # Total portfolio volatility
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    # MCR = (Σ w)_i / σ_p
    mcr = cov_matrix @ weights / port_vol
    mcr.name = 'marginal_contribution_to_risk'

    return mcr