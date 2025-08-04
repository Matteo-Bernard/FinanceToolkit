import pandas as pd
import numpy as np
from typing import Literal

def beta(asset: pd.Series, market: pd.Series, way: Literal['+', '-', 'all']='all') -> float:
    """
    #### Description:
    Beta is a measure of a financial instrument's sensitivity to market movements. A beta of 1 indicates the asset tends
    to move in line with the market, a beta greater than 1 suggests higher volatility, and a beta less than 1 indicates
    lower volatility compared to the market.

    #### Parameters:
    - asset (pd.Series): Time series data representing the returns of the asset.
    - market (pd.Series): Time series data representing the returns of the market.
    - way (Literal['+', '-', 'all']): Specifies which type of data points should be considered for the beta calculation:
        - '+' (positive): Only considers periods where the asset's returns are positive. This is useful for measuring
          the beta when the asset is performing well.
        - '-' (negative): Only considers periods where the asset's returns are negative. This is useful for measuring
          the beta when the asset is underperforming.
        - 'all': Considers all periods without any filtering, giving the traditional beta measurement.

    #### Returns:
    - float: Beta coefficient, which measures the asset's sensitivity to market movements based on the specified filter.
    """
    # Combine asset and market returns, calculate covariance and market variance
    df = pd.concat([asset, market], axis=1).dropna().pct_change().dropna()

    # Filter data based on the 'way' parameter, focusing on asset variations
    if way == '+':
        df = df[df.iloc[:, 0] > 0]  # Filter where asset returns are positive
    elif way == '-':
        df = df[df.iloc[:, 0] < 0]  # Filter where asset returns are negative
    elif way == 'all':
        pass  # No filtering needed

    # Calculate covariance and market variance
    covariance = df.cov().iloc[1, 0]
    market_variance = df.iloc[:, 1].var()

    # Calculate beta as the ratio of covariance to market variance
    return covariance / market_variance


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
    return (np.prod(1 + returns) ** (timeperiod / len(returns))) - 1

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
    returns = history.pct_change(fill_method=None).dropna()
    return np.std(returns) * np.sqrt(timeperiod)

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

def jensen_alpha(asset: pd.Series, market: pd.Series, riskfree: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Alpha measures the asset's performance relative to the market. A positive alpha indicates that the asset has outperformed
    the market, while a negative alpha suggests underperformance.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical performance of the asset.
    - market (pd.Series): Time series data representing the historical performance of the market.

    #### Returns:
    - float: Alpha, representing the excess return of the asset over the market.
    """
    # Calculate percentage change in asset and market
    asset = asset.dropna()
    market = market.dropna()
    asset_return = (asset.iloc[-1] - asset.iloc[0]) / asset.iloc[0]
    market_return = (market.iloc[-1] - market.iloc[0]) / market.iloc[0]
    asset_beta = beta(asset, market)

    #Calculte Jensen Alpha
    return asset_return - (riskfree + asset_beta * (market_return - riskfree))


def alpha(asset: pd.Series, market: pd.Series) -> float:
    """
    #### Description:
    Alpha measures the asset's performance relative to the market. A positive alpha indicates that the asset has outperformed
    the market, while a negative alpha suggests underperformance.

    #### Parameters:
    - asset (pd.Series): Time series data representing the historical performance of the asset.
    - market (pd.Series): Time series data representing the historical performance of the market.

    #### Returns:
    - float: Alpha, representing the excess return of the asset over the market.
    """
    # Calculate percentage change in asset and market
    asset_return = (asset.iloc[-1] - asset.iloc[0]) / asset.iloc[0]
    market_return = (market.iloc[-1] - market.iloc[0]) / market.iloc[0]

    # Calculate alpha as the difference in returns
    return asset_return - market_return


def sharpe(history: pd.Series, risk_free: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Sharpe Ratio for a given financial instrument based on its historical performance.
    The Sharpe Ratio helps assess the investment's return per unit of risk taken.

    #### Parameters:
    - history (list or numpy array): Historical price or return data of the financial instrument.
    - risk_free (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = theta(history=history, timeperiod=timeperiod)
    volatility = sigma(history=history, timeperiod=timeperiod)
    
    # Calculate Sharpe Ratio using the formula
    return (returns - risk_free) / volatility

def calmar(history: pd.Series, risk_free: float, timeperiod: int = 252) -> float:
    """
    #### Description:
    Calculate the Calmar Ratio for a given financial instrument based on its historical performance.
    The Calmar ratio uses a financial instrument’s maximum drawdown as its sole measure of risk

    #### Parameters:
    - history (list or numpy array): Historical price or return data of the financial instrument.
    - risk_free (float): The risk-free rate of return, typically representing the return on a risk-free investment.
    - timeperiod (int, optional): The time period used for calculating average return and volatility. Default is 252.

    #### Returns:
    - float: The Sharpe Ratio, a measure of the instrument's risk-adjusted performance.
    """
    # Calculate average return and volatility using helper functions
    returns = theta(history=history, timeperiod=timeperiod)
    maxdrawdown = max_drawdown(history=history)
    
    if abs(maxdrawdown) != 0:
        # Calculate Sharpe Ratio using the formula
        return (returns - risk_free) / abs(maxdrawdown)
    else:
        return np.nan

def indexing(data: pd.Series, base: int = 100, weight: pd.Series = None) -> pd.Series:
    """
    #### Description:
    The function calculates the indexed values based on the percentage change and cumulative product of the input data.
    Optionally, it supports weighting the components if a weight Series is provided.

    #### Parameters:
    - data (pandas.Series or pandas.DataFrame): Time series data representing the value of a financial instrument.
    - base (float, optional): Initial base value for indexing. Default is 100.
    - weight (pandas.Series, optional): Weighting factor for different components in the data. Default is None.

    #### Returns:
    - pandas.Series: Indexed time series data.
    """
    # Calculate percentage change and cumulative product for indexing
    data = base * (data.pct_change() + 1).cumprod()

    # Set the initial base value for indexing
    data.loc[data.index[0]] = base

    # Apply weights if provided
    if isinstance(weight, pd.Series):
        data = (data * weight).sum(axis=1)
    return data

def historical_var(history, freq: str = 'B', conf_level: float = 0.05) -> float:
    """
    #### Description:
    Calculate the Value at Risk (VaR) for a given financial instrument based on its historical performance.
    VaR measures the maximum potential loss at a specified confidence level over a given time horizon.

    #### Parameters:
    - history (pandas Series or DataFrame): Historical price or return data of the financial instrument.
    - freq (str): The frequency of the data, e.g., 'D' for daily, 'M' for monthly.
    - conf_level (float): The confidence level for VaR calculation, typically between 0 and 1.

    #### Returns:
    - float: The Value at Risk (VaR), representing the maximum potential loss at the specified confidence level.
    """
    # Resample the historical data based on the specified frequency
    history = history.resample(freq).ffill()

    # Calculate the percentage change and drop any NaN values
    history = history.pct_change().dropna()

    # Calculate VaR at the specified confidence level
    if isinstance(history, pd.Series):
        history = history.dropna().sort_values()
        var = history.iloc[round(len(history) * conf_level)]
    elif isinstance(history, pd.DataFrame):
        var = pd.Series(index=history.columns)
        for ticker in history.columns:
            data = history[ticker].dropna()
            data = data.sort_values()
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
    mu : array-like
        Annualized expected returns of each asset.
    sigma : array-like
        Annualized volatilities of each asset.
    corr_matrix : 2D array-like
        Correlation matrix between assets.
    p_0 : float
        Initial total portfolio value.
    allocation : array-like
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
    Calcule la contribution relative de chaque actif à la volatilité totale du portefeuille.

    Parameters
    ----------
    history : pd.DataFrame
        Séries temporelles des cours des actifs (colonnes = noms des actifs, lignes = dates).
    weights : pd.Series
        Pondérations du portefeuille (index = noms des actifs).

    Returns
    -------
    pd.Series
        Contribution relative de chaque actif à la volatilité totale du portefeuille (somme = 1.0).
    """
    # Calcul des rendements
    returns = history.pct_change().dropna()
    
    # Matrice de covariance
    cov_matrix = returns.cov()

    # Volatilité totale du portefeuille
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    # Contribution absolue à la volatilité
    marginal_contrib = cov_matrix @ weights / port_vol
    contrib_abs = weights * marginal_contrib

    # Contribution relative
    cvr = contrib_abs / port_vol
    cvr.name = 'relative_vol_contribution'

    return cvr

def marginal_contribution_to_risk(history: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """
    Calcule la marginal contribution to risk (MCR) de chaque actif dans un portefeuille.

    Parameters
    ----------
    history : pd.DataFrame
        Séries temporelles des prix ou des valeurs liquidatives des actifs (colonnes = noms des actifs, lignes = dates).
    weights : pd.Series
        Pondérations du portefeuille (index = noms des actifs, mêmes noms que les colonnes de `history`).

    Returns
    -------
    pd.Series
        Marginal contribution to risk (MCR) de chaque actif (même unité que la volatilité, ne somme pas à 1).
    """
    # Calcul des rendements
    returns = history.pct_change().dropna()

    # Matrice de covariance
    cov_matrix = returns.cov()

    # Volatilité totale du portefeuille
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)

    # MCR = (Σ w)_i / σ_p
    mcr = cov_matrix @ weights / port_vol
    mcr.name = 'marginal_contribution_to_risk'

    return mcr