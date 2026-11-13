"""
models.py - Model implementations for Herding: OLS, QuantileRegressor, and QRF
"""
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def fit_quantile_regression(X, y, tau=0.5, alpha=0):
    """
    Fit linear quantile regression at quantile tau using HiGHS interior point solver.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    tau : float, quantile level in (0, 1)
    alpha : float, L1 regularization strength
    
    Returns:
    --------
    fitted QuantileRegressor
    """
    model = QuantileRegressor(quantile=tau, alpha=alpha, solver='highs')
    model.fit(X, y)
    return model

def fit_qrf(X, y, tau=0.5, n_estimators=150, max_depth=4, learning_rate=0.05):
    """
    Fit non-parametric Quantile Regression Forest / Gradient Tree at quantile tau.
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    tau : float, quantile level in (0, 1)
    n_estimators : int
    max_depth : int
    learning_rate : float
    
    Returns:
    --------
    fitted GradientBoostingRegressor with quantile loss
    """
    model = GradientBoostingRegressor(
        loss='quantile',
        alpha=tau,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X, y)
    return model
