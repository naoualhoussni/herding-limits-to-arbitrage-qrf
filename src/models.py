"""
models.py - Implémentation des modèles de mimétisme : OLS, QuantileRegressor et QRF
"""
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def fit_quantile_regression(X, y, tau=0.5, alpha=0):
    model = QuantileRegressor(quantile=tau, alpha=alpha, solver='highs')
    model.fit(X, y)
    return model

def fit_qrf(X, y, tau=0.5, n_estimators=150, max_depth=4, learning_rate=0.05):
    model = GradientBoostingRegressor(loss='quantile', alpha=tau, n_estimators=n_estimators,
                                      max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    model.fit(X, y)
    return model
