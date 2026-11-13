"""
evaluate.py - Out-of-sample evaluation metrics: Pinball Loss (Tick Loss) and RMSE
"""
import numpy as np

def pinball_loss(y_true, y_pred, tau):
    """
    Compute Pinball Loss (Quantile Loss / Tick Loss) for quantile tau.
    
    L_tau(y, q) = max(tau * (y - q), (tau - 1) * (y - q))
    """
    err = y_true - y_pred
    return np.mean(np.maximum(tau * err, (tau - 1) * err))

def rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error (RMSE).
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
