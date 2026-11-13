"""
evaluate.py - Fonctions d'évaluation : Pinball Loss et RMSE
"""
import numpy as np

def pinball_loss(y_true, y_pred, tau):
    """Calcule la perte Pinball (Tick Loss) pour le quantile tau."""
    err = y_true - y_pred
    return np.mean(np.maximum(tau * err, (tau - 1) * err))

def rmse(y_true, y_pred):
    """Calcule l'erreur quadratique moyenne."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
