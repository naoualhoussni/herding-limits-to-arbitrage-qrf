"""
data_loader.py - Chargement et préparation des données de marché pour SHA et SZA
"""
import os
import pandas as pd

def load_daily_data(data_dir='data', market='SHA'):
    filename = f'daily_market_{market.lower()}.csv'
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

if __name__ == '__main__':
    df = load_daily_data('.', 'SHA')
    print("Dataset SHA chargé :", df.shape)
