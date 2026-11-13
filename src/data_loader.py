"""
data_loader.py - Data ingestion and validation for Shanghai (SHA) and Shenzhen (SZA) markets
"""
import os
import pandas as pd

def load_daily_data(data_dir='data', market='SHA'):
    """
    Load daily aggregated market dataset for a specific Chinese stock exchange.
    
    Parameters:
    -----------
    data_dir : str
        Directory where CSV files are located.
    market : str
        'SHA' for Shanghai A-shares or 'SZA' for Shenzhen A-shares.
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with parsed Date column.
    """
    filename = f'daily_market_{market.lower()}.csv'
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

if __name__ == '__main__':
    df = load_daily_data('.', 'SHA')
    print(f"Successfully loaded SHA dataset: {df.shape}")
