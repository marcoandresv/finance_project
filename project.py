import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

sp500 = "^GSPC"

"""
S&P 500 ticker = 
^GSPC
"""


def download_fred_csv(indicator, save_path):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={indicator}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {indicator} data to {save_path}")
    except Exception as e:
        print(f"Failed to download data for {indicator}: {e}")


# def load_csv(filepath, date_col="DATE", years=5):
def load_csv(filepath, date_col="observation_date", years=5):
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        end_date = df[date_col].max()  # to use the latest data
        start_date = end_date - pd.DateOffset(years=years)
        filtered_df = df[df[date_col] >= start_date]
        filtered_df = filtered_df.sort_values(date_col)
        print(f"Loaded and filtered data from {filepath}")
        return filtered_df
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def get_sp500_data(years=5):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    ticker = "^GSPC"
    try:
        sp500 = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        print("Retrieved S&P500 data")
        return sp500
    except Exception as e:
        print(f"Error processing S&P500 data: {e}")
        return None


"""

UNRATE = unemployment rate
CPIAUCSL = consumer price index for all urban consumers
INDPRO = industrial production index
FEDFUNDS = effective federal funds rate

"""

if __name__ == "__main__":
    # downloading data for the following indicators
    indicators = ["UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS"]
    # for indicator in indicators:
    #     download_fred_csv(indicator, f"data/raw/{indicator}.csv")
    """
    unrate_data = load_csv("data/raw/UNRATE.csv", date_col="observation_date")
    print(unrate_data.head())
    cpiaucsl_data = load_csv("data/raw/CPIAUCSL.csv", date_col="observation_date")
    print(cpiaucsl_data.head())
    indpro_data = load_csv("data/raw/INDPRO.csv", date_col="observation_date")
    print(indpro_data.head())
    fedfunds_data = load_csv("data/raw/FEDFUNDS.csv", date_col="observation_date")
    print(fedfunds_data.head())
    """

    for indicator in indicators:
        data_to_print = load_csv(
            f"data/raw/{indicator}.csv", date_col="observation_date", years=5
        )
        print(data_to_print.tail())

    sp500_data = get_sp500_data()
    print(sp500_data.head())
