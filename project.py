import os

import requests

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


if __name__ == "__main__":
    indicators = ["UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS"]
    for indicator in indicators:
        download_fred_csv(indicator, f"data/raw/{indicator}.csv")
