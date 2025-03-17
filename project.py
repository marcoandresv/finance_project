import pandas as pd
import yfinance as yf

sp500 = "^GSPC"

"""
S&P 500 ticker = 
^GSPC
"""


# fnc to load csv file
def load_csv(filepath):
    try:
        data = pd.read_csv(filepath)
        print("Data loaded")
        return data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def get_stock_data(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)  # S&P 500 ticker symbol
        return stock_data
    except Exception:
        return None


sp500_data = get_stock_data(sp500, "2020-01-01", "2025-01-01")
print(sp500_data.head())

# if __name__ == "__main__":
