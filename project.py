import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


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


def load_csv(filepath, date_col="observation_date", years=5):
    """Loads a CSV file, filters data for the last `years` years, and returns a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        end_date = df[date_col].max()
        start_date = end_date - pd.DateOffset(years=years)
        filtered_df = df[df[date_col] >= start_date].sort_values(date_col)
        print(f"Loaded and filtered data from {filepath}")
        return filtered_df
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def get_sp500_data(years=5):
    """Downloads historical S&P 500 data for the last `years` years."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    ticker = "^GSPC"
    try:
        # Get S&P500 data
        sp500 = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            group_by=None,
        )

        # Print columns to debug
        print(f"S&P500 data columns: {sp500.columns}")

        # Extract the Close column and handle various column formats
        if "Close" in sp500.columns:
            sp500_df = sp500[["Close"]].reset_index()
        elif ("Close", "^GSPC") in sp500.columns:
            sp500_df = sp500[[("Close", "^GSPC")]].reset_index()
        elif any("close" in str(col).lower() for col in sp500.columns):
            # Find any column containing 'close' (case insensitive)
            close_col = [col for col in sp500.columns if "close" in str(col).lower()][0]
            sp500_df = sp500[[close_col]].reset_index()
            sp500_df.rename(columns={close_col: "Close"}, inplace=True)
        else:
            # If all else fails, just take the price column (typically 4th column in most API responses)
            print(f"Could not find Close column. Using column: {sp500.columns[3]}")
            sp500_df = sp500[[sp500.columns[3]]].reset_index()
            sp500_df.rename(columns={sp500.columns[3]: "Close"}, inplace=True)

        # Rename columns
        sp500_df.rename(
            columns={"Date": "observation_date", "Close": "SP500_Close"}, inplace=True
        )

        # Ensure columns are flat
        if isinstance(sp500_df.columns, pd.MultiIndex):
            sp500_df.columns = [
                col[0] if isinstance(col, tuple) else col for col in sp500_df.columns
            ]

        print("Retrieved S&P500 data")
        return sp500_df
    except Exception as e:
        print(f"Error processing S&P500 data: {e}")
        # Additional debugging
        if "sp500" in locals():
            print(f"S&P500 data shape: {sp500.shape}")
            print(f"S&P500 data columns: {sp500.columns}")
            print(f"S&P500 data sample:\n{sp500.head()}")
        return None


def get_monthly_sp500(df):
    """Resamples S&P 500 data to monthly frequency using the last available value of each month."""
    try:
        # Make a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Ensure observation_date is datetime
        df_copy["observation_date"] = pd.to_datetime(df_copy["observation_date"])

        # Properly set index, resample, and reset index
        temp_df = df_copy.set_index("observation_date")
        monthly_df = temp_df.resample("ME").last().reset_index()

        # Format date to match FRED data (YYYY-MM-01)
        monthly_df["observation_date"] = monthly_df["observation_date"].dt.strftime(
            "%Y-%m-01"
        )
        monthly_df["observation_date"] = pd.to_datetime(monthly_df["observation_date"])

        # Ensure we don't have a MultiIndex
        if isinstance(monthly_df.columns, pd.MultiIndex):
            monthly_df.columns = [
                col[0] if isinstance(col, tuple) else col for col in monthly_df.columns
            ]

        print("Processed S&P500 data to monthly frequency")
        return monthly_df
    except Exception as e:
        print(f"Error processing monthly S&P500 data: {e}")
        print(f"DataFrame columns type: {type(df.columns)}")
        print(f"DataFrame columns: {df.columns}")
        return None


def merge_datasets(indicators_df, sp500_df):
    """Merges economic indicators with S&P500 data on observation_date."""
    try:
        # Debugging column types
        print("\nBefore merge:")
        print(f"Indicators DataFrame columns: {indicators_df.columns}")
        print(f"S&P500 DataFrame columns: {sp500_df.columns}")

        # Make sure both dataframes have observation_date as datetime
        indicators_df["observation_date"] = pd.to_datetime(
            indicators_df["observation_date"]
        )
        sp500_df["observation_date"] = pd.to_datetime(sp500_df["observation_date"])

        # Handle MultiIndex if present in SP500 data
        if isinstance(sp500_df.columns, pd.MultiIndex):
            # Convert MultiIndex to flat index
            sp500_df.columns = [
                col[0] if isinstance(col, tuple) else col for col in sp500_df.columns
            ]
            print("Converted S&P500 MultiIndex to flat index")

        # Perform the merge on observation_date
        merged_df = pd.merge(
            indicators_df, sp500_df, on="observation_date", how="inner"
        )

        print("Successfully merged datasets")
        print(
            f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns"
        )
        return merged_df
    except Exception as e:
        print(f"Error merging data: {e}")
        print(f"Indicators shape: {indicators_df.shape}, SP500 shape: {sp500_df.shape}")
        print(f"Indicators columns: {indicators_df.columns}")
        print(f"SP500 columns: {sp500_df.columns}")
        return None


if __name__ == "__main__":
    indicators = ["UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS"]

    # Ensure data directories exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Download economic indicators from FRED
    print("Downloading economic indicators from FRED...")
    for indicator in indicators:
        download_fred_csv(indicator, f"data/raw/{indicator}.csv")

    # Load and print last few rows of each indicator
    for indicator in indicators:
        data_to_print = load_csv(f"data/raw/{indicator}.csv")
        if data_to_print is not None:
            print(data_to_print.tail())
        else:
            print(f"Failed to load {indicator} data")

    # Get and process S&P 500 data
    print("Getting S&P 500 data...")
    sp500_data = get_sp500_data()

    if sp500_data is not None:
        # Print S&P500 columns to diagnose any MultiIndex issues
        print(f"S&P500 DataFrame columns type: {type(sp500_data.columns)}")
        print(f"S&P500 DataFrame columns: {sp500_data.columns}")

        # Ensure columns are not MultiIndex
        if isinstance(sp500_data.columns, pd.MultiIndex):
            sp500_data.columns = [
                col[0] if isinstance(col, tuple) else col for col in sp500_data.columns
            ]
            print("Converted S&P500 columns from MultiIndex to flat index")

        sp500_monthly = get_monthly_sp500(sp500_data)

        if sp500_monthly is not None:
            # Standardize the date format for all datasets to YYYY-MM-01
            for indicator in indicators:
                df_path = f"data/raw/{indicator}.csv"
                if os.path.exists(df_path):
                    df = load_csv(df_path)
                    if df is not None:
                        # Set all dates to first day of month for consistent merging
                        df["observation_date"] = df["observation_date"].dt.strftime(
                            "%Y-%m-01"
                        )
                        df["observation_date"] = pd.to_datetime(df["observation_date"])
                        df.to_csv(df_path, index=False)

            # Load all indicator data and merge them into one DataFrame
            indicator_dfs = []
            for indicator in indicators:
                df = load_csv(f"data/raw/{indicator}.csv")
                if df is not None:
                    indicator_dfs.append(df)
                else:
                    print(f"Warning: Skipping {indicator} as data could not be loaded")

            if indicator_dfs:
                indicators_merged = indicator_dfs[0]  # Start with the first DataFrame

                # Merge all indicator datasets on "observation_date"
                for df in indicator_dfs[1:]:
                    indicators_merged = pd.merge(
                        indicators_merged, df, on="observation_date", how="inner"
                    )

                # Debug information
                print("\nDebug information before final merge:")
                print(f"Indicators merged shape: {indicators_merged.shape}")
                print(f"SP500 monthly shape: {sp500_monthly.shape}")
                print(
                    "Indicators merged date range:",
                    indicators_merged["observation_date"].min(),
                    "to",
                    indicators_merged["observation_date"].max(),
                )
                print(
                    "SP500 monthly date range:",
                    sp500_monthly["observation_date"].min(),
                    "to",
                    sp500_monthly["observation_date"].max(),
                )

                # Merge the combined indicators with S&P500 data
                merged_data = merge_datasets(indicators_merged, sp500_monthly)

                if merged_data is not None:
                    # Save the merged dataset
                    merged_data.to_csv("data/processed/merged_data.csv", index=False)
                    print("Merged data saved to data/processed/merged_data.csv")

                    # Display information about the merged dataset
                    print("\nMerged dataset information:")
                    print(merged_data.info())
                    print("\nFirst 5 rows of merged data:")
                    print(merged_data.head())
                else:
                    print("ERROR: Failed to merge datasets.")
            else:
                print("ERROR: No indicator data available for merging.")
        else:
            print("ERROR: Failed to process S&P 500 data to monthly frequency.")
    else:
        print("ERROR: Failed to retrieve S&P 500 data.")
