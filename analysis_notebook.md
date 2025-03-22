---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: ''
    name: ''
---

# Understanding the dataset

**UNRATE** | _Unemployment Rate_
↳ Perentage of the laber force that is unemployed

**CPIAUCSL** | _Consumer Price Index for All Urban Consumers_
↳ Measure of inflation

**INDPRO** | _Industrial Production Index_
↳ Output of manufacturing, mining, and utilities sectors

**FEDFUNDS** | _Effective Federal Funds Rate_
↳ Interest rate at which banks lend to each other overnight

**^GSPC** | _S&P500 Index Value_
↳ The value of the top 500 US companies


# Imports and Data Load

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr

# Load the data
data_path = 'data/processed/merged_data.csv'
df = pd.read_csv(data_path, parse_dates=['observation_date'])

```

```python
plt.figure(figsize=(12,6))
for column in df.columns[1:]:
    plt.plot(df["observation_date"], df[column], label=column)
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Economic Indicators Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('visualization/initial_graph_indicators.png')
plt.show()

```

# Scale and Range Analysis

The economic indicators operate in a vastly different scales:
- **UNRATE**: Range 3.4-14.8% (mean 4.97%)
- **CPIAUCSL**: Range 255.8-319.8 (mean 290.18)
- **INDPRO**: Range 84.7-104.2 (mean 100.6)
- **FEDFUNDS**: Range 0.05-5.33% (mean 2.55%)
- **S&P 500**: Range 2,584.6-6,040.5 (mean 4,368.17)

The use of a MinMaxScaler is justified to properly visualize the data in the same scale range.


```python

# 1. IMPROVED VISUALIZATION: NORMALIZE DATA

# Set the date as index
df.set_index('observation_date', inplace=True)

# Create a copy of the original data for reference
df_original = df.copy()

# Create normalized versions of all variables (0-1 scale) for better visualization
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns,
    index=df.index
)

# 2. CALCULATE PERCENTAGE CHANGES
# This helps to see how variables change over time rather than their absolute values
df_pct_change = df.pct_change() * 100
df_pct_change = df_pct_change.dropna()

```

```python
# 3. CALCULATE ROLLING CORRELATIONS
# This shows how the relationship between variables changes over time
window_size = 12  # 12-month rolling window
rolling_corr = df_pct_change.rolling(window=window_size).corr(pairwise=True)

# Extract S&P 500 correlation with each indicator over time
spx_column = '^GSPC'
indicators = [col for col in df.columns if col != spx_column]
rolling_corr_with_spx = pd.DataFrame(index=df_pct_change.index[window_size-1:])

for indicator in indicators:
    indicator_corr = []
    for i in range(window_size-1, len(df_pct_change)):
        correlation = df_pct_change.iloc[i-window_size+1:i+1][[indicator, spx_column]].corr().iloc[0, 1]
        indicator_corr.append(correlation)
    rolling_corr_with_spx[indicator] = indicator_corr

```

```python
# 4. LAG ANALYSIS
# Create lagged versions of variables to see if past values of indicators predict future S&P 500 values
max_lag = 6  # Testing up to 6 months of lag
df_lag = pd.DataFrame(index=df_pct_change.index)
df_lag[spx_column] = df_pct_change[spx_column]

for indicator in indicators:
    for lag in range(1, max_lag + 1):
        df_lag[f"{indicator}_lag{lag}"] = df_pct_change[indicator].shift(lag)

df_lag = df_lag.dropna()

```

```python
# 5. CORRELATION ANALYSIS
# Calculate Pearson correlation coefficients
correlation_matrix = df_pct_change.corr()

# Calculate correlations with p-values
correlation_results = {}
for indicator in indicators:
    corr, p_value = pearsonr(df_pct_change[indicator].dropna(), df_pct_change[spx_column].dropna())
    correlation_results[indicator] = {'correlation': corr, 'p_value': p_value}

```

```python
# 6. GRANGER CAUSALITY TEST
# Test if indicators "Granger cause" S&P 500 movements
granger_results = {}
for indicator in indicators:
    data = df_pct_change[[indicator, spx_column]].dropna()
    result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    
    # Extract the p-values for each lag
    p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}
    granger_results[indicator] = p_values

```

```python
# 7. VISUALIZATIONS

# Plot 1: Normalized Data - Better for comparing trends
plt.figure(figsize=(14, 7))
for column in df_normalized.columns:
    plt.plot(df_normalized.index, df_normalized[column], label=column)
plt.title('Normalized Economic Indicators and S&P 500 (0-1 scale)')
plt.xlabel('Date')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization/normalized_indicators.png')
plt.show()

```

```python
# Plot 2: Percentage Changes - Better for volatility comparison
plt.figure(figsize=(14, 7))
for column in df_pct_change.columns:
    plt.plot(df_pct_change.index, df_pct_change[column], label=column)
plt.title('Monthly Percentage Changes in Economic Indicators and S&P 500')
plt.xlabel('Date')
plt.ylabel('% Change')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization/pct_change_indicators.png')
plt.show()

```

```python
# Plot 3: Rolling Correlations with S&P 500
plt.figure(figsize=(14, 7))
for indicator in indicators:
    plt.plot(rolling_corr_with_spx.index, rolling_corr_with_spx[indicator], label=f'{indicator} vs S&P 500')
plt.title(f'{window_size}-Month Rolling Correlation with S&P 500')
plt.xlabel('Date')
plt.ylabel('Correlation Coefficient')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization/rolling_correlations.png')
plt.show()

```

```python
# Plot 4: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Economic Indicators and S&P 500')
plt.tight_layout()
plt.savefig('visualization/correlation_heatmap.png')
plt.show()

```

```python
# Plot 5: Scatter Plots with Regression Lines
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, indicator in enumerate(indicators):
    ax = axes[i]
    sns.regplot(x=indicator, y=spx_column, data=df_pct_change, ax=ax, line_kws={"color": "red"})
    ax.set_title(f'{indicator} vs S&P 500 (% Change)')
    ax.grid(True, alpha=0.3)
    
    # Add correlation coefficient and p-value to the plot
    corr = correlation_results[indicator]['correlation']
    p_value = correlation_results[indicator]['p_value']
    ax.annotate(f'r = {corr:.2f}, p = {p_value:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('visualization/scatter_plots.png')
plt.show()

```

```python
# Plot 6: Granger Causality Test Results
plt.figure(figsize=(14, 7))
for indicator in indicators:
    lags = list(granger_results[indicator].keys())
    p_values = list(granger_results[indicator].values())
    plt.plot(lags, p_values, marker='o', label=indicator)

plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% Significance Level')
plt.title('Granger Causality Test p-values (Lower = Stronger Evidence)')
plt.xlabel('Lag (Months)')
plt.ylabel('p-value')
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualization/granger_causality.png')
plt.show()

```

```python
# 8. FINAL ANALYSIS OUTPUT

# Summary of key correlations with p-values
print("\n===== CORRELATION ANALYSIS =====")
for indicator, result in correlation_results.items():
    corr = result['correlation']
    p_value = result['p_value']
    significance = "Significant" if p_value < 0.05 else "Not significant"
    print(f"{indicator} correlation with S&P 500: {corr:.3f} (p-value: {p_value:.4f}) - {significance}")

# Summary of Granger causality tests
print("\n===== GRANGER CAUSALITY ANALYSIS =====")
print("Does the indicator help predict S&P 500 movements?")
for indicator, p_values in granger_results.items():
    significant_lags = [lag for lag, p_value in p_values.items() if p_value < 0.05]
    if significant_lags:
        print(f"{indicator} Granger-causes S&P 500 at lags: {significant_lags}")
    else:
        print(f"{indicator} does not Granger-cause S&P 500 at any tested lag")

# Descriptive statistics
print("\n===== DESCRIPTIVE STATISTICS =====")
print("Original Data:")
print(df_original.describe())

print("\nPercentage Changes:")
print(df_pct_change.describe())



```
