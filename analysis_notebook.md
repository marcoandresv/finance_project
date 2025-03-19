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
# Introduction
as;llkdjfa;lskdjfl;aksjdflk;asjd


# Data Overview


# Data Loading and Processing
## Imports and loading datasets

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'data/processed/merged_data.csv'
df = pd.read_csv(data_path, parse_dates=['observation_date'])
```

Displaying the overall structure of the dataset

```py

display(df.head())
display(df.info())

```
Handling missing/incorrect values:
```py
display(df.isnull().sum())
display(df.dtypes)




```

# Exploratory Data Analysis (EDA)

trdns

```py
plt.figure(figsize=(12,6))
for column in df.columns[1:]:
    plt.plot(df["observation_date"], df[column], label=column)
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Economic Indicators Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.show()

```

heatmap
```py
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), annot= True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Econ Indictarsand S&P50O')
plt.show()

```

# Key Insights


asdfjasldkfjsa
