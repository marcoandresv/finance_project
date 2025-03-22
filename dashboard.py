import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/processed/merged_data.csv", parse_dates=["observation_date"])
df.set_index("observation_date", inplace=True)

# Normalize data for comparison
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df_norm["observation_date"] = df.index

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Economic Indicators Dashboard"),
        html.Label("Select Indicator:"),
        dcc.Dropdown(
            id="indicator-dropdown",
            options=[
                {"label": col, "value": col} for col in df.columns if col != "^GSPC"
            ],
            value="UNRATE",
            clearable=False,
        ),
        dcc.Graph(id="time-series-graph"),
        dcc.Graph(id="scatter-plot"),
    ]
)


@app.callback(
    Output("time-series-graph", "figure"), Input("indicator-dropdown", "value")
)
def update_time_series(indicator):
    fig = px.line(
        df_norm,
        x="observation_date",
        y=[indicator, "^GSPC"],
        title=f"Normalized {indicator} vs S&P 500",
    )
    return fig


@app.callback(Output("scatter-plot", "figure"), Input("indicator-dropdown", "value"))
def update_scatter(indicator):
    df_pct = df.pct_change().dropna()
    fig = px.scatter(
        df_pct,
        x=indicator,
        y="^GSPC",
        trendline="ols",
        title=f"{indicator} vs S&P 500 (Percentage Change)",
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)
