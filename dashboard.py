import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("data/processed/merged_data.csv", parse_dates=["observation_date"])
df.set_index("observation_date", inplace=True)

# Normalize data
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df_norm["observation_date"] = df.index

# Calculate percentage changes
df_pct = df.pct_change().dropna()
df_pct["observation_date"] = df_pct.index

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Economic Indicators Dashboard"),
        dcc.Tabs(
            id="tabs",
            value="raw",
            children=[
                dcc.Tab(label="Raw Data", value="raw"),
                dcc.Tab(label="Normalized", value="normalized"),
                dcc.Tab(label="% Changes", value="pct_changes"),
                dcc.Tab(label="Correlations", value="correlations"),
                dcc.Tab(label="Outliers", value="outliers"),
            ],
        ),
        html.Div(id="tab-content"),
    ]
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def update_tab(selected_tab):
    if selected_tab == "raw":
        fig = px.line(df, x=df.index, y=df.columns, title="Raw Economic Indicators")
        insights = "This tab displays the raw data for all economic indicators. Trends can be observed over time."
    elif selected_tab == "normalized":
        fig = px.line(
            df_norm,
            x="observation_date",
            y=df_norm.columns[:-1],
            title="Normalized Economic Indicators",
        )
        insights = "This tab displays normalized data, allowing for comparison across different scales."
    elif selected_tab == "pct_changes":
        fig = px.line(
            df_pct,
            x="observation_date",
            y=df_pct.columns[:-1],
            title="Percentage Changes Over Time",
        )
        insights = "This tab shows percentage changes to analyze trends and volatility in indicators."
    elif selected_tab == "correlations":
        fig = px.imshow(
            df.corr(), text_auto=True, title="Correlation Matrix of Indicators"
        )
        insights = "This tab provides a correlation matrix, highlighting relationships between economic indicators."
    elif selected_tab == "outliers":
        z_scores = (df - df.mean()) / df.std()
        outliers = (z_scores.abs() > 3).sum()
        fig = px.bar(x=df.columns, y=outliers, title="Outliers in Economic Indicators")
        insights = "This tab highlights outliers using Z-score analysis. Outliers could indicate anomalies or economic shocks."

    return html.Div(
        [
            dcc.Graph(figure=fig),
            html.P(insights, style={"fontSize": "16px", "marginTop": "10px"}),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)
