import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def classify_airline(name):
    low_cost = ["SPIRIT", "FRONTIER", "ALLEGIANT", "JETBLUE"]
    regional = ["SKYWEST", "HAWAIIN", "SUN_COUNTRY"]
    legacy = ["AMERICAN", "DELTA", "UNITED", "SOUTHWEST", "ALASKA"]

    if name in low_cost:
        return "Low Cost"
    elif name in regional:
        return "Regional"
    elif name in legacy:
        return "Legacy"
    return "Unknown"


def extract_airlines(df):
    """Extract unique airline names based on known suffixes."""
    keywords = ["PASSENGER", "NET_INCOME", "OPERATING_REVENUE"]
    airlines = set()
    for col in df.columns:
        for key in keywords:
            if key in col:
                name = (
                    col.replace(f"_{key}", "")
                    .replace("_AIRLINE", "")
                    .replace("_AIR", "")
                    .upper()
                )
                airlines.add(name)
    return sorted(airlines)


def classify_airlines_df(df):
    airlines = extract_airlines(df)
    return pd.DataFrame(
        {"Airline": airlines, "Type": [classify_airline(a) for a in airlines]}
    )


def get_airlines_by_cluster(cluster_map):
    return [airline for group in cluster_map.values() for airline in group]


def extract_columns(df, airlines, keyword):
    return [
        col
        for col in df.columns
        if any(airline in col and keyword in col for airline in airlines)
    ]


def normalize_columns(df, columns):
    df_norm = df.copy()
    for col in columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        df_norm[col] = (
            (df_norm[col] - min_val) / (max_val - min_val)
            if max_val != min_val
            else 0
        )
    return df_norm


def melt_for_plotting(df, id_vars, var_name, value_name):
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)


def calc_recovery_rate(df, year_before, year_after, net_income_cols):
    pre = df[df["Year"] == year_before][net_income_cols].mean()
    post = df[df["Year"] == year_after][net_income_cols].mean()
    return ((post - pre) / pre) * 100


def calculate_performance(df, airlines):
    summary = []
    for airline in airlines:
        p_col = next(
            (c for c in df.columns if "PASSENGER" in c and airline in c), None
        )
        r_col = next(
            (
                c
                for c in df.columns
                if "OPERATING_REVENUE" in c and airline in c
            ),
            None,
        )
        i_col = next(
            (c for c in df.columns if "NET_INCOME" in c and airline in c), None
        )

        if p_col and r_col and i_col:
            summary.append(
                {
                    "Airline": airline,
                    "Total Revenue": df[r_col].sum(),
                    "Total Net Income": df[i_col].sum(),
                    "Avg Passenger Growth Rate": df[p_col].pct_change().mean(),
                }
            )
    return pd.DataFrame(summary)


def normalize_performance(perf_df):
    for metric in [
        "Total Revenue",
        "Total Net Income",
        "Avg Passenger Growth Rate",
    ]:
        min_val = perf_df[metric].min()
        max_val = perf_df[metric].max()
        perf_df[f"{metric} (Normalized)"] = (perf_df[metric] - min_val) / (
            max_val - min_val
        )
    perf_df["Overall Score"] = perf_df[
        [col for col in perf_df.columns if "Normalized" in col]
    ].mean(axis=1)
    return perf_df.sort_values(
        by="Overall Score", ascending=False
    ).reset_index(drop=True)


def test_airline_performance_by_range(df, start_year, end_year):
    if not (2003 <= start_year <= 2023 and 2003 <= end_year <= 2023):
        print("Please enter a year range between 2003 and 2023.")
        return None

    airlines = extract_airlines(df)
    performance_summary = []
    df_range = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

    for airline in airlines:
        p_col = next(
            (
                col
                for col in df.columns
                if "PASSENGER" in col and airline in col
            ),
            None,
        )
        r_col = next(
            (
                col
                for col in df.columns
                if "OPERATING_REVENUE" in col and airline in col
            ),
            None,
        )
        i_col = next(
            (
                col
                for col in df.columns
                if "NET_INCOME" in col and airline in col
            ),
            None,
        )

        if p_col and r_col and i_col:
            performance_summary.append(
                {
                    "Airline": airline,
                    "Revenue": df_range[r_col].sum(),
                    "Net Income": df_range[i_col].sum(),
                    "Passengers": df_range[p_col].sum(),
                }
            )

    year_df = pd.DataFrame(performance_summary)
    for metric in ["Revenue", "Net Income", "Passengers"]:
        min_val = year_df[metric].min()
        max_val = year_df[metric].max()
        year_df[f"{metric} (Norm)"] = (year_df[metric] - min_val) / (
            max_val - min_val
        )

    year_df["Performance Score"] = year_df[
        [col for col in year_df.columns if "Norm" in col]
    ].mean(axis=1)
    return year_df.sort_values(
        by="Performance Score", ascending=False
    ).reset_index(drop=True)


def smooth_forecast(series, metric_name, forecast_horizon=8):
    """Fits Exponential Smoothing Model and returns fitted values and forecast."""
    if metric_name == "Net Income":
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=4
        )
    else:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="mul", seasonal_periods=4
        )
    model_fit = model.fit()
    fitted = model_fit.fittedvalues
    forecast = model_fit.forecast(forecast_horizon)
    return fitted, forecast, model_fit


def plot_forecast(results, metric_name):
    """Plots Actual vs Forecast for a given metric."""
    fig = go.Figure()

    for group in ["Legacy", "LCC", "Regional"]:
        fitted = results[(metric_name, group)]["Fitted"]
        forecast = results[(metric_name, group)]["Forecast"]

        fig.add_trace(
            go.Scatter(
                x=fitted.index,
                y=fitted,
                mode="lines",
                name=f"{group} (Actual)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=forecast,
                mode="lines",
                name=f"{group} (Forecast)",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{metric_name} Forecast by Airline Group",
        xaxis_title="Date",
        yaxis_title=metric_name,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        width=1000,
        height=600,
    )
    fig.show()


def calculate_mape(actual, fitted):
    """Calculates Mean Absolute Percentage Error."""
    return np.mean(np.abs((actual - fitted) / actual)) * 100
