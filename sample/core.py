import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from helpers import (
    calc_recovery_rate,
    calculate_mape,
    extract_columns,
    get_airlines_by_cluster,
    melt_for_plotting,
    monte_carlo_forecast,
    normalize_columns,
    perform_t_test,
    plot_forecast,
    smooth_forecast,
    test_airline_performance_by_range,
)
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


def plot_passenger_growth_cluster(df, cluster_map, cluster_id):
    airlines = cluster_map[cluster_id]
    cols = extract_columns(df, airlines, "PASSENGER")
    df_yearly = df[["Year"] + cols].groupby("Year")[cols].mean().reset_index()

    fig = go.Figure()
    for col in cols:
        fig.add_trace(
            go.Scatter(
                x=df_yearly["Year"],
                y=df_yearly[col],
                mode="lines+markers",
                name=col.replace("_PASSENGER", ""),
            )
        )

    # Add vertical lines and annotations for economic events
    max_y = df_yearly[cols].max().max()
    fig.add_vline(x=2008, line_width=2, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=2008,
        y=max_y,
        text="2008 Financial Crisis",
        showarrow=False,
        yshift=30,
        font=dict(color="red"),
    )

    fig.add_vline(x=2020, line_width=2, line_dash="dash", line_color="purple")
    fig.add_annotation(
        x=2020,
        y=max_y * 0.75,
        text="COVID-19 Pandemic",
        showarrow=False,
        yshift=30,
        font=dict(color="purple"),
    )

    fig.update_layout(
        title=f"Passenger Growth Over Time by Cluster {cluster_id}",
        xaxis_title="Year",
        yaxis_title="Avg Passengers",
        template="plotly_white",
    )
    fig.show()
    fig.write_html("Passenger Growth Over Time by Cluster {cluster_id}")


def plot_all_airlines_normalized(df, airlines):
    cols = extract_columns(df, airlines, "PASSENGER")
    df_yearly = df[["Year"] + cols].groupby("Year")[cols].mean().reset_index()
    df_norm = normalize_columns(df_yearly, cols)
    melted = melt_for_plotting(
        df_norm, "Year", "Airline", "Normalized_Passenger"
    )
    melted["Airline"] = melted["Airline"].str.replace("_PASSENGER", "")

    fig = px.line(
        melted,
        x="Year",
        y="Normalized_Passenger",
        color="Airline",
        title="Normalized Passenger Growth Over Time by Individual Airline",
        template="plotly_white",
    )

    # Add vertical lines and annotations
    max_y = melted["Normalized_Passenger"].max()
    fig.add_vline(x=2008, line_width=2, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=2008,
        y=max_y,
        text="2008 Financial Crisis",
        showarrow=False,
        yshift=30,
        font=dict(color="red"),
    )

    fig.add_vline(x=2020, line_width=2, line_dash="dash", line_color="purple")
    fig.add_annotation(
        x=2020,
        y=max_y * 0.75,
        text="COVID-19 Pandemic",
        showarrow=False,
        yshift=30,
        font=dict(color="purple"),
    )

    fig.show()
    fig.write_html(
        "Normalized Passenger Growth Over Time by Individual Airline"
    )


def plot_market_share_clusters(df, cluster_map):
    df_clustered = df[["Year"]].copy()
    for c, airlines in cluster_map.items():
        df_clustered[f"Cluster {c}"] = df[
            extract_columns(df, airlines, "PASSENGER")
        ].sum(axis=1)
    df_clustered = df_clustered.groupby("Year").sum().reset_index()

    fig = px.area(
        df_clustered,
        x="Year",
        y=[f"Cluster {i}" for i in range(3)],
        title="Passenger Market Share Over Time by Cluster (Legacy, Low-cost, Regional)",
        template="plotly_white",
    )
    fig.show()
    fig.write_html(
        "Passenger Market Share Over Time by Cluster (Legacy, Low-cost, Regional)"
    )


def plot_cluster_net_income(df, cluster_map):
    df_income = df[["Year"]].copy()
    for c, airlines in cluster_map.items():
        df_income[f"Cluster {c}"] = df[
            extract_columns(df, airlines, "NET_INCOME")
        ].sum(axis=1)
    df_income = df_income.groupby("Year").sum().reset_index()

    fig = px.area(
        df_income,
        x="Year",
        y=[f"Cluster {i}" for i in range(3)],
        title="Net Income Over Time by Cluster (Legacy, Low-cost, Regional)",
        template="plotly_white",
    )
    fig.show()
    fig.write_html(
        "Net Income Over Time by Cluster (Legacy, Low-cost, Regional)"
    )


def plot_cluster_operating_revenue(df, cluster_map):
    df_revenue = df[["Year"]].copy()
    for c, airlines in cluster_map.items():
        df_revenue[f"Cluster {c}"] = df[
            extract_columns(df, airlines, "OPERATING_REVENUE")
        ].sum(axis=1)
    df_revenue = df_revenue.groupby("Year").sum().reset_index()

    fig = px.area(
        df_revenue,
        x="Year",
        y=[f"Cluster {i}" for i in range(3)],
        title="Operating Revenue Over Time by Cluster (Legacy, Low-cost, Regional)",
        template="plotly_white",
    )
    fig.show()
    fig.write_html(
        "Operating Revenue Over Time by Cluster (Legacy, Low-cost, Regional)"
    )


def plot_net_income_airlines(df, cluster_map):
    individual_income = {}

    for cluster, airlines in cluster_map.items():
        for airline in airlines:
            income_col = next(
                (
                    col
                    for col in df.columns
                    if "NET_INCOME" in col and airline in col
                ),
                None,
            )
            if income_col:
                airline_data = (
                    df[["Year", income_col]]
                    .groupby("Year")
                    .sum()
                    .reset_index()
                )
                airline_data.columns = ["Year", "Net Income"]
                airline_data["Airline"] = airline
                individual_income[airline] = airline_data

    all_airline_income = pd.concat(individual_income.values())

    fig = px.area(
        all_airline_income,
        x="Year",
        y="Net Income",
        color="Airline",
        title="Net Income Trends by Individual Airline",
        template="plotly_white",
        labels={"Net Income": "Net Income (USD)"},
    )

    fig.update_layout(
        yaxis_title="Net Income (USD)",
        hovermode="x unified",
        legend_title="Airline",
    )

    fig.show()
    fig.write_html("Net Income Trends by Individual Airline")


def plot_operating_revenue_airlines(df, cluster_map):
    individual_revenue = {}

    for cluster, airlines in cluster_map.items():
        for airline in airlines:
            revenue_col = next(
                (
                    col
                    for col in df.columns
                    if "OPERATING_REVENUE" in col and airline in col
                ),
                None,
            )
            if revenue_col:
                airline_data = (
                    df[["Year", revenue_col]]
                    .groupby("Year")
                    .sum()
                    .reset_index()
                )
                airline_data.columns = ["Year", "Operating Revenue"]
                airline_data["Airline"] = airline
                individual_revenue[airline] = airline_data

    all_airline_revenue = pd.concat(individual_revenue.values())

    fig = px.area(
        all_airline_revenue,
        x="Year",
        y="Operating Revenue",
        color="Airline",
        title="Operating Revenue Trends by Individual Airline",
        template="plotly_white",
        labels={"Operating Revenue": "Operating Revenue (USD)"},
    )

    fig.update_layout(
        yaxis_title="Operating Revenue (USD)",
        hovermode="x unified",
        legend_title="Airline",
    )

    fig.show()
    fig.write_html("Operating Revenue Trends by Individual Airline")


def plot_airline_performance_index(df, start_year, end_year):
    score_df = test_airline_performance_by_range(df, start_year, end_year)
    if score_df is not None:
        fig = px.bar(
            score_df,
            x="Airline",
            y="Performance Score",
            color="Airline",
            title=f"Performance Index by Individual Airline ({start_year}-{end_year})",
            text="Performance Score",
            template="plotly_white",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.show()
        fig.write_html("Performance Index by Individual Airline (2003 - 2023)")


def plot_market_share_volatility(df):
    cols = [
        col
        for col in df.columns
        if "PASSENGER" in col or "OPERATING_REVENUE" in col
    ]
    df_market = df[cols].fillna(0)
    share_df = df_market.div(df_market.sum(axis=1), axis=0)
    vol = share_df.std().sort_values(ascending=False).reset_index()
    vol.columns = ["Metric", "Volatility"]
    fig = px.bar(
        vol,
        x="Volatility",
        y="Metric",
        orientation="h",
        title="Market Share Volatility",
        template="plotly_white",
    )
    fig.show()
    fig.write_html("Market Share Volatility")


def plot_financial_resilience(df):
    groups = {
        "Group1_Net_Income": [
            "AMERICAN_AIRLINE_NET_INCOME",
            "DELTA_AIRLINE_NET_INCOME",
            "SOUTHWEST_AIRLINE_NET_INCOME",
            "UNITED_AIRLINE_NET_INCOME",
        ],
        "Group2_Net_Income": [
            "FRONTIER_NET_INCOME",
            "ALLEGIANT_NET_INCOME",
            "SPIRIT_NET_INCOME",
            "JETBLUE_NET_INCOME",
        ],
        "Group3_Net_Income": [
            "SUN_COUNTRY_NET_INCOME",
            "ALASKA_NET_INCOME",
            "HAWAIIN_NET_INCOME",
            "SKYWEST_NET_INCOME",
        ],
    }

    # Calculate totals per year for each group
    for name, cols in groups.items():
        df[name] = df[cols].sum(axis=1, min_count=1)

    pre_crisis_2007 = df[df["Year"] == 2007][list(groups.keys())].mean()
    recovery_2010 = df[df["Year"] == 2010][list(groups.keys())].mean()
    pre_crisis_2018 = df[df["Year"] == 2018][list(groups.keys())].mean()
    recovery_2022 = df[df["Year"] == 2022][list(groups.keys())].mean()

    recovery_rate_2008_2010 = (
        (recovery_2010 - pre_crisis_2007) / pre_crisis_2007
    ) * 100
    recovery_rate_2019_2022 = (
        (recovery_2022 - pre_crisis_2018) / pre_crisis_2018
    ) * 100

    recovery_df = pd.DataFrame(
        {
            "Group": [
                "Major Airlines",
                "Low-Cost Carriers",
                "Regional Airlines",
            ],
            "2008-2010": recovery_rate_2008_2010.values,
            "2019-2022": recovery_rate_2019_2022.values,
        }
    )

    recovery_long = recovery_df.melt(
        id_vars="Group", var_name="Period", value_name="Recovery Rate"
    )

    fig = px.bar(
        recovery_long,
        x="Group",
        y="Recovery Rate",
        color="Period",
        barmode="group",
        title="Financial Resilience Comparison: 2008-2010 vs 2019-2022",
        labels={
            "Group": "Airline Group",
            "Recovery Rate": "Recovery Rate (%)",
        },
        template="plotly_white",
    )

    fig.update_layout(
        xaxis_title="Airline Group",
        yaxis_title="Recovery Rate (%)",
        legend_title="Period",
        height=500,
    )

    fig.show()
    fig.write_html("Financial Resilience Comparison: 2008-2010 vs 2019-2022")


def smooth_forecast(series, steps=16):
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=4,
        use_boxcox=False,
    )
    model_fit = model.fit(optimized=True)
    forecast = model_fit.forecast(steps)
    fitted = model_fit.fittedvalues
    fitted.index = series.index
    return fitted, forecast, model_fit


def plot_combined_forecast(results, metric_name):
    fig = go.Figure()
    colors = {"Legacy": "blue", "LCC": "green", "Regional": "red"}

    for group in ["Legacy", "LCC", "Regional"]:
        r = results[(metric_name, group)]
        fitted_dates = r["Fitted"].index
        forecast_len = len(r["Forecast"])
        start_date = fitted_dates[-1] + pd.offsets.QuarterBegin()
        forecast_dates = pd.date_range(
            start=start_date, periods=forecast_len, freq="QS"
        )

        fitted_labels = fitted_dates.to_series().dt.to_period("Q").astype(str)
        forecast_labels = (
            forecast_dates.to_series().dt.to_period("Q").astype(str)
        )

        # Add fitted and forecast lines
        fig.add_trace(
            go.Scatter(
                x=fitted_labels,
                y=r["Fitted"],
                mode="lines",
                name=f"{group} Fitted",
                line=dict(color=colors[group], dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_labels,
                y=r["Forecast"],
                mode="lines",
                name=f"{group} Forecast",
                line=dict(color=colors[group], dash="dot"),
            )
        )

        # Add confidence interval as shaded area
        fig.add_trace(
            go.Scatter(
                x=list(forecast_labels) + list(forecast_labels[::-1]),
                y=list(r["CI_Upper"]) + list(r["CI_Lower"][::-1]),
                fill="toself",
                fillcolor=colors[group]
                .replace("blue", "rgba(0,0,255,0.1)")
                .replace("green", "rgba(0,128,0,0.1)")
                .replace("red", "rgba(255,0,0,0.1)"),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=f"{group} CI",
            )
        )

    fig.update_layout(
        title=f"Forecasting Airline {metric_name} Growth (Legacy, LCC, Regional)",
        xaxis_title="Quarter",
        yaxis_title=metric_name,
        template="plotly_white",
        hovermode="x unified",
    )

    fig.show()


def run_passenger_revenue_forecasting(df):
    forecast_horizon = 16
    metrics = {
        "Passengers": [
            "Legacy_Passengers",
            "LCC_Passengers",
            "Regional_Passengers",
        ],
        "Revenue": ["Legacy_Revenue", "LCC_Revenue", "Regional_Revenue"],
    }

    results = {}

    print("\nEvaluation Metrics (Lower is better):\n")
    for metric_name, cols in metrics.items():
        print(f"--- {metric_name} ---")
        for group, col in zip(["Legacy", "LCC", "Regional"], cols):
            series = df[col]
            fitted, forecast, model_fit = smooth_forecast(
                series, forecast_horizon
            )

            common_index = fitted.index.intersection(series.index)
            actual_trimmed = series.loc[common_index]
            fitted_trimmed = fitted.loc[common_index]

            mae = mean_absolute_error(actual_trimmed, fitted_trimmed)
            rmse = mean_squared_error(
                actual_trimmed, fitted_trimmed, squared=False
            )
            mape = calculate_mape(actual_trimmed, fitted_trimmed)

            p_value = perform_t_test(actual_trimmed, fitted_trimmed)
            residuals = (actual_trimmed - fitted_trimmed).dropna()
            ci_lower, ci_upper = monte_carlo_forecast(forecast, residuals)

            results[(metric_name, group)] = {
                "Fitted": fitted_trimmed,
                "Forecast": forecast,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "p_value": p_value,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
            }

            print(
                f"{group}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, p-value={p_value:.4f}"
            )
        print()

    for metric_name in metrics:
        plot_combined_forecast(results, metric_name)

    forecast_table = {}
    for metric_name in metrics:
        for group in ["Legacy", "LCC", "Regional"]:
            forecast_table[f"{group}_{metric_name}"] = results[
                (metric_name, group)
            ]["Forecast"]

    forecast_df = pd.DataFrame(forecast_table)
    print("\nForecasted Values for 2024–2026:\n")
    print(forecast_df)


# Main controller
def run_analysis():
    url = "https://drive.google.com/uc?export=download&id=1a1aWrDUc3Tdgxy_eMqDe_LRS7ehb-lfo"
    df = pd.read_excel(url)

    cluster_map = {
        0: ["ALASKA", "AMERICAN", "DELTA", "SOUTHWEST", "UNITED"],
        1: ["ALLEGIANT", "FRONTIER", "JETBLUE", "SPIRIT"],
        2: ["SKYWEST", "HAWAIIN", "SUN_COUNTRY"],
    }

    airlines = get_airlines_by_cluster(cluster_map)

    # Build datetime index from Year and Quarter
    df["Month"] = df["Quarter"].map({"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10})
    df["Date_temp"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.set_index("Date_temp")
    df.index.name = "Date"
    df = df.sort_index()

    # Aggregate metrics
    df["Legacy_Passengers"] = df[
        [
            "AMERICAN_PASSENGER",
            "DELTA_PASSENGER",
            "UNITED_PASSENGER",
            "SOUTHWEST_PASSENGER",
        ]
    ].sum(axis=1)
    df["LCC_Passengers"] = df[
        [
            "FRONTIER_PASSENGER",
            "ALLEGIANT_PASSENGER",
            "SPIRIT_PASSENGER",
            "SUN_COUNTRY_PASSENGER",
            "JETBLUE_PASSENGER",
        ]
    ].sum(axis=1)
    df["Regional_Passengers"] = df[
        ["ALASKA_PASSENGER", "HAWAIIN_PASSENGER", "SKYWEST_PASSENGER"]
    ].sum(axis=1)

    df["Legacy_Net_Income"] = df[
        [
            "AMERICAN_AIRLINE_NET_INCOME",
            "DELTA_AIRLINE_NET_INCOME",
            "UNITED_AIRLINE_NET_INCOME",
            "SOUTHWEST_AIRLINE_NET_INCOME",
        ]
    ].sum(axis=1)
    df["LCC_Net_Income"] = df[
        [
            "FRONTIER_NET_INCOME",
            "ALLEGIANT_NET_INCOME",
            "SPIRIT_NET_INCOME",
            "SUN_COUNTRY_NET_INCOME",
            "JETBLUE_NET_INCOME",
        ]
    ].sum(axis=1)
    df["Regional_Net_Income"] = df[
        ["ALASKA_NET_INCOME", "HAWAIIN_NET_INCOME", "SKYWEST_NET_INCOME"]
    ].sum(axis=1)

    df["Legacy_Revenue"] = df[
        [
            "AMERICAN_AIRLINE_OPERATING_REVENUE",
            "DELTA_AIR_LINE_OPERATING_REVENUE",
            "UNITED_AIRLINE_OPERATING_REVENUE",
            "SOUTHWEST_AIRLINE_OPERATING_REVENUE",
        ]
    ].sum(axis=1)
    df["LCC_Revenue"] = df[
        [
            "FRONTIER_OPERATING_REVENUE",
            "ALLEGIANT_OPERATING_REVENUE",
            "SPIRIT_OPERATING_REVENUE",
            "SUN_COUNTRY_OPERATING_REVENUE",
            "JETBLUE_OPERATING_REVENUE",
        ]
    ].sum(axis=1)
    df["Regional_Revenue"] = df[
        [
            "ALASKA_OPERTING_REVENUE",
            "HAWAIIN_OPERATING_REVENUE",
            "SKYWEST_OPERATING_REVENUE",
        ]
    ].sum(axis=1)

    # Visualization section
    plot_passenger_growth_cluster(df, cluster_map, 0)
    plot_passenger_growth_cluster(df, cluster_map, 1)
    plot_passenger_growth_cluster(df, cluster_map, 2)
    plot_all_airlines_normalized(df, airlines)
    plot_market_share_clusters(df, cluster_map)
    plot_cluster_net_income(df, cluster_map)
    plot_cluster_operating_revenue(df, cluster_map)
    plot_net_income_airlines(df, cluster_map)
    plot_operating_revenue_airlines(df, cluster_map)
    plot_airline_performance_index(df, 2003, 2023)
    plot_market_share_volatility(df)
    plot_financial_resilience(df)

    # Forecasting + Evaluation + Interactive plotting
    run_passenger_revenue_forecasting(df)

    print(
        "\nAll plots and forecast evaluations are complete. Interactive plots + p-values + Monte Carlo simulation included.\n"
    )


if __name__ == "__main__":
    run_analysis()
