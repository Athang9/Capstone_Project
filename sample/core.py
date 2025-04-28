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
    normalize_columns,
    plot_forecast,
    smooth_forecast,
    test_airline_performance_by_range,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def run_forecasting(df):
    forecast_horizon = 8
    metrics = {
        "Passengers": [
            "Legacy_Passengers",
            "LCC_Passengers",
            "Regional_Passengers",
        ],
        "Net Income": [
            "Legacy_Net_Income",
            "LCC_Net_Income",
            "Regional_Net_Income",
        ],
        "Revenue": ["Legacy_Revenue", "LCC_Revenue", "Regional_Revenue"],
    }

    results = {}

    for metric_name, cols in metrics.items():
        for group, col in zip(["Legacy", "LCC", "Regional"], cols):
            fitted, forecast, model_fit = smooth_forecast(
                df[col], metric_name, forecast_horizon
            )
            mae = mean_absolute_error(df[col], fitted)
            rmse = mean_squared_error(df[col], fitted, squared=False)
            results[(metric_name, group)] = {
                "Fitted": fitted,
                "Forecast": forecast,
                "MAE": mae,
                "RMSE": rmse,
            }

    for metric_name in metrics.keys():
        plot_forecast(results, metric_name)

    print("\nEvaluation Metrics (Lower is better):\n")
    for metric_name in metrics.keys():
        print(f"--- {metric_name} ---")
        for group in ["Legacy", "LCC", "Regional"]:
            mae = results[(metric_name, group)]["MAE"]
            rmse = results[(metric_name, group)]["RMSE"]
            print(f"{group}: MAE={mae:.2f}, RMSE={rmse:.2f}")
        print()

    print("\nEvaluation Metrics (MAPE - Lower % is better):\n")
    for metric_name in metrics.keys():
        print(f"--- {metric_name} ---")
        for group in ["Legacy", "LCC", "Regional"]:
            actual = df[
                metrics[metric_name][
                    ["Legacy", "LCC", "Regional"].index(group)
                ]
            ]
            fitted = results[(metric_name, group)]["Fitted"]
            mape = calculate_mape(actual, fitted)
            print(f"{group}: MAPE={mape:.2f}%")
        print()

    forecast_table = {}
    for metric_name in metrics.keys():
        for group in ["Legacy", "LCC", "Regional"]:
            forecast_table[f"{group}_{metric_name}"] = results[
                (metric_name, group)
            ]["Forecast"]

    forecast_df = pd.DataFrame(forecast_table)
    print("\nForecasted Values for 2024–2025:\n")
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

    df["Month"] = df["Quarter"].map({"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10})
    df["Date_temp"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.set_index("Date_temp")
    df.index.name = "Date"
    df = df.sort_index()

    legacy_cols = [
        "AMERICAN_PASSENGER",
        "DELTA_PASSENGER",
        "UNITED_PASSENGER",
        "SOUTHWEST_PASSENGER",
    ]
    lcc_cols = [
        "FRONTIER_PASSENGER",
        "ALLEGIANT_PASSENGER",
        "SPIRIT_PASSENGER",
        "SUN_COUNTRY_PASSENGER",
        "JETBLUE_PASSENGER",
    ]
    regional_cols = [
        "ALASKA_PASSENGER",
        "HAWAIIN_PASSENGER",
        "SKYWEST_PASSENGER",
    ]

    legacy_net_cols = [
        "AMERICAN_AIRLINE_NET_INCOME",
        "DELTA_AIRLINE_NET_INCOME",
        "UNITED_AIRLINE_NET_INCOME",
        "SOUTHWEST_AIRLINE_NET_INCOME",
    ]
    lcc_net_cols = [
        "FRONTIER_NET_INCOME",
        "ALLEGIANT_NET_INCOME",
        "SPIRIT_NET_INCOME",
        "SUN_COUNTRY_NET_INCOME",
        "JETBLUE_NET_INCOME",
    ]
    regional_net_cols = [
        "ALASKA_NET_INCOME",
        "HAWAIIN_NET_INCOME",
        "SKYWEST_NET_INCOME",
    ]

    legacy_rev_cols = [
        "AMERICAN_AIRLINE_OPERATING_REVENUE",
        "DELTA_AIR_LINE_OPERATING_REVENUE",
        "UNITED_AIRLINE_OPERATING_REVENUE",
        "SOUTHWEST_AIRLINE_OPERATING_REVENUE",
    ]
    lcc_rev_cols = [
        "FRONTIER_OPERATING_REVENUE",
        "ALLEGIANT_OPERATING_REVENUE",
        "SPIRIT_OPERATING_REVENUE",
        "SUN_COUNTRY_OPERATING_REVENUE",
        "JETBLUE_OPERATING_REVENUE",
    ]
    regional_rev_cols = [
        "ALASKA_OPERTING_REVENUE",
        "HAWAIIN_OPERATING_REVENUE",
        "SKYWEST_OPERATING_REVENUE",
    ]

    df["Legacy_Passengers"] = df[legacy_cols].sum(axis=1)
    df["LCC_Passengers"] = df[lcc_cols].sum(axis=1)
    df["Regional_Passengers"] = df[regional_cols].sum(axis=1)

    df["Legacy_Net_Income"] = df[legacy_net_cols].sum(axis=1)
    df["LCC_Net_Income"] = df[lcc_net_cols].sum(axis=1)
    df["Regional_Net_Income"] = df[regional_net_cols].sum(axis=1)

    df["Legacy_Revenue"] = df[legacy_rev_cols].sum(axis=1)
    df["LCC_Revenue"] = df[lcc_rev_cols].sum(axis=1)
    df["Regional_Revenue"] = df[regional_rev_cols].sum(axis=1)

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
    run_forecasting(df)


if __name__ == "__main__":
    run_analysis()
