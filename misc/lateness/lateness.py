import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

LATENESS_FILE = pathlib.Path("lateness.ods").resolve()


def preprocess_data() -> pd.DataFrame:
    "Preprocess the data for display."
    data = pd.read_excel(LATENESS_FILE)
    data["start_time"] = abs(data["start_time"])
    return data


def lateness_mean(data) -> go.Figure:
    """Return a figure of the mean lateness of each teacher."""
    av_late = (
        data.groupby("teacher")
        .mean()
        .sort_values(
            by="start_time",
            ascending=False,
        )
    )
    return px.bar(
        data_frame=av_late,
        y="start_time",
        color="start_time",
        color_continuous_scale="Bluered",
        barmode="group",
        title="Mean lateness",
    )


def lateness_sum(data) -> go.Figure:
    sum_late = (
        data.groupby("teacher")
        .sum()
        .sort_values(
            "start_time",
            ascending=False,
        )
    )
    return px.bar(
        sum_late,
        y="start_time",
        color="start_time",
        color_continuous_scale="Bluered",
        barmode="group",
        title="Cumulated lateness",
    )


def lateness_median(data) -> go.Figure:
    median_late = (
        data.groupby("teacher")
        .median()
        .sort_values(
            "start_time",
            ascending=False,
        )
    )
    return px.bar(
        median_late,
        y="start_time",
        color="start_time",
        color_continuous_scale="Bluered",
        barmode="group",
        title="Median lateness",
    )


if __name__ == "__main__":
    data = preprocess_data()

    # Mean lateness
    fig = lateness_mean(data)
    fig.write_image("images/lateness_mean_test.png")

    # Summed lateness
    fig = lateness_sum(data)
    fig.write_image("images/lateness_sum.png")

    # Median lateness
    fig = lateness_median(data)
    fig.write_image("images/lateness_median.png")
