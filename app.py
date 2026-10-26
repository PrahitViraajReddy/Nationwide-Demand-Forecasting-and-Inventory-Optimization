"""Interactive Dash demo for retail demand forecasting.

Upload a small long-format CSV with columns date,sales and optionally id/item_id/store_id.
The app generates a weekly seasonal-naive forecast from the uploaded observed history.
It does not pretend the M5 dataset or trained models are bundled with the deployment.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

ROOT = Path(__file__).resolve().parent
app = Dash(__name__, title="Retail Demand & Inventory Intelligence")
app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            [
                html.P("RETAIL INTELLIGENCE / FORECAST LAB", className="eyebrow"),
                html.H1("Demand Forecasting & Inventory Optimization"),
                html.P(
                    "Upload observed demand, generate a transparent baseline forecast, "
                    "and inspect the validation error before adding advanced models.",
                    className="subtitle",
                ),
            ],
            className="hero",
        ),
        html.Section(
            [
                html.Div([html.Span("FORECAST METHOD"), html.Strong("Weekly seasonal naive")], className="status-card"),
                html.Div([html.Span("INPUT DATA"), html.Strong("Upload required", id="data-status")], className="status-card"),
                html.Div([html.Span("EVALUATION"), html.Strong("Awaiting data", id="metric-status")], className="status-card"),
            ],
            className="status-grid",
        ),
        html.Section(
            [
                html.H2("1. Load observed demand"),
                html.P(
                    "Upload a small CSV in long format with date and sales columns. Optional series identifiers: "
                    "id, item_id, store_id. One row should represent one series on one date."
                ),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["Drag and drop a CSV here, or ", html.A("choose a file")]),
                    className="upload-box",
                    multiple=False,
                ),
                html.P("Required columns: date, sales · Optional: id, item_id, store_id", className="muted"),
                html.Div(id="upload-message", className="message"),
                html.Div(
                    [
                        html.Label("Series"),
                        dcc.Dropdown(id="series-select", options=[], value=None, clearable=False, placeholder="Upload a CSV first"),
                        html.Label("Forecast horizon (days)"),
                        dcc.Slider(id="horizon", min=7, max=28, step=7, value=28,
                                   marks={7: "7 days", 14: "14 days", 21: "21 days", 28: "28 days"}),
                        html.Button("Generate forecast", id="forecast-button", n_clicks=0, className="primary-button"),
                    ],
                    className="controls",
                ),
            ],
            className="panel",
        ),
        html.Section(
            [
                html.H2("2. Forecast workspace"),
                html.Div(id="forecast-summary", className="summary"),
                dcc.Loading(dcc.Graph(id="forecast-chart", figure=empty_figure(), config={"responsive": True})),
                html.Div(
                    [
                        html.Div([html.Span("Validation WAPE"), html.Strong("—", id="wape-value")], className="metric-card"),
                        html.Div([html.Span("Validation MAE"), html.Strong("—", id="mae-value")], className="metric-card"),
                        html.Div([html.Span("Rows forecast"), html.Strong("0", id="forecast-count")], className="metric-card"),
                    ],
                    className="metric-grid",
                ),
                html.Button("Download forecast CSV", id="download-button", n_clicks=0, className="secondary-button"),
                dcc.Download(id="download-forecast"),
                html.P(
                    "Method note: this is a weekly seasonal-naive benchmark, not LightGBM/Prophet/TFT. "
                    "Validation scores use the final available horizon when enough history exists. "
                    "No model performance is reported until data is uploaded.",
                    className="note",
                ),
            ],
            className="panel",
        ),
        html.Footer("Reproducible retail forecasting · interactive baseline demo"),
        dcc.Store(id="data-store"),
        dcc.Store(id="forecast-store"),
    ],
)


def empty_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", height=390, margin={"l": 25, "r": 20, "t": 35, "b": 25},
        xaxis_title="Date", yaxis_title="Units sold",
        annotations=[{"text": "Upload demand history and generate a forecast", "xref": "paper",
                      "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 15}}],
    )
    return fig


def parse_upload(contents: str) -> pd.DataFrame:
    try:
        _, encoded = contents.split(",", 1)
        raw = base64.b64decode(encoded)
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if not {"date", "sales"}.issubset(frame.columns):
        raise ValueError("CSV must contain columns named date and sales.")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
    frame = frame.dropna(subset=["date", "sales"])
    if frame.empty:
        raise ValueError("No valid date/sales rows were found.")
    if (frame["sales"] < 0).any():
        raise ValueError("Sales must be non-negative.")
    series_col = next((c for c in ("id", "item_id", "store_id") if c in frame.columns), None)
    if series_col is None:
        frame["series_id"] = "Uploaded series"
    else:
        frame["series_id"] = frame[series_col].astype(str)
    frame = frame.groupby(["series_id", "date"], as_index=False)["sales"].sum()
    frame = frame.sort_values(["series_id", "date"])
    if frame.duplicated(["series_id", "date"]).any():
        raise ValueError("Each series/date pair must be unique after aggregation.")
    return frame


@app.callback(
    Output("data-store", "data"),
    Output("series-select", "options"),
    Output("series-select", "value"),
    Output("upload-message", "children"),
    Output("data-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def load_upload(contents, filename):
    if not contents:
        return no_update, no_update, no_update, no_update, no_update
    try:
        frame = parse_upload(contents)
        payload = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
        series = sorted(frame["series_id"].unique().tolist())
        options = [{"label": value, "value": value} for value in series]
        msg = f"Loaded {len(frame):,} daily rows across {len(series):,} series from {filename or 'CSV'}."
        return payload, options, series[0], msg, f"{len(series):,} series loaded"
    except ValueError as exc:
        return None, [], None, f"Upload error: {exc}", "Invalid upload"


@app.callback(
    Output("forecast-chart", "figure"),
    Output("forecast-summary", "children"),
    Output("wape-value", "children"),
    Output("mae-value", "children"),
    Output("forecast-count", "children"),
    Output("metric-status", "children"),
    Output("forecast-store", "data"),
    Input("forecast-button", "n_clicks"),
    State("data-store", "data"),
    State("series-select", "value"),
    State("horizon", "value"),
    prevent_initial_call=True,
)
def make_forecast(n_clicks, payload, selected_series, horizon):
    if not payload or not selected_series:
        return empty_figure(), "Upload valid data and select a series first.", "—", "—", "0", "Awaiting data", None
    frame = pd.DataFrame(payload)
    frame["date"] = pd.to_datetime(frame["date"])
    history = frame.loc[frame["series_id"] == selected_series].sort_values("date").copy()
    history = history.drop_duplicates("date", keep="last")
    if len(history) < 14:
        return empty_figure(), "At least 14 daily observations are required for a weekly seasonal baseline.", "—", "—", "0", "Insufficient history", None
    if not history["date"].is_monotonic_increasing:
        history = history.sort_values("date")
    observed = history["sales"].astype(float).to_numpy()
    horizon = int(horizon or 28)
    last_date = history["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    last_week = observed[-7:]
    predictions = np.resize(last_week, horizon).astype(float)
    forecast = pd.DataFrame({"series_id": selected_series, "date": future_dates, "forecast": predictions})

    validation_size = min(horizon, max(7, len(observed) // 5))
    if len(observed) > validation_size + 7:
        train = observed[:-validation_size]
        actual = observed[-validation_size:]
        validation_pred = np.resize(train[-7:], validation_size)
        denom = float(np.abs(actual).sum())
        wape_value = float(np.abs(actual - validation_pred).sum() / denom) if denom > 0 else None
        mae_value = float(np.mean(np.abs(actual - validation_pred)))
        wape_text = f"{wape_value:.1%}" if wape_value is not None else "Undefined (zero actual demand)"
        mae_text = f"{mae_value:.2f} units"
        metric_status = "Validation evaluated"
    else:
        wape_text, mae_text, metric_status = "—", "—", "Insufficient validation history"

    fig = go.Figure()
    recent = history.tail(90)
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["sales"], name="Observed demand", mode="lines",
                             line={"width": 2}))
    fig.add_trace(go.Scatter(x=forecast["date"], y=forecast["forecast"], name="Seasonal-naive forecast",
                             mode="lines+markers", line={"dash": "dash", "width": 2}))
    fig.add_vline(x=last_date, line_dash="dot", line_color="#64748b")
    fig.update_layout(template="plotly_white", height=430, hovermode="x unified",
                      margin={"l": 25, "r": 20, "t": 35, "b": 25},
                      xaxis_title="Date", yaxis_title="Units sold", legend={"orientation": "h", "y": 1.08})
    summary = f"{selected_series}: {len(history):,} observations through {last_date.date()}; forecasting {horizon} days."
    return fig, summary, wape_text, mae_text, str(len(forecast)), metric_status, forecast.assign(date=forecast["date"].dt.strftime("%Y-%m-%d")).to_dict("records")


@app.callback(
    Output("download-forecast", "data"),
    Input("download-button", "n_clicks"),
    State("forecast-store", "data"),
    prevent_initial_call=True,
)
def download_forecast(n_clicks, rows):
    if not rows:
        return no_update
    return dcc.send_data_frame(pd.DataFrame(rows).to_csv, "demand_forecast.csv", index=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=False)
