"""Interactive Dash demo for retail demand forecasting.

Upload a small long-format CSV with columns date,sales and optionally id/item_id/store_id.
The app generates a weekly seasonal-naive forecast from the uploaded observed history.
It does not pretend the M5 dataset or trained models are bundled with the deployment.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

def empty_figure():
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white", height=390, margin={"l": 25, "r": 20, "t": 35, "b": 25},
        xaxis_title="Date", yaxis_title="Units sold",
        annotations=[{"text": "Upload demand history and generate a forecast", "xref": "paper",
                      "yref": "paper", "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 15}}],
    )
    return fig


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
                html.Button("Load M5 sample from repository", id="load-m5-button", n_clicks=0, className="secondary-button"),
                html.P("Loads the first 20 M5 item-store series when Git LFS data files are available.", className="muted"),
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
                        html.Div([html.Span("Validation RMSSE"), html.Strong("—", id="rmsse-value")], className="metric-card"),
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
        html.Section(
            [
                html.H2("3. Replenishment decision"),
                html.P(
                    "Optimize an order against the selected series' forecast using explicit holding and stockout costs. "
                    "This is a deterministic planning scenario, not a claim of realized savings."
                ),
                html.Div(
                    [
                        html.Div([html.Label("On-hand inventory (units)"), dcc.Input(id="on-hand-input", type="number", min=0, value=20, step=1)]),
                        html.Div([html.Label("Lead time (days)"), dcc.Input(id="lead-time-input", type="number", min=0, value=7, step=1)]),
                        html.Div([html.Label("Review period (days)"), dcc.Input(id="review-period-input", type="number", min=0, value=7, step=1)]),
                        html.Div([html.Label("Holding cost / unit"), dcc.Input(id="holding-cost-input", type="number", min=0, value=0.01, step=0.01)]),
                        html.Div([html.Label("Stockout cost / unit"), dcc.Input(id="stockout-cost-input", type="number", min=0, value=1.0, step=0.1)]),
                        html.Div([html.Label("Maximum order quantity"), dcc.Input(id="max-order-input", type="number", min=0, value=500, step=1)]),
                        html.Div([html.Label("Order multiple"), dcc.Input(id="order-multiple-input", type="number", min=1, value=1, step=1)]),
                    ],
                    className="inventory-controls",
                ),
                html.Button("Optimize replenishment", id="optimize-inventory-button", n_clicks=0, className="primary-button"),
                html.Div(id="inventory-summary", className="message"),
                html.Div(
                    [
                        html.Div([html.Span("Recommended order"), html.Strong("—", id="order-quantity")], className="metric-card"),
                        html.Div([html.Span("Projected ending stock"), html.Strong("—", id="ending-inventory")], className="metric-card"),
                        html.Div([html.Span("Projected shortage"), html.Strong("—", id="projected-shortage")], className="metric-card"),
                        html.Div([html.Span("Modeled cost"), html.Strong("—", id="modeled-cost")], className="metric-card"),
                    ],
                    className="metric-grid",
                ),
            ],
            className="panel",
        ),
        html.Footer("Reproducible retail forecasting · baseline forecast + constrained replenishment"),
        dcc.Store(id="data-store"),
        dcc.Store(id="forecast-store"),
    ],
)


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


def load_repository_m5_sample(series_limit: int = 20) -> pd.DataFrame:
    """Load a small real M5 sample from Git LFS-backed files without reading all rows."""
    from demand_inventory.data import validate_m5_files

    paths = validate_m5_files(ROOT)
    sales = pd.read_csv(paths["sales"], nrows=series_limit)
    id_columns = [column for column in ("id", "item_id", "store_id") if column in sales.columns]
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    if not id_columns or not day_columns:
        raise ValueError("The repository sales CSV does not have the expected M5 wide format.")
    long = sales.melt(id_vars=id_columns, value_vars=day_columns, var_name="d", value_name="sales")
    calendar = pd.read_csv(paths["calendar"], usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    long = long.merge(calendar, on="d", how="left", validate="many_to_one")
    long["sales"] = pd.to_numeric(long["sales"], errors="coerce")
    long = long.dropna(subset=["date", "sales"])
    long["series_id"] = long["id"].astype(str)
    return long.groupby(["series_id", "date"], as_index=False)["sales"].sum().sort_values(["series_id", "date"])


@app.callback(
    Output("data-store", "data"),
    Output("series-select", "options"),
    Output("series-select", "value"),
    Output("upload-message", "children"),
    Output("data-status", "children"),
    Input("upload-data", "contents"),
    Input("load-m5-button", "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def load_upload(contents, m5_clicks, filename):
    if ctx.triggered_id == "load-m5-button":
        if not m5_clicks:
            return no_update, no_update, no_update, no_update, no_update
    elif not contents:
        return no_update, no_update, no_update, no_update, no_update
    try:
        if ctx.triggered_id == "load-m5-button":
            frame = load_repository_m5_sample()
            source = "repository M5 sample"
        else:
            frame = parse_upload(contents)
            source = filename or "uploaded CSV"
        payload = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
        series = sorted(frame["series_id"].unique().tolist())
        options = [{"label": value, "value": value} for value in series]
        msg = f"Loaded {len(frame):,} daily rows across {len(series):,} series from {source}."
        return payload, options, series[0], msg, f"{len(series):,} series loaded"
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        return None, [], None, f"Data load error: {exc}", "Data unavailable"


@app.callback(
    Output("forecast-chart", "figure"),
    Output("forecast-summary", "children"),
    Output("wape-value", "children"),
    Output("rmsse-value", "children"),
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
        return empty_figure(), "Upload valid data and select a series first.", "—", "—", "—", "0", "Awaiting data", None
    frame = pd.DataFrame(payload)
    frame["date"] = pd.to_datetime(frame["date"])
    history = frame.loc[frame["series_id"] == selected_series].sort_values("date").copy()
    history = history.drop_duplicates("date", keep="last")
    if len(history) < 14:
        return empty_figure(), "At least 14 daily observations are required for a weekly seasonal baseline.", "—", "—", "—", "0", "Insufficient history", None
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
        try:
            from demand_inventory.metrics import rmsse

            rmsse_value = rmsse(actual, validation_pred, train, seasonal_period=7)
            rmsse_text = f"{rmsse_value:.3f}"
        except ValueError:
            rmsse_text = "Undefined"
        metric_status = "Validation evaluated"
    else:
        wape_text, rmsse_text, mae_text, metric_status = "—", "—", "—", "Insufficient validation history"

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
    return fig, summary, wape_text, rmsse_text, mae_text, str(len(forecast)), metric_status, forecast.assign(date=forecast["date"].dt.strftime("%Y-%m-%d")).to_dict("records")


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


@app.callback(
    Output("inventory-summary", "children"),
    Output("order-quantity", "children"),
    Output("ending-inventory", "children"),
    Output("projected-shortage", "children"),
    Output("modeled-cost", "children"),
    Input("optimize-inventory-button", "n_clicks"),
    State("forecast-store", "data"),
    State("on-hand-input", "value"),
    State("lead-time-input", "value"),
    State("review-period-input", "value"),
    State("holding-cost-input", "value"),
    State("stockout-cost-input", "value"),
    State("max-order-input", "value"),
    State("order-multiple-input", "value"),
    prevent_initial_call=True,
)
def optimize_inventory(n_clicks, forecast_rows, on_hand, lead_time, review_period, holding_cost, stockout_cost, max_order, order_multiple):
    if not forecast_rows:
        return "Generate a forecast before optimizing replenishment.", "—", "—", "—", "—"
    try:
        from demand_inventory.inventory import optimize_replenishment

        lead_time = int(lead_time or 0)
        review_period = int(review_period or 0)
        coverage = lead_time + review_period
        if coverage < 1:
            raise ValueError("Lead time plus review period must be at least one day.")
        forecast_frame = pd.DataFrame(forecast_rows).sort_values("date")
        if coverage > len(forecast_frame):
            raise ValueError(
                f"The selected forecast covers {len(forecast_frame)} days, but the planning window needs {coverage}. "
                "Increase the forecast horizon or reduce lead time/review period."
            )
        decision = optimize_replenishment(
            forecast_frame["forecast"].head(coverage).to_numpy(dtype=float),
            on_hand_inventory=float(on_hand or 0),
            lead_time_days=lead_time,
            review_period_days=review_period,
            holding_cost_per_unit=float(holding_cost or 0),
            stockout_cost_per_unit=float(stockout_cost or 0),
            max_order_quantity=int(max_order) if max_order is not None else None,
            order_multiple=int(order_multiple or 1),
        )
        summary = (
            f"Optimized for {decision.expected_demand:.1f} forecast units over {coverage} days. "
            "Costs are illustrative inputs and exclude purchase costs and supplier constraints."
        )
        return (
            summary,
            f"{decision.order_quantity:,} units",
            f"{decision.projected_ending_inventory:.1f} units",
            f"{decision.projected_shortage:.1f} units",
            f"{decision.modeled_total_cost:.2f}",
        )
    except (ValueError, ImportError, RuntimeError) as exc:
        return f"Optimization unavailable: {exc}", "—", "—", "—", "—"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=False)
