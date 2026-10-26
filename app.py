"""Plotly Dash entry point for the demand and inventory platform.

The interface intentionally shows an honest empty state until evaluated forecasts
and replenishment outputs are generated; it never displays fabricated KPIs.
"""

import os
from pathlib import Path

import dash
from dash import html
import plotly.graph_objects as go
from dash import dcc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw" / "m5"
RESULTS_FILE = ROOT / "artifacts" / "forecast_results.csv"

app = dash.Dash(__name__, title="Retail Demand & Inventory Intelligence")
app.layout = html.Div(
    [
        html.Header(
            [
                html.P("RETAIL INTELLIGENCE / M5", className="eyebrow"),
                html.H1("Demand Forecasting & Inventory Optimization"),
                html.P(
                    "Forecast demand across stores, inspect uncertainty, and translate "
                    "predictions into replenishment decisions.",
                    className="subtitle",
                ),
            ],
            className="hero",
        ),
        html.Section(
            [
                html.Div(
                    [html.Span("DATA STATUS"), html.Strong("Available" if DATA_DIR.exists() else "Not configured")],
                    className="status-card",
                ),
                html.Div(
                    [html.Span("FORECAST OUTPUT"), html.Strong("Loaded" if RESULTS_FILE.exists() else "Not generated")],
                    className="status-card",
                ),
                html.Div(
                    [html.Span("MODEL METRICS"), html.Strong("Pending evaluation")],
                    className="status-card",
                ),
            ],
            className="status-grid",
        ),
        html.Main(
            [
                html.H2("Forecast workspace"),
                html.P(
                    "This workspace will display actuals, point forecasts, prediction intervals, "
                    "hierarchical roll-ups, and inventory recommendations once the evaluation "
                    "pipeline produces artifacts."
                ),
                dcc.Graph(
                    id="forecast-empty-state",
                    figure=go.Figure().update_layout(
                        title="No evaluated forecast loaded",
                        annotations=[
                            {
                                "text": "Run the forecasting pipeline to populate this chart.",
                                "xref": "paper",
                                "yref": "paper",
                                "x": 0.5,
                                "y": 0.5,
                                "showarrow": False,
                                "font": {"size": 16},
                            }
                        ],
                        xaxis={"visible": False},
                        yaxis={"visible": False},
                        template="plotly_white",
                        height=360,
                        margin={"l": 20, "r": 20, "t": 60, "b": 20},
                    ),
                    config={"displayModeBar": False, "responsive": True},
                ),
                html.P(
                    "Data and model outputs are not fabricated. The dashboard is a presentation "
                    "shell until reproducible forecasts and optimization results are available.",
                    className="note",
                ),
            ],
            className="panel",
        ),
        html.Footer("Built for reproducible retail forecasting · Project foundation v0.1.0"),
    ],
    className="page",
)

if __name__ == "__main__":
    # Render requires the web server to bind to all interfaces and its assigned port.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
        debug=False,
    )
