"""Interactive vehicle quality and supply-chain dashboard.

The application intentionally reads only the final, cleaned dataset.  All paths are
resolved relative to this file so that the submission folder is portable.
"""

from __future__ import annotations

import math
import os
import webbrowser
from collections.abc import Iterable
from functools import lru_cache
from threading import Timer
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from flask import Response, request, stream_with_context

# =============================================================================
# Imports and configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_BUILD = "7"

# Change only this value if the final CSV receives a different filename.
DATA_FILENAME = "SoSe26_Case_Study_finalData_Group_40.csv"
DATA_PATH = os.path.join(BASE_DIR, "data", DATA_FILENAME)
APP_HOST = "127.0.0.1"
APP_PORT = 8050
APP_URL = f"http://{APP_HOST}:{APP_PORT}"

REQUIRED_COLUMNS = {
    "ID_Fahrzeug",
    "Fahrzeug_Produktionsdatum",
    "Fehlerhaft_Own",
    "EarliestPartDatum",
    "LatestComponentDatum",
    "Fehlerhaft_FromComponents",
    "Fahrzeug_Typ",
    "LeadTime_Total_Days",
    "Stage1_PartsToLastComponent_Days",
    "Stage2_ComponentToVehicle_Days",
    "Fahrzeug_Fehlerhaft_Final",
}

GEO_COLUMN_ALIASES = {
    "registration_date": ("Zulassung_Datum", "Zulassung"),
    "municipality": ("Zulassung_Ort", "Gemeinde"),
    "postal_code": ("Zulassung_PLZ",),
    "latitude": ("Zulassung_lat", "Breitengrad"),
    "longitude": ("Zulassung_lon", "Laengengrad"),
}

GEO_LEVEL_LABELS = {
    "region": "Postal regions (coarse)",
    "municipality": "Municipalities",
    "postal_code": "Postal codes",
}

DATE_COLUMNS = {
    "Fahrzeug_Produktionsdatum",
    "EarliestPartDatum",
    "LatestComponentDatum",
    "Zulassung",
    "Zulassung_Datum",
}

TYPE_COLORS = {
    "OEM1_Typ11": "#1877C9",
    "OEM1_Typ12": "#55A7E8",
    "OEM2_Typ21": "#38B7A5",
    "OEM2_Typ22": "#7857C6",
}

# Map bubbles are coloured by how long the vehicles registered at a location
# took to build: cool for fast, warm for slow.
LEADTIME_COLORSCALE = [
    [0.0, "#7FC3EE"],
    [0.4, "#2B8FD8"],
    [0.7, "#F59E55"],
    [1.0, "#F07A6A"],
]

# The three defect measures plotted over time. "Total" follows the brief's
# cascading rule and is therefore near-saturated; showing it beside the two
# sources is what makes that visible instead of misleading.
SOURCE_COLORS = {
    "in_house": "#1877C9",
    "component": "#F59E55",
    "total": "#7857C6",
    "intact": "#38B7A5",
}

CAUSE_COLORS = {
    "In-house only": "#1877C9",
    "Component only": "#F59E55",
    "Both": "#7857C6",
    "None": "#A9CDE9",
}

CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

# Optional columns: present once the notebook has been re-run with the
# bottleneck features. The dashboard degrades gracefully when they are absent.
BOTTLENECK_COLUMNS = {
    "Bottleneck_Komponente_Typ",
    "Bottleneck_PartToComp_Days",
    "Bottleneck_Einzelteil_Typ",
    "Bottleneck_Tier2_Werk",
}

# Same cut-off the notebook uses for the extreme tail, so the dashboard and the
# report always quote the same percentages.
SLOW_THRESHOLD_DAYS = 100

# Components used in the OEM2 vehicle family, highlighted in the charts.
OEM2_COMPONENTS = {
    "K6", "K7", "K3AG2", "K3SG2", "K2LE2", "K2ST2", "K1BE2", "K1DI2",
}

COMPONENT_COLORS = {"oem2": "#F07A6A", "oem1": "#1877C9"}


# =============================================================================
# Data loading and lightweight feature engineering
# =============================================================================


def _string_dtype() -> str:
    """Use Arrow-backed strings when available to reduce memory consumption."""

    try:
        import pyarrow  # noqa: F401

        return "string[pyarrow]"
    except ImportError:
        return "string"


def detect_geo_schema(columns: Iterable[str]) -> dict[str, str | None]:
    """Resolve supported geo column aliases without requiring a manual flag."""

    available = set(columns)
    return {
        role: next(
            (candidate for candidate in candidates if candidate in available),
            None,
        )
        for role, candidates in GEO_COLUMN_ALIASES.items()
    }


def load_dataset(path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load the final CSV once with memory-conscious data types.

    The original column list is retained separately because internally derived
    columns must not appear in the complete-data table or CSV download.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Final dataset not found at {path}. Add the CSV to the data folder."
        )

    header = pd.read_csv(path, nrows=0)
    source_columns = header.columns.tolist()
    missing = sorted(REQUIRED_COLUMNS - set(source_columns))
    if missing:
        raise ValueError(
            "The final dataset is missing required columns: " + ", ".join(missing)
        )

    dtype_map = {
        "ID_Fahrzeug": _string_dtype(),
        "Fahrzeug_Typ": "category",
        "Fehlerhaft_Own": "Int8",
        "Fehlerhaft_FromComponents": "Int8",
        "Fahrzeug_Fehlerhaft_Final": "Int8",
        "LeadTime_Total_Days": "float32",
        "Stage1_PartsToLastComponent_Days": "float32",
        "Stage2_ComponentToVehicle_Days": "float32",
        "Gemeinde": "category",
        "Zulassung_Ort": "category",
        "Zulassung_PLZ": _string_dtype(),
        "Laengengrad": "float32",
        "Breitengrad": "float32",
        "Zulassung_lat": "float32",
        "Zulassung_lon": "float32",
        "Bottleneck_Komponente_Typ": "category",
        "Bottleneck_Einzelteil_Typ": "category",
        "Bottleneck_Tier2_Werk": "category",
        "Bottleneck_PartToComp_Days": "float32",
    }
    dtype_map = {
        column: dtype
        for column, dtype in dtype_map.items()
        if column in source_columns
    }
    parse_dates = [
        column for column in DATE_COLUMNS if column in source_columns
    ]

    data = pd.read_csv(
        path,
        dtype=dtype_map,
        parse_dates=parse_dates,
        memory_map=True,
    )

    # Coercion makes malformed date cells explicit as missing rather than
    # allowing a mixed-type column to break the date-based pages later.
    for column in parse_dates:
        data[column] = pd.to_datetime(data[column], errors="coerce")

    production_date = data["Fahrzeug_Produktionsdatum"]
    if production_date.notna().sum() == 0:
        raise ValueError(
            "Fahrzeug_Produktionsdatum contains no valid production dates."
        )
    if data["Fahrzeug_Typ"].notna().sum() == 0:
        raise ValueError("Fahrzeug_Typ contains no usable vehicle types.")
    geo_schema = detect_geo_schema(data.columns)
    registration_date_column = geo_schema["registration_date"]
    if registration_date_column:
        data["_Analysis_Date"] = data[registration_date_column].combine_first(
            production_date
        )
    else:
        data["_Analysis_Date"] = production_date

    data["_Production_Month"] = (
        production_date.dt.to_period("M").dt.to_timestamp()
    )
    data["_Analysis_Month"] = (
        data["_Analysis_Date"].dt.to_period("M").dt.to_timestamp()
    )
    data["_Analysis_Day"] = data["_Analysis_Date"].dt.normalize()
    data["_Production_Year"] = production_date.dt.year.astype("Int16")

    own = data["Fehlerhaft_Own"].fillna(0).eq(1)
    component = data["Fehlerhaft_FromComponents"].fillna(0).eq(1)
    cause = np.select(
        [own & ~component, ~own & component, own & component],
        ["In-house only", "Component only", "Both"],
        default="None",
    )
    data["_Defect_Cause"] = pd.Categorical(
        cause,
        categories=["In-house only", "Component only", "Both", "None"],
    )

    return data, source_columns


def prepare_aggregates(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create reusable server-side aggregates for high-traffic charts."""

    monthly_daily = (
        data.dropna(subset=["_Analysis_Day"])
        .groupby(
            ["_Analysis_Day", "Fahrzeug_Typ"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("vehicle_count")
        .reset_index()
    )

    quality_base = (
        data.dropna(subset=["_Production_Month"])
        .groupby(
            ["_Production_Month", "Fahrzeug_Typ"],
            observed=True,
            sort=True,
        )
        .agg(
            total_vehicles=("ID_Fahrzeug", "size"),
            final_defects=("Fahrzeug_Fehlerhaft_Final", "sum"),
            in_house_defects=("Fehlerhaft_Own", "sum"),
            component_defects=("Fehlerhaft_FromComponents", "sum"),
        )
        .reset_index()
    )

    cause_counts = (
        data.dropna(subset=["_Production_Month"])
        .groupby(
            ["_Production_Month", "Fahrzeug_Typ", "_Defect_Cause"],
            observed=False,
            sort=True,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for cause_name in CAUSE_COLORS:
        if cause_name not in cause_counts.columns:
            cause_counts[cause_name] = 0

    quality = quality_base.merge(
        cause_counts,
        on=["_Production_Month", "Fahrzeug_Typ"],
        how="left",
        validate="one_to_one",
    )

    return {
        "monthly_daily": monthly_daily,
        "quality": quality,
    }


def prepare_geo_aggregates(
    data: pd.DataFrame,
    geo_schema: dict[str, str | None],
) -> dict[str, pd.DataFrame]:
    """Aggregate detected real coordinates at every available location level."""

    latitude_column = geo_schema["latitude"]
    longitude_column = geo_schema["longitude"]
    if not latitude_column or not longitude_column:
        return {}

    # "region" is the first digit of the postal code (the German Leitzone), a
    # coarse grouping derived from the dataset's own PLZ column. It exists so
    # the map has a readable overview level - roughly ten bubbles instead of
    # several thousand.
    level_columns = {
        "region": geo_schema["postal_code"],
        "municipality": geo_schema["municipality"],
        "postal_code": geo_schema["postal_code"],
    }
    aggregates: dict[str, pd.DataFrame] = {}
    for level in ("region", "municipality", "postal_code"):
        location_column = level_columns[level]
        if not location_column:
            continue

        required = [
            "ID_Fahrzeug",
            "Fahrzeug_Typ",
            "LeadTime_Total_Days",
            location_column,
            latitude_column,
            longitude_column,
        ]
        postal_column = geo_schema["postal_code"]
        if level == "municipality" and postal_column:
            required.append(postal_column)
        geo_data = data[required].dropna(
            subset=[location_column, latitude_column, longitude_column]
        ).copy()
        geo_data = geo_data.loc[
            geo_data[latitude_column].between(-90, 90)
            & geo_data[longitude_column].between(-180, 180)
        ]
        if geo_data.empty:
            continue

        location = geo_data[location_column].astype("string").str.strip()
        if level == "municipality" and postal_column:
            postal = (
                geo_data[postal_column]
                .astype("string")
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
                .str.zfill(5)
            )
            has_postal = postal.notna() & postal.ne("")
            location = location.where(
                ~has_postal,
                location + " (" + postal + ")",
            )
        elif level == "postal_code":
            location = (
                "PLZ "
                + location.str.replace(r"\.0$", "", regex=True).str.zfill(5)
            )
        elif level == "region":
            digit = (
                location.str.replace(r"\.0$", "", regex=True)
                .str.zfill(5)
                .str[0]
            )
            location = "PLZ " + digit + "0000-" + digit + "9999"
        geo_data["_Geo_Location"] = location

        aggregate = (
            geo_data.groupby(
                ["_Geo_Location", "Fahrzeug_Typ"],
                observed=True,
                sort=True,
            )
            .agg(
                vehicle_count=("ID_Fahrzeug", "size"),
                # Sum and count instead of a median: unlike a median these
                # combine exactly when several vehicle types are selected, so
                # the mean stays correct under every filter.
                leadtime_sum=("LeadTime_Total_Days", "sum"),
                leadtime_count=("LeadTime_Total_Days", "count"),
                longitude=(longitude_column, "mean"),
                latitude=(latitude_column, "mean"),
            )
            .reset_index()
            .rename(columns={"_Geo_Location": "location"})
        )
        if not aggregate.empty:
            aggregates[level] = aggregate
    return aggregates


try:
    DATA_FRAME, SOURCE_COLUMNS = load_dataset(DATA_PATH)
    DATA_LOAD_ERROR: str | None = None
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    DATA_FRAME = pd.DataFrame()
    SOURCE_COLUMNS = []
    DATA_LOAD_ERROR = str(error)

if DATA_LOAD_ERROR is None:
    VEHICLE_TYPES = sorted(
        DATA_FRAME["Fahrzeug_Typ"].dropna().astype(str).unique().tolist()
    )
    YEAR_VALUES = sorted(
        DATA_FRAME["_Production_Year"].dropna().astype(int).unique().tolist()
    )
    AGGREGATES = prepare_aggregates(DATA_FRAME)
    # The map is built only from real geodata columns already present in the
    # final dataset (municipality, postal code, and coordinates). No external
    # file and no derived federal-state layer are used, so the app reads the
    # final dataset only, as required by the brief.
    GEO_SCHEMA = detect_geo_schema(DATA_FRAME.columns)
    GEO_AGGREGATES = prepare_geo_aggregates(DATA_FRAME, GEO_SCHEMA)
    GEO_AVAILABLE = bool(GEO_AGGREGATES)
    BOTTLENECK_AVAILABLE = BOTTLENECK_COLUMNS.issubset(DATA_FRAME.columns)
else:
    VEHICLE_TYPES = []
    YEAR_VALUES = []
    AGGREGATES = {}
    GEO_SCHEMA = detect_geo_schema([])
    GEO_AVAILABLE = False
    GEO_AGGREGATES = {}
    BOTTLENECK_AVAILABLE = False


# =============================================================================
# Figure and layout helpers
# =============================================================================


def color_for_type(vehicle_type: str, index: int = 0) -> str:
    """Return a stable colour for known and previously unseen vehicle types."""

    fallback = ["#1877C9", "#55A7E8", "#38B7A5", "#7857C6", "#F59E55"]
    return TYPE_COLORS.get(vehicle_type, fallback[index % len(fallback)])


def selected_types(values: Iterable[str] | None) -> list[str]:
    """Interpret an empty multi-select as all available vehicle types."""

    valid = [value for value in (values or []) if value in VEHICLE_TYPES]
    return valid or VEHICLE_TYPES


def apply_figure_style(
    figure: go.Figure,
    title: str,
    x_title: str | None = None,
    y_title: str | None = None,
    legend_title: str | None = None,
) -> go.Figure:
    """Apply the dashboard's common Plotly styling and accessible labels."""

    figure.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Source Sans Pro, Arial, sans-serif", "color": "#17324D"},
        margin={"l": 64, "r": 28, "t": 78, "b": 58},
        hoverlabel={"font": {"family": "Source Sans Pro, Arial, sans-serif"}},
        legend={
            "title": {"text": legend_title or ""},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 1,
        },
    )
    if x_title is not None:
        figure.update_xaxes(
            title=x_title,
            showgrid=False,
            linecolor="#C8DDEC",
        )
    if y_title is not None:
        figure.update_yaxes(
            title=y_title,
            gridcolor="#E7F1F8",
            zerolinecolor="#C8DDEC",
        )
    return figure


def empty_figure(title: str, message: str = "No data for this selection.") -> go.Figure:
    """Return a styled, informative empty state instead of a blank graph."""

    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#557086"},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return apply_figure_style(figure, title)


def graph_card(
    graph_id: str,
    subtitle: str,
    class_name: str = "chart-card",
) -> html.Div:
    """Wrap a responsive graph with a concise stakeholder-facing subtitle."""

    return html.Div(
        [
            html.P(subtitle, className="chart-subtitle"),
            dcc.Loading(
                dcc.Graph(
                    id=graph_id,
                    config=CHART_CONFIG,
                    className="dashboard-graph",
                ),
                type="circle",
                color="#2B8FD8",
            ),
        ],
        className=class_name,
    )


def page_intro(title: str, subtitle: str) -> html.Div:
    """Create a consistent page heading."""

    return html.Div(
        [html.H1(title), html.P(subtitle, className="page-subtitle")],
        className="page-intro",
    )


def vehicle_type_dropdown(component_id: str) -> dcc.Dropdown:
    """Create the standard multi-select vehicle filter."""

    return dcc.Dropdown(
        id=component_id,
        options=[{"label": item, "value": item} for item in VEHICLE_TYPES],
        value=VEHICLE_TYPES,
        multi=True,
        clearable=True,
        placeholder="All vehicle types",
    )


def year_slider(component_id: str) -> dcc.RangeSlider:
    """Create a compact year-range control from the available production years."""

    minimum = min(YEAR_VALUES)
    maximum = max(YEAR_VALUES)
    if len(YEAR_VALUES) <= 12:
        marked_years = YEAR_VALUES
    else:
        indices = np.linspace(0, len(YEAR_VALUES) - 1, 12, dtype=int)
        marked_years = [YEAR_VALUES[index] for index in sorted(set(indices))]
    marks = {year: str(year) for year in marked_years}
    return dcc.RangeSlider(
        id=component_id,
        min=minimum,
        max=maximum,
        step=1,
        value=[minimum, maximum],
        marks=marks,
        allowCross=False,
        tooltip={"placement": "bottom", "always_visible": False},
    )


def filter_card(children: list, title: str = "Filters") -> html.Aside:
    """Build the reusable left-hand filter panel."""

    return html.Aside(
        [html.H2(title, className="filter-title"), *children],
        className="filter-card",
    )


def field(label: str, component: html.Component, hint: str | None = None) -> html.Div:
    """Pair a form control with its label and optional explanatory hint."""

    children: list[html.Component] = [html.Label(label), component]
    if hint:
        children.append(html.P(hint, className="field-hint"))
    return html.Div(children, className="filter-field")


def two_column_page(sidebar: html.Aside, content: html.Component) -> html.Div:
    """Arrange filters and analytical content responsively."""

    return html.Div([sidebar, content], className="analysis-layout")


def metric_card(label: str, value_id: str, note: str) -> html.Div:
    """Create a quality KPI card whose value is supplied by a callback."""

    return html.Div(
        [
            html.P(label, className="metric-label"),
            html.P("—", id=value_id, className="metric-value"),
            html.P(note, className="metric-note"),
        ],
        className="metric-card",
    )


def geographic_page() -> html.Div:
    """Render either the live map controls or the automatic geo placeholder."""

    intro = page_intro(
        "Geographic distribution",
        "Explore where registered vehicles are concentrated across Germany.",
    )
    if not GEO_AVAILABLE:
        missing = []
        if not GEO_SCHEMA["latitude"]:
            missing.append("Latitude: Zulassung_lat or Breitengrad")
        if not GEO_SCHEMA["longitude"]:
            missing.append("Longitude: Zulassung_lon or Laengengrad")
        if not any(
            GEO_SCHEMA[level]
            for level in ("municipality", "postal_code")
        ):
            missing.append(
                "Location: Zulassung_Ort (or Gemeinde) or Zulassung_PLZ"
            )
        return html.Div(
            [
                intro,
                html.Div(
                    [
                        html.Div("Map data pending", className="status-pill"),
                        html.H2("The map will activate automatically"),
                        html.P(
                            "No coordinates are invented. Add the merged registration "
                            "and geodata columns to the final CSV, then "
                            "restart the app."
                        ),
                        html.H3("Columns still required"),
                        html.Ul([html.Li(column) for column in missing]),
                        html.P(
                            "When coordinates and at least one supported location "
                            "column are detected, this panel is replaced by an "
                            "interactive bubble map.",
                            className="placeholder-note",
                        ),
                    ],
                    className="not-available-panel",
                ),
            ]
        )

    level_options = [
        {"label": GEO_LEVEL_LABELS[level], "value": level}
        for level in GEO_AGGREGATES
    ]
    default_level = (
        "municipality"
        if "municipality" in GEO_AGGREGATES
        else next(iter(GEO_AGGREGATES))
    )
    filter_children = [
            field(
                "Geographic level",
                dcc.Dropdown(
                    id="geo-level",
                    options=level_options,
                    value=default_level,
                    clearable=False,
                ),
                hint=(
                    "Levels are built from the registration municipality and "
                    "postal code already contained in the final dataset."
                ),
            ),
            field("Vehicle type", vehicle_type_dropdown("geo-types")),
        ]
    sidebar = filter_card(filter_children)
    content = html.Section(
        [
            graph_card(
                "geo-map",
                "Bubble size represents registrations; hover for the vehicle-type mix.",
                class_name="chart-card map-card",
            )
        ],
        className="content-column",
    )
    return html.Div([intro, two_column_page(sidebar, content)])


def monthly_page() -> html.Div:
    """Create the monthly distribution page."""

    monthly_daily = AGGREGATES["monthly_daily"]
    minimum_timestamp = monthly_daily["_Analysis_Day"].min()
    maximum_timestamp = monthly_daily["_Analysis_Day"].max()
    default_start_timestamp = max(
        minimum_timestamp,
        (maximum_timestamp.to_period("M") - 23).to_timestamp(),
    )
    minimum = minimum_timestamp.date()
    maximum = maximum_timestamp.date()
    default_start = default_start_timestamp.date()
    filters = html.Section(
        [
            html.Div(
                [
                    html.H2("Filters"),
                    html.P(
                        "Choose a period, vehicle types, and chart view."
                    ),
                ],
                className="monthly-filter-heading",
            ),
            html.Div(
                [
                    field(
                        "Date range",
                        dcc.DatePickerRange(
                            id="monthly-date-range",
                            min_date_allowed=minimum,
                            max_date_allowed=maximum,
                            start_date=default_start,
                            end_date=maximum,
                            display_format="YYYY-MM-DD",
                            minimum_nights=0,
                            number_of_months_shown=1,
                            clearable=False,
                            updatemode="bothdates",
                        ),
                        "Registration date is used when available; otherwise "
                        "production date.",
                    ),
                    field(
                        "Vehicle type",
                        vehicle_type_dropdown("monthly-types"),
                    ),
                    field(
                        "Chart view",
                        dcc.RadioItems(
                            id="monthly-chart-view",
                            options=[
                                {"label": "Stacked bars", "value": "bar"},
                                {"label": "Stacked area", "value": "area"},
                            ],
                            value="bar",
                            className="segmented-control",
                            inputClassName="segmented-input",
                            labelClassName="segmented-label",
                        ),
                    ),
                ],
                className="monthly-filter-grid",
            ),
        ],
        className="monthly-filter-panel",
    )
    content = html.Section(
        [
            html.Div(
                id="monthly-summary",
                className="selection-summary monthly-selection-summary",
            ),
            graph_card(
                "monthly-chart",
                "Monthly bars remain available for detailed analysis; axis labels "
                "are spaced automatically for longer periods.",
                class_name="chart-card monthly-chart-card",
            ),
        ],
        className="content-column monthly-content",
    )
    return html.Div(
        [
            page_intro(
                "Monthly distribution",
                "Monitor vehicle volume and model mix over the selected period.",
            ),
            filters,
            content,
        ]
    )


def quality_page() -> html.Div:
    """Create the principal quality-analysis page."""

    sidebar = filter_card(
        [
            field("Vehicle type", vehicle_type_dropdown("quality-types")),
            field("Production year", year_slider("quality-years")),
            field(
                "Trend view",
                dcc.RadioItems(
                    id="quality-trend-view",
                    options=[
                        {"label": "By defect source", "value": "source"},
                        {"label": "By vehicle type", "value": "type"},
                    ],
                    value="source",
                    className="segmented-control",
                    inputClassName="segmented-input",
                    labelClassName="segmented-label",
                ),
                hint=(
                    "The total rate is near-saturated by design; splitting it "
                    "by source shows which part of it actually moves."
                ),
            ),
        ]
    )
    content = html.Section(
        [
            html.Div(
                [
                    metric_card(
                        "Total vehicles", "quality-total", "Selected population"
                    ),
                    metric_card(
                        "Vehicles with any flag",
                        "quality-rate",
                        "At least one of the ~19 parts, components, or the "
                        "vehicle record itself is flagged (case-study brief).",
                    ),
                    metric_card(
                        "Completely unflagged",
                        "quality-clean",
                        "Nothing flagged anywhere in the vehicle",
                    ),
                    metric_card(
                        "In-house involvement",
                        "quality-own-share",
                        "Share of flagged vehicles",
                    ),
                ],
                className="metric-grid",
            ),
            html.Div(
                [
                    html.Strong("Read this figure as a cascade, not a verdict"),
                    html.Span(
                        "A vehicle carries roughly 19 items that can each be "
                        "flagged: about 14 single parts, its 4 components, and "
                        "the vehicle record itself. Each one is individually "
                        "around 90% clean - but requiring all 19 to be clean at "
                        "once is what leaves only a small share unflagged."
                    ),
                    html.Small(
                        "So the headline figure mostly reflects how many parts a "
                        "vehicle contains, not how well it was assembled. The "
                        "in-house rate below is the one the plant controls."
                    ),
                ],
                className="finding-callout",
            ),
            html.Div(id="quality-callout", className="finding-callout"),
            graph_card(
                "quality-trend",
                "In-house assembly is the rate the plant actually controls. The "
                "total sits far above it and tracks the component line almost "
                "exactly - the supply chain, not final assembly, sets it. The "
                "defect-free line is the share leaving the plant with nothing "
                "flagged anywhere, i.e. the complement of the total.",
            ),
            graph_card(
                "quality-causes",
                "Exclusive cause categories prevent double counting vehicles "
                "affected by both sources.",
            ),
        ],
        className="content-column",
    )
    return html.Div(
        [
            page_intro(
                "Quality analysis",
                "Separate in-house assembly problems from component-related "
                "supply-chain defects.",
            ),
            two_column_page(sidebar, content),
        ]
    )


def lead_time_page() -> html.Div:
    """Create the lead-time analysis page."""

    sidebar = filter_card(
        [
            field("Vehicle type", vehicle_type_dropdown("lead-types")),
            field("Production year", year_slider("lead-years")),
        ]
    )
    content = html.Section(
        [
            graph_card(
                "lead-box",
                "Boxes are calculated server-side; only distribution summaries are "
                "sent to the browser.",
            ),
            html.Div(
                [
                    graph_card(
                        "lead-stages",
                        "Compare time spent before and after the last component is "
                        "available.",
                        class_name="chart-card half-card",
                    ),
                    graph_card(
                        "lead-median",
                        "The headline number per model, read straight off the "
                        "bar - the boxplot beside it shows the spread behind "
                        "each median.",
                        class_name="chart-card half-card",
                    ),
                ],
                className="chart-grid",
            ),
            graph_card(
                "lead-trend",
                "Yearly medians reveal sustained improvements or bottlenecks by "
                "vehicle type.",
            ),
        ],
        className="content-column",
    )
    return html.Div(
        [
            page_intro(
                "Lead-time analysis",
                "Locate supply-chain bottlenecks and compare throughput with quality "
                "outcomes.",
            ),
            two_column_page(sidebar, content),
        ]
    )


def component_color(component_type: str) -> str:
    """Colour OEM2-family components distinctly from OEM1-family ones."""

    family = "oem2" if component_type in OEM2_COMPONENTS else "oem1"
    return COMPONENT_COLORS[family]


def bottleneck_page() -> html.Div:
    """Create the component/part bottleneck page behind the headline finding."""

    intro = page_intro(
        "Component bottleneck",
        "Trace long lead times down to the component, part, and supplier plant "
        "responsible.",
    )
    if not BOTTLENECK_AVAILABLE:
        missing = sorted(BOTTLENECK_COLUMNS - set(DATA_FRAME.columns))
        return html.Div(
            [
                intro,
                html.Div(
                    [
                        html.Div("Bottleneck data pending", className="status-pill"),
                        html.H2("This page activates automatically"),
                        html.P(
                            "Re-run the case-study notebook to add the bottleneck "
                            "features to the final dataset, then restart the app. "
                            "Nothing else needs to change."
                        ),
                        html.H3("Columns still required"),
                        html.Ul([html.Li(column) for column in missing]),
                    ],
                    className="not-available-panel",
                ),
            ]
        )

    sidebar = filter_card(
        [
            field("Vehicle type", vehicle_type_dropdown("bottleneck-types")),
            field("Production year", year_slider("bottleneck-years")),
        ]
    )
    content = html.Section(
        [
            html.Div(
                [
                    metric_card(
                        "Vehicles in selection",
                        "bottleneck-total",
                        "Each vehicle counted once",
                    ),
                    metric_card(
                        "Held up over 100 days",
                        "bottleneck-slow",
                        f"Bottleneck component took more than "
                        f"{SLOW_THRESHOLD_DAYS} days",
                    ),
                    metric_card(
                        "Most frequent bottleneck",
                        "bottleneck-worst",
                        "Component causing the longest delays",
                    ),
                ],
                className="metric-grid",
            ),
            html.Div(id="bottleneck-callout", className="finding-callout"),
            graph_card(
                "bottleneck-components",
                "Each vehicle is attributed to whichever of its four installed "
                "components took longest. Red marks the OEM2 component family.",
            ),
            html.Div(
                [
                    graph_card(
                        "bottleneck-parts",
                        "Among vehicles delayed beyond 100 days: which individual "
                        "part was the oldest one waiting in the component.",
                        class_name="chart-card half-card",
                    ),
                    graph_card(
                        "bottleneck-plants",
                        "Among those same vehicles: which Tier-2 supplier plants "
                        "the delaying parts came from.",
                        class_name="chart-card half-card",
                    ),
                ],
                className="chart-grid",
            ),
        ],
        className="content-column",
    )
    return html.Div([intro, two_column_page(sidebar, content)])


def full_dataset_page() -> html.Div:
    """Create the server-side paginated complete-data table."""

    columns = [{"name": column, "id": column} for column in SOURCE_COLUMNS]
    return html.Div(
        [
            page_intro(
                "Full dataset",
                "Review every row of the final cleaned dataset without loading "
                "millions of records into the browser.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Search by vehicle ID"),
                            dcc.Input(
                                id="table-search",
                                type="search",
                                debounce=True,
                                placeholder="Enter an exact or partial ID",
                                className="search-input",
                            ),
                        ],
                        className="table-search-block",
                    ),
                    html.Div(id="table-summary", className="table-summary"),
                ],
                className="table-toolbar",
            ),
            dcc.Loading(
                dash_table.DataTable(
                    id="full-data-table",
                    columns=columns,
                    data=[],
                    page_action="custom",
                    page_current=0,
                    page_size=50,
                    page_count=1,
                    sort_action="custom",
                    sort_mode="single",
                    sort_by=[],
                    fixed_rows={"headers": True},
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "66vh",
                        "overflowY": "auto",
                    },
                    style_cell={
                        "fontFamily": "Source Sans Pro, Arial, sans-serif",
                        "fontSize": 14,
                        "padding": "10px 12px",
                        "textAlign": "left",
                        "minWidth": "145px",
                        "maxWidth": "260px",
                        "whiteSpace": "normal",
                    },
                    style_header={
                        "backgroundColor": "#DDF0FC",
                        "color": "#17324D",
                        "fontWeight": 700,
                        "border": "1px solid #B8D7EB",
                    },
                    style_data={
                        "backgroundColor": "#FFFFFF",
                        "border": "1px solid #E1EDF5",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#F7FBFE",
                        }
                    ],
                ),
                type="circle",
                color="#2B8FD8",
            ),
        ]
    )


def download_page() -> html.Div:
    """Create the streamed CSV download page."""

    return html.Div(
        [
            page_intro(
                "Download",
                "Export the complete cleaned dataset or only the vehicle type needed "
                "for further analysis.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            field(
                                "Vehicle type",
                                dcc.Dropdown(
                                    id="download-type",
                                    options=[
                                        {"label": "All vehicle types", "value": "all"},
                                        *[
                                            {"label": item, "value": item}
                                            for item in VEHICLE_TYPES
                                        ],
                                    ],
                                    value="all",
                                    clearable=False,
                                ),
                            ),
                            html.A(
                                "Download CSV",
                                id="download-link",
                                href="/download/final-dataset.csv?vehicle_type=all",
                                className="primary-button",
                            ),
                        ],
                        className="download-card",
                    ),
                    html.Div(
                        [
                            html.H2("Memory-efficient export"),
                            html.P(
                                "The server streams the chosen rows in chunks. The "
                                "complete multi-million-row dataset is never sent to "
                                "the browser as a table object."
                            ),
                            html.P(
                                "Only columns from the final CSV are included; "
                                "internal dashboard helper fields are excluded.",
                                className="download-note",
                            ),
                        ],
                        className="download-explainer",
                    ),
                ],
                className="download-layout",
            ),
        ]
    )


NAV_ITEMS = [
    ("Geographic", "/geographic"),
    ("Monthly", "/monthly"),
    ("Quality", "/quality"),
    ("Lead time", "/lead-time"),
    ("Bottleneck", "/bottleneck"),
    ("Full dataset", "/dataset"),
    ("Download", "/download"),
]


def navigation() -> html.Header:
    """Create the branded top navigation bar."""

    links = [
        dcc.Link(label, href=path, id=f"nav-{index}", className="nav-link")
        for index, (label, path) in enumerate(NAV_ITEMS)
    ]
    return html.Header(
        html.Div(
            [
                dcc.Link(
                    html.Img(
                        src=app.get_asset_url("logo.jpg"),
                        alt="Quality Science logo",
                    ),
                    href="/geographic",
                    className="brand-link",
                ),
                html.Nav(
                    links,
                    className="top-nav",
                    **{"aria-label": "Main navigation"},
                ),
            ],
            className="nav-inner",
        ),
        className="site-header",
    )


def data_error_panel() -> html.Div:
    """Explain a missing or invalid final CSV without crashing the web server."""

    return html.Main(
        [
            page_intro(
                "Dataset setup required",
                "The application is installed correctly, but the final CSV could "
                "not be loaded.",
            ),
            html.Div(
                [
                    html.H2("Add the final dataset"),
                    html.P(DATA_LOAD_ERROR),
                    html.Code(os.path.join("data", DATA_FILENAME)),
                    html.P(
                        "After adding or renaming the file, restart the application. "
                        "The dashboard and optional map will configure themselves "
                        "from the detected columns."
                    ),
                ],
                className="not-available-panel error-panel",
            ),
        ],
        className="page-shell",
    )


# =============================================================================
# Dash layout
# =============================================================================


app = Dash(
    __name__,
    assets_folder=os.path.join(BASE_DIR, "assets"),
    suppress_callback_exceptions=True,
    title=f"Vehicle Quality Dashboard · UI build {UI_BUILD}",
    update_title="Updating…",
)
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navigation(),
        html.Div(id="page-content", className="page-shell"),
        html.Footer(
            [
                html.Span(
                    "Vehicle Quality & Supply-Chain Dashboard · "
                    "Final cleaned dataset only"
                ),
                html.Span(
                    f"UI build {UI_BUILD}",
                    className="build-pill",
                ),
            ],
            className="site-footer",
        ),
    ]
)


# =============================================================================
# Page routing and callbacks
# =============================================================================


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname: str) -> html.Component:
    """Route URL paths to page layouts."""

    if DATA_LOAD_ERROR is not None:
        return data_error_panel()

    routes = {
        "/": geographic_page,
        "/geographic": geographic_page,
        "/monthly": monthly_page,
        "/quality": quality_page,
        "/lead-time": lead_time_page,
        "/bottleneck": bottleneck_page,
        "/dataset": full_dataset_page,
        "/download": download_page,
    }
    page_factory = routes.get(pathname, geographic_page)
    return page_factory()


@app.callback(
    [Output(f"nav-{index}", "className") for index in range(len(NAV_ITEMS))],
    Input("url", "pathname"),
)
def highlight_navigation(pathname: str) -> list[str]:
    """Mark the current page in the top navigation."""

    current = "/geographic" if pathname == "/" else pathname
    return [
        "nav-link active" if path == current else "nav-link"
        for _, path in NAV_ITEMS
    ]


@app.callback(
    Output("monthly-chart", "figure"),
    Output("monthly-summary", "children"),
    Input("monthly-date-range", "start_date"),
    Input("monthly-date-range", "end_date"),
    Input("monthly-types", "value"),
    Input("monthly-chart-view", "value"),
)
def update_monthly_chart(
    start_date: str,
    end_date: str,
    vehicle_types: list[str] | None,
    chart_view: str,
) -> tuple[go.Figure, str]:
    """Filter and render the monthly volume view from the pre-aggregation."""

    data = AGGREGATES["monthly_daily"]
    types = selected_types(vehicle_types)
    start_day = pd.Timestamp(start_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    filtered_daily = data.loc[
        data["Fahrzeug_Typ"].astype(str).isin(types)
        & data["_Analysis_Day"].between(start_day, end_day)
    ]
    if filtered_daily.empty:
        return empty_figure("Monthly vehicle distribution"), "0 vehicles selected"

    filtered = filtered_daily.assign(
        _Analysis_Month=filtered_daily["_Analysis_Day"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    filtered = (
        filtered.groupby(
            ["_Analysis_Month", "Fahrzeug_Typ"],
            observed=True,
            sort=True,
        )["vehicle_count"]
        .sum()
        .reset_index()
    )

    start_month = start_day.to_period("M").to_timestamp()
    end_month = end_day.to_period("M").to_timestamp()
    end_month_last_day = end_day.to_period("M").end_time.normalize()
    partial_months: set[pd.Timestamp] = set()
    if start_day > start_month:
        partial_months.add(start_month)
    if end_day < end_month_last_day:
        partial_months.add(end_month)

    figure = go.Figure()
    for index, vehicle_type in enumerate(types):
        group = filtered.loc[
            filtered["Fahrzeug_Typ"].astype(str).eq(vehicle_type)
        ]
        if group.empty:
            continue
        month_status = [
            "Partial month" if month in partial_months else "Complete month"
            for month in group["_Analysis_Month"]
        ]
        if chart_view == "area":
            figure.add_trace(
                go.Scatter(
                    x=group["_Analysis_Month"],
                    y=group["vehicle_count"],
                    name=vehicle_type,
                    mode="lines",
                    stackgroup="one",
                    line={"color": color_for_type(vehicle_type, index), "width": 2},
                    customdata=month_status,
                    hovertemplate=(
                        "%{x|%b %Y}<br>Vehicles: %{y:,.0f}<br>%{customdata}"
                        "<extra>"
                        + vehicle_type
                        + "</extra>"
                    ),
                )
            )
        else:
            figure.add_trace(
                go.Bar(
                    x=group["_Analysis_Month"],
                    y=group["vehicle_count"],
                    name=vehicle_type,
                    marker_color=color_for_type(vehicle_type, index),
                    marker_pattern_shape=[
                        "/" if status == "Partial month" else ""
                        for status in month_status
                    ],
                    marker_pattern_solidity=0.25,
                    customdata=month_status,
                    hovertemplate=(
                        "%{x|%b %Y}<br>Vehicles: %{y:,.0f}<br>%{customdata}"
                        "<extra>"
                        + vehicle_type
                        + "</extra>"
                    ),
                )
            )

    if chart_view == "bar":
        figure.update_layout(barmode="stack", bargap=0.2)
    for partial_month in sorted(partial_months):
        figure.add_vrect(
            x0=partial_month,
            x1=partial_month + pd.offsets.MonthBegin(1),
            fillcolor="#F59E55",
            opacity=0.08,
            line_width=0,
            annotation_text="Partial month",
            annotation_position="top left",
            annotation_font={"color": "#A45A1D", "size": 12},
        )
    apply_figure_style(
        figure,
        "Monthly vehicle distribution by type",
        "Month",
        "Vehicles (count)",
        "Vehicle type",
    )
    month_count = (
        (end_month.year - start_month.year) * 12
        + end_month.month
        - start_month.month
        + 1
    )
    if month_count <= 18:
        date_tick = "M1"
        date_format = "%b\n%Y"
    elif month_count <= 48:
        date_tick = "M3"
        date_format = "%b\n%Y"
    else:
        date_tick = "M12"
        date_format = "%Y"
    figure.update_xaxes(
        dtick=date_tick,
        tickformat=date_format,
        tickangle=0,
        ticklabelmode="period",
        automargin=True,
    )
    figure.update_layout(
        hovermode="x unified",
        height=540,
    )
    summary = (
        f"{int(filtered['vehicle_count'].sum()):,} vehicles · "
        f"{start_day:%d %b %Y} to {end_day:%d %b %Y} · "
        f"{len(types)} type(s)"
    )
    if partial_months:
        summary += " · partial month highlighted"
    return figure, summary


@app.callback(
    Output("quality-total", "children"),
    Output("quality-rate", "children"),
    Output("quality-clean", "children"),
    Output("quality-own-share", "children"),
    Output("quality-callout", "children"),
    Output("quality-trend", "figure"),
    Output("quality-causes", "figure"),
    Input("quality-types", "value"),
    Input("quality-years", "value"),
    Input("quality-trend-view", "value"),
)
def update_quality(
    vehicle_types: list[str] | None,
    years: list[int],
    trend_view: str,
) -> tuple[str, str, str, str, html.Component, go.Figure, go.Figure]:
    """Update quality KPIs, comparison callout, trend, and cause breakdown."""

    types = selected_types(vehicle_types)
    start_year, end_year = years
    data = AGGREGATES["quality"]
    production_year = data["_Production_Month"].dt.year
    filtered = data.loc[
        data["Fahrzeug_Typ"].astype(str).isin(types)
        & production_year.between(start_year, end_year)
    ]
    if filtered.empty:
        empty_trend = empty_figure("Defect rate over time")
        empty_causes = empty_figure("Defect cause breakdown")
        return (
            "0",
            "0.0%",
            "0.0%",
            "0.0%",
            "No data selected.",
            empty_trend,
            empty_causes,
        )

    total = int(filtered["total_vehicles"].sum())
    final_defects = int(filtered["final_defects"].sum())
    own_defects = int(filtered["in_house_defects"].sum())
    component_defects = int(filtered["component_defects"].sum())
    defect_rate = final_defects / total if total else 0
    own_share = own_defects / final_defects if final_defects else 0
    component_share = component_defects / final_defects if final_defects else 0

    difference = component_share - own_share
    if final_defects == 0:
        finding = "No defective vehicles occur in the selected population."
    elif difference > 0:
        finding = (
            f"Supply-chain involvement is {abs(difference) * 100:.1f} percentage "
            "points higher "
            "than in-house involvement among defective vehicles."
        )
    elif difference < 0:
        finding = (
            f"In-house involvement is {abs(difference) * 100:.1f} percentage "
            "points higher "
            "than component involvement among defective vehicles."
        )
    else:
        finding = (
            "Component and in-house defects have equal involvement among "
            "defective vehicles."
        )
    callout = html.Div(
        [
            html.Strong("Main finding · supply chain vs in-house"),
            html.Span(finding),
            html.Small(
                "A vehicle can involve both sources, so the two involvement shares "
                "need not sum to 100%."
            ),
        ]
    )

    trend_figure = go.Figure()
    if trend_view == "source":
        # One row per month across the whole selection, so the three measures
        # share a denominator and can be read against each other directly.
        monthly = (
            filtered.groupby("_Production_Month", observed=True, sort=True)[
                [
                    "total_vehicles",
                    "final_defects",
                    "in_house_defects",
                    "component_defects",
                ]
            ]
            .sum()
            .reset_index()
        )
        denominator = monthly["total_vehicles"].replace(0, np.nan)
        source_series = [
            (
                "In-house assembly",
                monthly["in_house_defects"].div(denominator),
                SOURCE_COLORS["in_house"],
                "solid",
            ),
            (
                "Via installed components",
                monthly["component_defects"].div(denominator),
                SOURCE_COLORS["component"],
                "solid",
            ),
            (
                "Total (brief's definition)",
                monthly["final_defects"].div(denominator),
                SOURCE_COLORS["total"],
                "dot",
            ),
            (
                "Defect-free (intact)",
                monthly["total_vehicles"]
                .sub(monthly["final_defects"])
                .div(denominator),
                SOURCE_COLORS["intact"],
                "dash",
            ),
        ]
        for label, values, colour, dash in source_series:
            trend_figure.add_trace(
                go.Scatter(
                    x=monthly["_Production_Month"],
                    y=values,
                    name=label,
                    mode="lines",
                    line={"color": colour, "width": 2.5, "dash": dash},
                    hovertemplate=(
                        "%{x|%b %Y}<br>" + label + ": %{y:.2%}<extra></extra>"
                    ),
                )
            )
        apply_figure_style(
            trend_figure,
            "Defect rate over time by source",
            "Production month",
            "Share of vehicles (%)",
            "Defect source",
        )
        trend_figure.update_layout(hovermode="x unified")
    else:
        trend = (
            filtered.groupby(
                ["_Production_Month", "Fahrzeug_Typ"],
                observed=True,
                sort=True,
            )[["total_vehicles", "final_defects"]]
            .sum()
            .reset_index()
        )
        trend["defect_rate"] = trend["final_defects"].div(
            trend["total_vehicles"].replace(0, np.nan)
        )
        for index, vehicle_type in enumerate(types):
            group = trend.loc[trend["Fahrzeug_Typ"].astype(str).eq(vehicle_type)]
            if group.empty:
                continue
            trend_figure.add_trace(
                go.Scatter(
                    x=group["_Production_Month"],
                    y=group["defect_rate"],
                    name=vehicle_type,
                    mode="lines+markers",
                    line={
                        "color": color_for_type(vehicle_type, index),
                        "width": 2.5,
                    },
                    marker={"size": 6},
                    hovertemplate=(
                        "%{x|%b %Y}<br>Defect rate: %{y:.2%}<extra>"
                        + vehicle_type
                        + "</extra>"
                    ),
                )
            )
        apply_figure_style(
            trend_figure,
            "Total defect rate over time by vehicle type",
            "Production month",
            "Defect rate (%)",
            "Vehicle type",
        )
    trend_figure.update_yaxes(tickformat=".1%", rangemode="tozero")
    trend_figure.update_xaxes(tickformat="%b\n%Y")

    causes = list(CAUSE_COLORS)
    counts = [int(filtered[cause].sum()) for cause in causes]
    shares = [count / total if total else 0 for count in counts]
    cause_figure = go.Figure(
        go.Bar(
            x=causes,
            y=shares,
            customdata=np.array(counts)[:, None],
            marker_color=[CAUSE_COLORS[cause] for cause in causes],
            text=[f"{share:.1%}" for share in shares],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>Share: %{y:.2%}<br>Vehicles: "
                "%{customdata[0]:,.0f}<extra></extra>"
            ),
        )
    )
    apply_figure_style(
        cause_figure,
        "Defect cause breakdown",
        "Exclusive defect category",
        "Share of vehicles (%)",
    )
    cause_figure.update_yaxes(tickformat=".1%", rangemode="tozero")
    cause_figure.update_layout(showlegend=False)

    return (
        f"{total:,}",
        f"{defect_rate:.2%}",
        f"{1 - defect_rate:.2%}",
        f"{own_share:.1%}",
        callout,
        trend_figure,
        cause_figure,
    )


@lru_cache(maxsize=12)
def lead_time_statistics(
    vehicle_types: tuple[str, ...], start_year: int, end_year: int
) -> tuple[tuple[tuple, ...], ...]:
    """Cache exact box and median summaries for lead-time filter selections."""

    columns = [
        "_Production_Year",
        "Fahrzeug_Typ",
        "Fahrzeug_Fehlerhaft_Final",
        "LeadTime_Total_Days",
        "Stage1_PartsToLastComponent_Days",
        "Stage2_ComponentToVehicle_Days",
    ]
    filtered = DATA_FRAME.loc[
        DATA_FRAME["Fahrzeug_Typ"].astype(str).isin(vehicle_types)
        & DATA_FRAME["_Production_Year"].between(start_year, end_year),
        columns,
    ]

    box_results = []
    for vehicle_type in vehicle_types:
        values = filtered.loc[
            filtered["Fahrzeug_Typ"].astype(str).eq(vehicle_type),
            "LeadTime_Total_Days",
        ].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        lower_candidates = values[values >= q1 - 1.5 * iqr]
        upper_candidates = values[values <= q3 + 1.5 * iqr]
        lower = lower_candidates.min() if lower_candidates.size else values.min()
        upper = upper_candidates.max() if upper_candidates.size else values.max()
        box_results.append(
            (
                vehicle_type,
                float(q1),
                float(median),
                float(q3),
                float(lower),
                float(upper),
                int(values.size),
            )
        )

    stage_columns = [
        "Stage1_PartsToLastComponent_Days",
        "Stage2_ComponentToVehicle_Days",
    ]
    stage_frame = (
        filtered.groupby("Fahrzeug_Typ", observed=True)[stage_columns]
        .median()
        .reset_index()
    )
    stage_results = tuple(
        (str(vehicle_type), float(stage_1), float(stage_2))
        for vehicle_type, stage_1, stage_2 in stage_frame.itertuples(
            index=False, name=None
        )
    )

    trend_frame = (
        filtered.groupby(
            ["_Production_Year", "Fahrzeug_Typ"],
            observed=True,
            sort=True,
        )["LeadTime_Total_Days"]
        .median()
        .rename("median_lead_time")
        .reset_index()
    )
    trend_results = tuple(
        (int(year), str(vehicle_type), float(median))
        for year, vehicle_type, median in trend_frame.itertuples(
            index=False, name=None
        )
    )

    return (
        tuple(box_results),
        stage_results,
        trend_results,
    )


@app.callback(
    Output("lead-box", "figure"),
    Output("lead-stages", "figure"),
    Output("lead-trend", "figure"),
    Output("lead-median", "figure"),
    Input("lead-types", "value"),
    Input("lead-years", "value"),
)
def update_lead_time(
    vehicle_types: list[str] | None,
    years: list[int],
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Update all lead-time views with the same vehicle and year filters."""

    types = selected_types(vehicle_types)
    start_year, end_year = years
    (
        box_statistics,
        stage_statistics,
        trend_statistics,
    ) = lead_time_statistics(tuple(types), start_year, end_year)

    if box_statistics:
        box_figure = go.Figure()
        for index, stats in enumerate(box_statistics):
            vehicle_type, q1, median, q3, lower, upper, count = stats
            box_figure.add_trace(
                go.Box(
                    name=vehicle_type,
                    x=[vehicle_type],
                    q1=[q1],
                    median=[median],
                    q3=[q3],
                    lowerfence=[lower],
                    upperfence=[upper],
                    customdata=[[count]],
                    marker_color=color_for_type(vehicle_type, index),
                    boxpoints=False,
                    hovertemplate=(
                        "Median: %{median:.1f} days<br>"
                        "Q1: %{q1:.1f} days<br>Q3: %{q3:.1f} days<br>"
                        "Vehicles: %{customdata[0]:,.0f}<extra>"
                        + vehicle_type
                        + "</extra>"
                    ),
                )
            )
        apply_figure_style(
            box_figure,
            "Total lead-time distribution by vehicle type",
            "Vehicle type",
            "Total lead time (days)",
            "Vehicle type",
        )
        box_figure.update_layout(
            showlegend=False,
            boxmode="group",
            boxgap=0.35,
            boxgroupgap=0.15,
        )
    else:
        box_figure = empty_figure("Total lead-time distribution by vehicle type")

    if not stage_statistics:
        return (
            box_figure,
            empty_figure("Median duration per supply-chain stage"),
            empty_figure("Median total lead time over the years"),
            empty_figure("Median total lead time by vehicle type"),
        )

    stage_specs = [
        ("Parts → last component", "Stage1_PartsToLastComponent_Days"),
        ("Last component → vehicle", "Stage2_ComponentToVehicle_Days"),
    ]
    stage_lookup = {
        vehicle_type: (stage_1, stage_2)
        for vehicle_type, stage_1, stage_2 in stage_statistics
    }
    stage_figure = go.Figure()
    for index, vehicle_type in enumerate(types):
        if vehicle_type not in stage_lookup:
            continue
        medians = stage_lookup[vehicle_type]
        stage_figure.add_trace(
            go.Bar(
                x=[label for label, _ in stage_specs],
                y=medians,
                name=vehicle_type,
                marker_color=color_for_type(vehicle_type, index),
                text=[f"{value:.1f} d" for value in medians],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>Median: %{y:.2f} days<extra>"
                + vehicle_type
                + "</extra>",
            )
        )
    apply_figure_style(
        stage_figure,
        "Median duration per supply-chain stage",
        "Supply-chain stage",
        "Median duration (days)",
        "Vehicle type",
    )
    stage_figure.update_layout(barmode="group")

    trend_figure = go.Figure()
    for index, vehicle_type in enumerate(types):
        group = [
            (year, median)
            for year, result_type, median in trend_statistics
            if result_type == vehicle_type
        ]
        if not group:
            continue
        years_for_type, medians = zip(*group, strict=True)
        trend_figure.add_trace(
            go.Scatter(
                x=years_for_type,
                y=medians,
                name=vehicle_type,
                mode="lines+markers",
                line={"color": color_for_type(vehicle_type, index), "width": 2.5},
                hovertemplate=(
                    "Year: %{x}<br>Median lead time: %{y:.2f} days<extra>"
                    + vehicle_type
                    + "</extra>"
                ),
            )
        )
    apply_figure_style(
        trend_figure,
        "Median total lead time over the years",
        "Production year",
        "Median total lead time (days)",
        "Vehicle type",
    )
    trend_figure.update_xaxes(dtick=1)

    # Medians already come out of the cached box statistics, so the headline
    # figure per model costs no extra pass over the data.
    median_labels = [stats[0] for stats in box_statistics]
    median_values = [stats[2] for stats in box_statistics]
    median_counts = [stats[6] for stats in box_statistics]
    median_figure = go.Figure(
        go.Bar(
            x=median_labels,
            y=median_values,
            customdata=np.array(median_counts)[:, None],
            marker_color=[
                color_for_type(label, index)
                for index, label in enumerate(median_labels)
            ],
            text=[f"{value:.0f} d" for value in median_values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>Median: %{y:.1f} days<br>Vehicles: "
                "%{customdata[0]:,.0f}<extra></extra>"
            ),
        )
    )
    apply_figure_style(
        median_figure,
        "Median total lead time by vehicle type",
        "Vehicle type",
        "Median total lead time (days)",
    )
    median_figure.update_layout(showlegend=False)

    return box_figure, stage_figure, trend_figure, median_figure


@lru_cache(maxsize=12)
def bottleneck_statistics(
    vehicle_types: tuple[str, ...], start_year: int, end_year: int
) -> tuple:
    """Cache component, part, and plant summaries for a filter selection."""

    filtered = DATA_FRAME.loc[
        DATA_FRAME["Fahrzeug_Typ"].astype(str).isin(vehicle_types)
        & DATA_FRAME["_Production_Year"].between(start_year, end_year),
        [
            "Bottleneck_Komponente_Typ",
            "Bottleneck_PartToComp_Days",
            "Bottleneck_Einzelteil_Typ",
            "Bottleneck_Tier2_Werk",
        ],
    ]
    if filtered.empty:
        return (), (), (), (0, 0)

    is_slow = filtered["Bottleneck_PartToComp_Days"] > SLOW_THRESHOLD_DAYS
    summary = (
        filtered.assign(_slow=is_slow)
        .groupby("Bottleneck_Komponente_Typ", observed=True)
        .agg(
            median_days=("Bottleneck_PartToComp_Days", "median"),
            vehicles=("Bottleneck_PartToComp_Days", "size"),
            slow_share=("_slow", "mean"),
        )
        .reset_index()
    )
    component_results = tuple(
        (str(name), float(median), int(count), float(share) * 100)
        for name, median, count, share in summary.itertuples(
            index=False, name=None
        )
    )

    slow_rows = filtered.loc[is_slow]
    part_results = tuple(
        (str(name), int(count))
        for name, count in slow_rows["Bottleneck_Einzelteil_Typ"]
        .value_counts()
        .head(8)
        .items()
    )
    plant_results = tuple(
        (str(name), int(count))
        for name, count in slow_rows["Bottleneck_Tier2_Werk"]
        .value_counts()
        .head(8)
        .items()
    )
    return (
        component_results,
        part_results,
        plant_results,
        (int(len(filtered)), int(is_slow.sum())),
    )


def ranked_bar(
    rows: tuple[tuple[str, int], ...],
    title: str,
    x_title: str,
    colour: str,
    total_slow: int,
) -> go.Figure:
    """Render a ranked count bar chart shared by the part and plant views."""

    if not rows:
        return empty_figure(title, "No vehicles beyond the delay threshold.")
    labels = [name for name, _ in rows]
    counts = [count for _, count in rows]
    shares = [count / total_slow if total_slow else 0 for count in counts]
    figure = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=colour,
            customdata=np.array(shares)[:, None],
            text=[f"{share:.1%}" for share in shares],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Vehicles: %{x:,.0f}<br>"
                "Share of delayed: %{customdata[0]:.1%}<extra></extra>"
            ),
        )
    )
    apply_figure_style(figure, title, x_title, None)
    figure.update_layout(showlegend=False)
    figure.update_yaxes(autorange="reversed")
    return figure


@app.callback(
    Output("bottleneck-total", "children"),
    Output("bottleneck-slow", "children"),
    Output("bottleneck-worst", "children"),
    Output("bottleneck-callout", "children"),
    Output("bottleneck-components", "figure"),
    Output("bottleneck-parts", "figure"),
    Output("bottleneck-plants", "figure"),
    Input("bottleneck-types", "value"),
    Input("bottleneck-years", "value"),
)
def update_bottleneck(
    vehicle_types: list[str] | None,
    years: list[int],
) -> tuple:
    """Update the bottleneck KPIs, finding, and the three diagnostic charts."""

    types = selected_types(vehicle_types)
    start_year, end_year = years
    components, parts, plants, totals = bottleneck_statistics(
        tuple(types), start_year, end_year
    )
    total_vehicles, slow_vehicles = totals

    if not components:
        blank = empty_figure("Bottleneck component by vehicle")
        return (
            "0",
            "0",
            "—",
            "No data for this selection.",
            blank,
            empty_figure("Delaying part"),
            empty_figure("Tier-2 supplier plant"),
        )

    # The component with the largest share of >100-day vehicles is the one
    # driving the extreme tail; ties fall back to the higher median.
    worst = max(components, key=lambda row: (row[3], row[1]))
    worst_name, worst_median, _worst_count, worst_share = worst

    by_share = sorted(components, key=lambda row: row[3], reverse=True)
    share_figure = go.Figure(
        go.Bar(
            x=[row[0] for row in by_share],
            y=[row[3] for row in by_share],
            marker_color=[component_color(row[0]) for row in by_share],
            customdata=np.array(
                [[row[1], row[2]] for row in by_share], dtype=float
            ),
            text=[f"{row[3]:.1f}%" for row in by_share],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{x}<br>Over " + str(SLOW_THRESHOLD_DAYS) + " days: %{y:.2f}%"
                "<br>Median: %{customdata[0]:.1f} days"
                "<br>Vehicles: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    apply_figure_style(
        share_figure,
        f"Share of vehicles held up more than {SLOW_THRESHOLD_DAYS} days, "
        "by bottleneck component",
        "Bottleneck component",
        "Share of vehicles (%)",
    )
    share_figure.update_layout(showlegend=False)

    parts_figure = ranked_bar(
        parts,
        "Delaying part (vehicles beyond the threshold)",
        "Vehicles",
        "#7857C6",
        slow_vehicles,
    )
    plants_figure = ranked_bar(
        plants,
        "Tier-2 supplier plant (vehicles beyond the threshold)",
        "Vehicles",
        "#38B7A5",
        slow_vehicles,
    )

    top_part = parts[0][0] if parts else "n/a"
    top_plant = plants[0][0] if plants else "n/a"
    callout = html.Div(
        [
            html.Strong("Main finding · where the delay comes from"),
            html.Span(
                f"{worst_name} is the bottleneck component with the largest "
                f"extreme tail: {worst_share:.1f}% of the vehicles it holds up "
                f"wait longer than {SLOW_THRESHOLD_DAYS} days "
                f"(median {worst_median:.0f} days). Among those delayed "
                f"vehicles the oldest waiting part is most often {top_part}, "
                f"supplied from {top_plant}."
            ),
            html.Small(
                "Recommended focus: OEM2_Typ21 — it shares the highest median "
                "lead time with Typ22 but affects far more vehicles, and both "
                "root causes apply to it."
            ),
        ]
    )

    slow_pct = slow_vehicles / total_vehicles if total_vehicles else 0
    return (
        f"{total_vehicles:,}",
        f"{slow_vehicles:,} ({slow_pct:.1%})",
        worst_name,
        callout,
        share_figure,
        parts_figure,
        plants_figure,
    )


def geo_hover_text(
    location: str,
    pivot: pd.DataFrame,
    types: list[str],
    note: str | None = None,
) -> tuple[str, int]:
    """Build one bubble's tooltip and its total, derived from the type lines.

    The total is summed from the very lines shown underneath it, so the two can
    never disagree. `note` carries the mean lead time, so the value the bubble's
    colour encodes is also readable as a number.
    """

    lines = []
    total = 0
    for vehicle_type in types:
        count = (
            int(pivot.loc[location, vehicle_type])
            if location in pivot.index and vehicle_type in pivot.columns
            else 0
        )
        total += count
        lines.append(f"{vehicle_type}: {count:,}")
    header = [f"<b>{location}</b>", f"Total vehicles: {total:,}"]
    if note:
        header.append(note)
    text = "<br>".join([*header, *lines])
    return text, total


@app.callback(
    Output("geo-map", "figure"),
    Input("geo-level", "value"),
    Input("geo-types", "value"),
)
def update_geo_map(
    level: str,
    vehicle_types: list[str] | None,
) -> go.Figure:
    """Build the map only from detected, real geodata columns."""

    types = selected_types(vehicle_types)
    if level not in GEO_AGGREGATES:
        level = next(iter(GEO_AGGREGATES))
    data = GEO_AGGREGATES[level]
    filtered = data.loc[data["Fahrzeug_Typ"].astype(str).isin(types)]
    if filtered.empty:
        return empty_figure("Vehicle registrations across Germany")

    grouped = (
        filtered.groupby("location", observed=True, sort=True)
        .agg(
            vehicle_count=("vehicle_count", "sum"),
            leadtime_sum=("leadtime_sum", "sum"),
            leadtime_count=("leadtime_count", "sum"),
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
        )
        .reset_index()
    )
    grouped["mean_leadtime"] = grouped["leadtime_sum"].div(
        grouped["leadtime_count"].replace(0, np.nan)
    )

    # Colour range clipped to the 5th–95th percentile so the bulk of the
    # locations spread across the full scale instead of collapsing into one
    # shade; a couple of extreme sites no longer flatten everyone else.
    # Falls back to real min/max when there aren't enough distinct values.
    leadtimes = grouped["mean_leadtime"].dropna()
    if len(leadtimes) >= 2:
        cmin = float(leadtimes.quantile(0.01))
        cmax = float(leadtimes.quantile(0.99))
        if cmin == cmax:  # all values (near) identical → let Plotly auto-range
            cmin = cmax = None
    else:
        cmin = cmax = None

    pivot = filtered.pivot_table(
        index="location",
        columns="Fahrzeug_Typ",
        values="vehicle_count",
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )

    # Every location is drawn - nothing is hidden behind a cutoff. Dense areas
    # are resolved by zooming in, which is what the map is for.
    grouped = grouped.reset_index(drop=True)
    total_locations = len(grouped)

    # Bubbles are scaled by area against the largest one on screen; the smaller
    # ceiling and the outline keep neighbouring towns distinguishable.
    max_count = max(float(grouped["vehicle_count"].max()), 1.0)
    # Map markers cannot carry an outline, so separation comes from the smaller
    # size ceiling plus partial transparency; allowoverlap keeps every bubble
    # drawn instead of silently dropping the ones underneath.
    marker_base = {
        "sizemode": "area",
        "sizeref": 2.0 * max_count / (34.0**2),
        "sizemin": 4,
        "opacity": 0.72,
        "allowoverlap": True,
    }

    hover = [
        geo_hover_text(
            str(name),
            pivot,
            types,
            note=(
                f"Mean lead time: {mean_days:,.1f} days"
                if mean_days == mean_days
                else "Mean lead time: n/a"
            ),
        )[0]
        for name, mean_days in zip(
            grouped["location"], grouped["mean_leadtime"], strict=True
        )
    ]
    figure = go.Figure(
        go.Scattermap(
            lat=grouped["latitude"],
            lon=grouped["longitude"],
            mode="markers",
            marker={
                **marker_base,
                "size": grouped["vehicle_count"],
                "color": grouped["mean_leadtime"],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": LEADTIME_COLORSCALE,
                "colorbar": {
                    "title": {"text": "Mean lead<br>time (days)"},
                    "thickness": 14,
                    "len": 0.7,
                    "outlinewidth": 0,
                },
                "showscale": True,
            },
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name="Registrations",
        )
    )

    level_label = {
        "region": "postal region",
        "municipality": "municipality",
        "postal_code": "postal code",
    }.get(level, "location")
    subtitle = f"all {total_locations:,}"
    figure.update_layout(
        title={
            "text": (
                f"Vehicle registrations by {level_label} "
                f"<span style='font-size:13px;color:#557086'>"
                f"({subtitle})</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        map={
            "style": "open-street-map",
            "center": {"lat": 51.15, "lon": 10.45},
            "zoom": 4.9 if level == "region" else 5.4,
        },
        margin={"l": 0, "r": 0, "t": 64, "b": 0},
        font={"family": "Source Sans Pro, Arial, sans-serif", "color": "#17324D"},
        paper_bgcolor="rgba(0,0,0,0)",
        autosize=True,
        showlegend=False,
    )
    return figure

@lru_cache(maxsize=4)
def table_positions(
    search_value: str, sort_column: str, sort_direction: str
) -> np.ndarray | None:
    """Cache row positions for expensive table search/sort combinations."""

    if search_value:
        mask = DATA_FRAME["ID_Fahrzeug"].str.contains(
            search_value, case=False, regex=False, na=False
        )
        positions = np.flatnonzero(mask.to_numpy())
    else:
        positions = None

    if not sort_column:
        return positions

    if sort_column not in SOURCE_COLUMNS:
        return positions
    ascending = sort_direction != "desc"
    if positions is None:
        return DATA_FRAME[sort_column].sort_values(
            ascending=ascending, na_position="last", kind="mergesort"
        ).index.to_numpy()

    subset = DATA_FRAME.iloc[positions]
    return subset[sort_column].sort_values(
        ascending=ascending, na_position="last", kind="mergesort"
    ).index.to_numpy()


def serializable_records(frame: pd.DataFrame) -> list[dict]:
    """Convert a small table page to JSON-safe display values."""

    display = frame.copy()
    for column in DATE_COLUMNS.intersection(display.columns):
        display[column] = display[column].dt.strftime("%Y-%m-%d")
    display = display.astype(object).where(pd.notna(display), None)
    return display.to_dict("records")


@app.callback(
    Output("full-data-table", "data"),
    Output("full-data-table", "page_count"),
    Output("table-summary", "children"),
    Input("full-data-table", "page_current"),
    Input("full-data-table", "page_size"),
    Input("full-data-table", "sort_by"),
    Input("table-search", "value"),
)
def update_table(
    page_current: int,
    page_size: int,
    sort_by: list[dict],
    search_value: str | None,
) -> tuple[list[dict], int, str]:
    """Send only the requested table page to the browser."""

    search = (search_value or "").strip()
    sort_column = sort_by[0]["column_id"] if sort_by else ""
    sort_direction = sort_by[0]["direction"] if sort_by else ""
    positions = table_positions(search, sort_column, sort_direction)
    total_rows = len(DATA_FRAME) if positions is None else len(positions)
    page_count = max(1, math.ceil(total_rows / page_size))
    safe_page = min(page_current, page_count - 1)
    start = safe_page * page_size
    end = min(start + page_size, total_rows)
    if positions is None:
        page = DATA_FRAME.iloc[start:end][SOURCE_COLUMNS]
    else:
        page = DATA_FRAME.iloc[positions[start:end]][SOURCE_COLUMNS]
    summary = (
        f"Showing {start + 1 if total_rows else 0:,}–{end:,} of "
        f"{total_rows:,} rows"
    )
    return serializable_records(page), page_count, summary


@app.callback(Output("download-link", "href"), Input("download-type", "value"))
def update_download_link(vehicle_type: str) -> str:
    """Encode the chosen type into the streamed-download URL."""

    value = vehicle_type if vehicle_type in VEHICLE_TYPES else "all"
    return f"/download/final-dataset.csv?vehicle_type={quote(value)}"


@server.route("/download/final-dataset.csv")
def stream_dataset_download() -> Response:
    """Stream the selected final-data rows as CSV in bounded-memory chunks."""

    if DATA_LOAD_ERROR is not None:
        return Response(DATA_LOAD_ERROR, status=503, mimetype="text/plain")

    vehicle_type = request.args.get("vehicle_type", "all")
    if vehicle_type != "all" and vehicle_type not in VEHICLE_TYPES:
        return Response("Unknown vehicle type.", status=400, mimetype="text/plain")

    if vehicle_type == "all":
        row_positions = None
        row_count = len(DATA_FRAME)
        suffix = "all"
    else:
        mask = DATA_FRAME["Fahrzeug_Typ"].astype(str).eq(vehicle_type)
        row_positions = np.flatnonzero(mask.to_numpy())
        row_count = len(row_positions)
        suffix = vehicle_type

    def generate_csv():
        chunk_size = 50_000
        for start in range(0, row_count, chunk_size):
            if row_positions is None:
                chunk = DATA_FRAME.iloc[start : start + chunk_size][
                    SOURCE_COLUMNS
                ]
            else:
                positions = row_positions[start : start + chunk_size]
                chunk = DATA_FRAME.iloc[positions][SOURCE_COLUMNS]
            yield chunk.to_csv(
                index=False,
                header=start == 0,
                date_format="%Y-%m-%d",
            )
        if row_count == 0:
            yield DATA_FRAME.iloc[0:0][SOURCE_COLUMNS].to_csv(index=False)

    dataset_stem = os.path.splitext(DATA_FILENAME)[0]
    filename = f"{dataset_stem}_{suffix}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(
        stream_with_context(generate_csv()),
        mimetype="text/csv",
        headers=headers,
    )


# =============================================================================
# Run
# =============================================================================


def open_dashboard_in_browser() -> None:
    """Open the local dashboard after the development server starts."""

    webbrowser.open_new_tab(APP_URL)


if __name__ == "__main__":
    browser_timer = Timer(1.0, open_dashboard_in_browser)
    browser_timer.daemon = True
    browser_timer.start()
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=False,
        use_reloader=False,
    )
