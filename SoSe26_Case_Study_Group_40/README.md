# Vehicle Quality & Supply-Chain Dashboard

This project is a portable Dash application for the university case study **Data Analysis and Web App**. The final cleaned vehicle CSV is the **only** analytical source: every page, chart, map, table, and download is built from it and nothing else, as required by the brief.

## Folder structure

```text
SoSe26_Case_Study_Group_40/
├── app.py
├── requirements.txt
├── README.md
├── assets/
│   ├── logo.jpg
│   └── style.css
└── data/
    └── SoSe26_Case_Study_finalData_Group_40.csv
```

The supplied Quality Science logo is included as `assets/logo.jpg`. Dash loads `assets/style.css` automatically.

## Install and run

Python 3.10 or newer is recommended.

From a terminal opened in this folder:

```bash
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install the dependencies and start the app:

```bash
pip install -r requirements.txt
python app.py
```

The dashboard opens automatically in the default browser. If browser auto-open is
blocked by the operating system, manually open <http://127.0.0.1:8050>.

All paths in the code are relative to `app.py`, so the project folder can be copied to another computer without path changes.

> **Note on startup time:** the app loads roughly 3.2 million rows once at
> startup (about 30–60 s, ~1–2 GB RAM). This is expected — it is not a hang.

## Dataset setup

Place the final cleaned CSV at:

```text
data/SoSe26_Case_Study_finalData_Group_40.csv
```

This file is produced by the case-study notebook (Step 2). If the filename ever
changes, edit only `DATA_FILENAME` near the top of `app.py`.

Required columns:

- `ID_Fahrzeug`
- `Fahrzeug_Produktionsdatum`
- `Fehlerhaft_Own`
- `EarliestPartDatum`
- `LatestComponentDatum`
- `Fehlerhaft_FromComponents`
- `Fahrzeug_Typ`
- `LeadTime_Total_Days`
- `Stage1_PartsToLastComponent_Days`
- `Stage2_ComponentToVehicle_Days`
- `Fahrzeug_Fehlerhaft_Final`

The geographic map additionally uses the registration columns already merged into
the final dataset (`Zulassung_Ort`, `Zulassung_PLZ`, `Zulassung_lat`,
`Zulassung_lon`, `Zulassung_Datum`). These come from the KBA registration and
geodata tables and are part of the final dataset — no external file is read.

The CSV is loaded once at startup with compact numeric, categorical, date, and Arrow-backed string types. High-traffic chart aggregates are also created once at startup. The complete-data table uses server-side pagination and sends only 50 rows at a time to the browser. Box-plot statistics are calculated server-side and cached; raw row-level observations are not shipped to Plotly.

## Pages

### Geographic distribution

An interactive bubble map of vehicle registrations across Germany, built entirely
from the real coordinates in the final dataset. Two aggregation levels are
available and can be switched from the sidebar:

- **Municipalities** — from `Zulassung_Ort` (with `Zulassung_PLZ` shown for
  disambiguation)
- **Postal codes** — from `Zulassung_PLZ`

Bubble size represents the number of registrations; the hover shows the total and
the per-vehicle-type breakdown. Coordinates come from `Zulassung_lat` /
`Zulassung_lon`. No coordinates are invented and no federal-state layer derived
from any external lookup is used — only the final dataset.

The column detection also accepts the earlier planned aliases (`Gemeinde`,
`Breitengrad`, `Laengengrad`, `Zulassung`) so the map still activates if a column
was named differently upstream. The map activates automatically when latitude,
longitude, and at least one location column are present.

### Monthly distribution

Displays a stacked bar or stacked area chart by vehicle type. The page uses `Zulassung_Datum` (or the supported alias `Zulassung`) when available and falls back row by row to `Fahrzeug_Produktionsdatum` when necessary. Date and vehicle-type filters affect the chart together. When the selected range includes an incomplete first or last calendar month, that period is visibly labelled **Partial month** and calculated only from the selected days.

### Quality analysis

Shows total vehicles, overall final defect rate, in-house involvement, component involvement, defect-rate trends by vehicle type, and mutually exclusive defect-cause categories. A short note explains why the overall defect rate is high (the brief's definition cascades part- and component-level defects up to the vehicle). The highlighted finding explicitly compares supply-chain and in-house involvement. Vehicles affected by both sources are identified separately to prevent accidental double counting.

### Lead-time analysis

Includes side-by-side server-side box summaries of total lead time, median durations for both supply-chain stages, yearly median lead-time trends, and a median defective-versus-defect-free comparison. Vehicle-type and production-year filters apply to every view on the page.

### Full dataset

Shows every original column and every row from the final CSV through server-side pagination. Users can sort one column at a time and search by complete or partial vehicle ID. The full dataset is never transferred to the browser at once.

### Download

Streams either all final rows or one selected vehicle type as CSV in chunks. Internal helper columns used by the dashboard are excluded from the export.

## Submission checks

Before submitting:

1. Confirm every file listed in the folder structure is present.
2. Run `python app.py` from the submission folder.
3. Open every page and test all filters, table search/sorting/pagination, and download options.
4. Test both map aggregation levels (municipalities and postal codes).
5. Capture meaningful screenshots for the notebook's results section.
6. Copy the whole folder to a second location or computer and repeat the install/run test.

Do not add the original production-database tables to this folder. Only the final
processed CSV belongs in `data/`.
