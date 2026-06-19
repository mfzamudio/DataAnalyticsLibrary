# DataAnalyticsLibrary

A small, heavily-documented Python library that walks through the standard
stages of a data-analytics project. Every stage receives and/or returns a
pandas `DataFrame`, so the modules chain cleanly into a single pipeline.

The code is written for learning and as a practical starting point: each
module, function, and non-obvious line is commented in English.

---

## The analytics lifecycle implemented here

| # | Stage              | Module                      | Question it answers          |
|---|--------------------|-----------------------------|------------------------------|
| 1 | Data Loading       | `analytics.dataloading`     | Get raw (batch) data into a DataFrame |
| 1b| Streaming Ingestion| `analytics.datastreaming`   | Consume Kafka/RabbitMQ/simulated streams |
| 2 | Data Cleansing     | `analytics.datacleansing`   | Fix data-quality issues       |
| 3 | Data Exploration   | `analytics.dataexploration` | What does the data look like? |
| 4 | Data Visualization | `analytics.datavisualization` | Show it (basic + advanced)  |
| 5 | Descriptive        | `analytics.descriptiveanalysis` | What happened?            |
| 6 | Diagnostic         | `analytics.diagnosticanalysis`  | Why did it happen?        |
| 7 | Predictive         | `analytics.predictiveanalysis`  | What will happen?         |
| 8 | Prescriptive       | `analytics.prescriptiveanalysis`| What should we do?        |
| 9 | Machine Learning   | `analytics.machinelearning` | Reusable model pipeline (used by stage 7) |

---

## Project structure

```
DataAnalyticsLibrary/
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── pyproject.toml                # pytest configuration
├── .gitignore
├── analytics/                    # the library package
│   ├── __init__.py
│   ├── dataloading.py            # CSV, text, Excel, JSON, Parquet, SQL -> DataFrame
│   ├── datastreaming.py          # Kafka/RabbitMQ/simulated streams -> micro-batch DataFrames
│   ├── datacleansing.py          # names, whitespace, missing, dupes, types, outliers
│   ├── dataexploration.py        # numeric/categorical summaries, correlations (EDA)
│   ├── datavisualization.py      # basic & advanced charts (matplotlib/seaborn)
│   ├── descriptiveanalysis.py    # aggregation, KPIs, trends
│   ├── diagnosticanalysis.py     # correlation-to-target, t-test, chi-square, drill-down
│   ├── predictiveanalysis.py     # regression, classification, clustering
│   ├── prescriptiveanalysis.py   # scenarios, rules, budget optimization
│   └── machinelearning.py        # split / preprocess / pipeline / cross-val / persist
├── data/
│   └── generate_sample_data.py   # creates a synthetic sample_sales.csv
├── notebooks/
│   ├── README.md                 # how to run the demo
│   └── pipeline_demo.ipynb       # end-to-end pipeline example
└── tests/                        # pytest unit tests (one file per module)
    ├── conftest.py               # shared fixtures + sys.path setup
    ├── test_dataloading.py
    ├── test_datacleansing.py
    ├── test_dataexploration.py
    ├── test_datavisualization.py
    ├── test_descriptiveanalysis.py
    ├── test_diagnosticanalysis.py
    ├── test_predictiveanalysis.py
    ├── test_prescriptiveanalysis.py
    ├── test_machinelearning.py
    └── test_datastreaming.py
```

> CI lives in `.github/workflows/ci.yml` (runs the test suite on Python 3.10–3.12
> and executes the demo notebook).

---

## Prerequisites

- **Python 3.10+** (developed and tested on 3.12).
- `pip` and the ability to create a virtual environment.

---

## Setup (venv + requirements.txt)

From the project root:

```bash
# 1. Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Generate the synthetic sample dataset (once)
python data/generate_sample_data.py
```

> **Database driver note:** `load_database` uses SQLAlchemy, which is
> installed. SQLite works out of the box; for PostgreSQL/MySQL install the
> matching driver (`pip install psycopg2-binary` or `pip install pymysql`).

---

## Quick start

```python
from analytics import dataloading, datacleansing, dataexploration

# 1. Load any supported source into a DataFrame
df = dataloading.load_csv("data/sample_sales.csv", parse_dates=["Order Date"])

# 2. Clean it (each step returns a new DataFrame; nothing is mutated in place)
df = datacleansing.standardize_column_names(df)
df = datacleansing.handle_missing(df, strategy="median")

# 3. Explore it
print(dataexploration.overview(df))
print(dataexploration.numeric_summary(df))
```

### A streaming example (no broker required)

```python
from analytics import datastreaming as ds, datacleansing

# A simulated, infinite event stream — swap for KafkaStreamSource /
# RabbitMQStreamSource in production (same interface).
source = ds.SimulatedStreamSource(n_messages=250)

# Consume the stream as DataFrame micro-batches and clean each one.
for batch in ds.micro_batch(source, batch_size=100):
    clean = datacleansing.standardize_column_names(batch)
    print("batch rows:", len(clean))
```

> Real brokers: `KafkaStreamSource` (needs `kafka-python` + a Kafka cluster)
> and `RabbitMQStreamSource` (needs `pika` + a RabbitMQ broker). These expose
> the exact same `StreamSource` interface as the simulator, so your pipeline
> code does not change. See `analytics/datastreaming.py` for a PySpark
> Structured Streaming template too.

### A predictive example

```python
from analytics import predictiveanalysis

result = predictiveanalysis.train_regressor(df, target="profit", model="random_forest")
print(result["metrics"])                       # {'r2': ..., 'mae': ..., 'rmse': ...}

from analytics import machinelearning
machinelearning.save_model(result["pipeline"], "profit_model.joblib")
```

---

## Running the demo notebook

See [`notebooks/README.md`](notebooks/README.md) for full instructions. Short
version:

```bash
source .venv/bin/activate
python -m ipykernel install --user --name dal --display-name "Python 3 (DataAnalyticsLibrary)"
jupyter lab notebooks/pipeline_demo.ipynb
```

---

## Running the tests

The library has a `pytest` suite (71 tests) covering every module. Plotting
tests run headless via the `Agg` backend.

```bash
source .venv/bin/activate
pytest                 # run everything
pytest -v              # verbose, one line per test
pytest tests/test_datacleansing.py   # a single module
```

---

## Design principles

- **One standard currency:** every loader returns a pandas `DataFrame`; every
  transformer takes and returns one.
- **Non-mutating transformers:** cleansing functions `copy()` their input, so
  pipelines are reproducible and side-effect free.
- **Fail loudly:** invalid input raises a clear exception instead of returning
  `None`.
- **Optional dependencies stay optional:** plotting, SQL, and stats imports are
  lazy and raise a helpful message if a package is missing.
- **Separation of concerns:** numbers (`dataexploration`) and charts
  (`datavisualization`) live in different modules.

---

## Honest scope / caveats

- The **synthetic dataset is fabricated** (NumPy random); it is for
  demonstration only and carries no real-world meaning.
- `prescriptiveanalysis` provides solid, documented *starting points*
  (what-if, rules, linear programming), not a turnkey decision engine — real
  prescriptive systems are domain-specific.
- Diagnostic tests report **associations, not causation**.
- Models in `predictiveanalysis` use sensible defaults and are **not
  hyperparameter-tuned**; tune them for production use.
