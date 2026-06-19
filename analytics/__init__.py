"""
analytics
=========

A small, well-documented library that walks through the standard stages of a
data analytics project. Every module receives and/or returns a pandas
``DataFrame`` so the stages can be chained together into a single pipeline.

The data analytics lifecycle implemented here (Gartner-style maturity model):

    1. Data Loading        -> ``dataloading``        : ingest raw (batch) data into a DataFrame.
    1b Streaming Ingestion -> ``datastreaming``       : consume Kafka/RabbitMQ/simulated streams.
    2. Data Cleansing      -> ``datacleansing``       : fix quality issues.
    3. Data Exploration    -> ``dataexploration``     : understand the data (EDA).
    4. Data Visualization  -> ``datavisualization``   : basic & advanced charts.
    5. Descriptive         -> ``descriptiveanalysis`` : "What happened?"
    6. Diagnostic          -> ``diagnosticanalysis``  : "Why did it happen?"
    7. Predictive          -> ``predictiveanalysis``  : "What will happen?"
    8. Prescriptive        -> ``prescriptiveanalysis``: "What should we do?"
    9. Machine Learning     -> ``machinelearning``     : reusable model pipeline.

Typical usage
-------------
>>> from analytics import dataloading, datacleansing, dataexploration
>>> df = dataloading.load_csv("data/sales.csv")
>>> df = datacleansing.standardize_column_names(df)
>>> report = dataexploration.summary_report(df)

The public API version. Bump on breaking changes.
"""

# Semantic version of the library. Kept here so callers can introspect it.
__version__ = "0.1.0"

# Re-export the stage modules so users can simply do ``from analytics import dataloading``.
from . import (  # noqa: F401  (imported for re-export convenience, not used here)
    dataloading,
    datastreaming,
    datacleansing,
    dataexploration,
    datavisualization,
    descriptiveanalysis,
    diagnosticanalysis,
    predictiveanalysis,
    prescriptiveanalysis,
    machinelearning,
)

# Explicit public surface. Controls what ``from analytics import *`` exposes.
__all__ = [
    "dataloading",
    "datastreaming",
    "datacleansing",
    "dataexploration",
    "datavisualization",
    "descriptiveanalysis",
    "diagnosticanalysis",
    "predictiveanalysis",
    "prescriptiveanalysis",
    "machinelearning",
    "__version__",
]
