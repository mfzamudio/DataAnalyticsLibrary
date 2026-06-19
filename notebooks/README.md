# Demo Notebook — `pipeline_demo.ipynb`

An end-to-end walkthrough of the `analytics` library: loading &rarr; cleansing
&rarr; exploration &rarr; visualization &rarr; descriptive &rarr; diagnostic
&rarr; predictive &rarr; prescriptive analytics.

The notebook uses two data sources:
- A **synthetic sales CSV** (`../data/sample_sales.csv`) for the
  loading/cleansing/EDA/descriptive/diagnostic stages and a regression model.
- The **built-in scikit-learn Wine dataset** for a clean classification example.

---

## Prerequisites

1. You have completed the setup in the [top-level README](../README.md):
   created the `.venv`, installed `requirements.txt`, and run the data
   generator.

   ```bash
   # from the project root
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python data/generate_sample_data.py
   ```

2. The virtual environment is **active** in the shell you launch Jupyter from.

---

## Register the kernel (one time)

So the notebook runs against the project's virtual environment:

```bash
source .venv/bin/activate
python -m ipykernel install --user \
    --name dal \
    --display-name "Python 3 (DataAnalyticsLibrary)"
```

---

## Run it

### Option A — JupyterLab (interactive)

```bash
source .venv/bin/activate
jupyter lab notebooks/pipeline_demo.ipynb
```

Then select the kernel **"Python 3 (DataAnalyticsLibrary)"** (top-right in
JupyterLab) and run all cells.

### Option B — execute headless (CI / quick check)

Runs the whole notebook and writes an executed copy without opening a UI:

```bash
source .venv/bin/activate
jupyter nbconvert --to notebook --execute \
    --output executed_demo.ipynb \
    notebooks/pipeline_demo.ipynb
```

A non-zero exit code means a cell failed.

---

## How imports resolve

The first ("Setup") cell adds the **project root** to `sys.path`, so
`import analytics` works even though the notebook lives in `notebooks/`. No
`pip install` of the library itself is required — it is imported directly from
the source tree.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: .../sample_sales.csv` | Run `python data/generate_sample_data.py` from the project root. |
| `ModuleNotFoundError: analytics` | Make sure you launched Jupyter with the `.venv` active and ran the Setup cell. |
| Kernel not listed | Re-run the `ipykernel install` command above, then refresh Jupyter. |
| `ImportError: ... requires statsmodels/scipy` | `pip install -r requirements.txt` again; a dependency is missing. |
