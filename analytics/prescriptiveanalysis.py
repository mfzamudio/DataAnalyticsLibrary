"""
prescriptiveanalysis
====================

STAGE 8 of the analytics lifecycle: **Prescriptive Analytics** -> "What should
we do?"

Goal of this stage
------------------
Go beyond forecasting to RECOMMEND actions. Prescriptive analytics combines a
predictive model (or known relationships) with business rules and/or
optimization to choose the best decision under constraints.

Methodology / steps
-------------------
    1. Define the objective (maximize revenue, minimize cost/risk).
    2. Define decision variables and constraints (budget, capacity).
    3. Generate candidate scenarios OR solve an optimization.
    4. Score / rank options and recommend the best.

This module provides three pragmatic, dependency-light patterns:

    * ``scenario_analysis``   : evaluate "what-if" inputs through a model.
    * ``recommend_by_rules``  : threshold-based next-best-action rules.
    * ``optimize_allocation`` : linear-programming budget allocation
                                (uses scipy.optimize.linprog).

NOTE: real prescriptive systems are domain-specific. Treat these as solid,
documented starting points to adapt, not turnkey decision engines.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


def scenario_analysis(
    pipeline,
    base_row: pd.DataFrame,
    *,
    variable: str,
    values: Sequence,
) -> pd.DataFrame:
    """
    "What-if" analysis: sweep one input variable through a fitted model and see
    how the prediction responds.

    Parameters
    ----------
    pipeline : fitted sklearn pipeline/estimator
        Anything with a ``.predict`` method (e.g. from
        :func:`analytics.predictiveanalysis.train_regressor`).
    base_row : pandas.DataFrame
        A single-row DataFrame of baseline feature values (the scenario's
        starting point). Must contain every column the pipeline expects.
    variable : str
        The feature to vary.
    values : sequence
        The candidate values to test for ``variable``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``<variable>`` and ``prediction``, one row per scenario.
    """
    if len(base_row) != 1:
        raise ValueError("base_row must be a single-row DataFrame.")

    rows = []
    for v in values:
        scenario = base_row.copy()
        scenario[variable] = v
        pred = pipeline.predict(scenario)[0]
        rows.append({variable: v, "prediction": pred})
    return pd.DataFrame(rows)


def recommend_by_rules(
    df: pd.DataFrame,
    rules: list[tuple[Callable[[pd.DataFrame], pd.Series], str]],
    *,
    default_action: str = "no_action",
    action_column: str = "recommended_action",
) -> pd.DataFrame:
    """
    Assign a recommended action to each row using ordered business rules.

    Rules are evaluated TOP TO BOTTOM; the first matching rule wins (like an
    if/elif chain). This is the transparent, explainable backbone of many
    "next best action" systems.

    Parameters
    ----------
    df : pandas.DataFrame
    rules : list of (condition, action)
        ``condition`` is a callable taking the DataFrame and returning a
        boolean Series (the row mask); ``action`` is the label to assign where
        the mask is True and no earlier rule already matched.
    default_action : str, default "no_action"
        Action assigned to rows that match no rule.
    action_column : str, default "recommended_action"
        Name of the output column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with the recommendation column added.

    Examples
    --------
    >>> rules = [
    ...     (lambda d: d["churn_risk"] > 0.8, "call_customer"),
    ...     (lambda d: d["churn_risk"] > 0.5, "send_discount"),
    ... ]
    >>> recommend_by_rules(df, rules)
    """
    out = df.copy()
    out[action_column] = default_action
    # Track which rows are still unassigned so earlier rules take precedence.
    unassigned = pd.Series(True, index=out.index)
    for condition, action in rules:
        mask = condition(out) & unassigned
        out.loc[mask, action_column] = action
        unassigned &= ~mask
    return out


def optimize_allocation(
    *,
    returns: Sequence[float],
    total_budget: float,
    min_alloc: Optional[Sequence[float]] = None,
    max_alloc: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Allocate a fixed budget across options to MAXIMIZE total expected return,
    via linear programming.

    Solves:  maximize  sum(returns_i * x_i)
             subject to sum(x_i) <= total_budget,
                        min_alloc_i <= x_i <= max_alloc_i

    Wraps ``scipy.optimize.linprog`` (which minimizes, so we negate the
    objective).

    Parameters
    ----------
    returns : sequence of float
        Expected return PER UNIT allocated to each option (e.g. ROI per dollar
        of marketing spend on each channel).
    total_budget : float
        Total budget to distribute.
    min_alloc : sequence of float, optional
        Lower bound per option. Defaults to all zeros.
    max_alloc : sequence of float, optional
        Upper bound per option. Defaults to ``total_budget`` each (unbounded
        within the budget).
    labels : sequence of str, optional
        Names for the options (for a readable result).

    Returns
    -------
    pandas.DataFrame
        Columns: ``option``, ``allocation``, ``expected_return``. Includes a
        ``status`` attribute via ``df.attrs["status"]`` describing the solver
        outcome.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    try:
        from scipy.optimize import linprog
    except ImportError as exc:  # pragma: no cover
        raise ImportError("optimize_allocation requires scipy.") from exc

    n = len(returns)
    # Use a distinct name so the ndarray doesn't shadow the Sequence parameter.
    returns_arr = np.asarray(returns, dtype=float)

    # linprog MINIMIZES c @ x; negate returns to MAXIMIZE.
    c = -returns_arr

    # Single inequality constraint: sum(x) <= total_budget.
    A_ub = [np.ones(n)]
    b_ub = [total_budget]

    # Per-option bounds.
    lo = list(min_alloc) if min_alloc is not None else [0.0] * n
    hi = list(max_alloc) if max_alloc is not None else [total_budget] * n
    bounds = list(zip(lo, hi))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    names = list(labels) if labels is not None else [f"option_{i}" for i in range(n)]
    allocation = res.x if res.success else np.zeros(n)
    out = pd.DataFrame(
        {
            "option": names,
            "allocation": np.round(allocation, 4),
            "expected_return": np.round(allocation * returns_arr, 4),
        }
    )
    # Stash solver metadata where it does not clutter the rows.
    out.attrs["status"] = res.message
    out.attrs["success"] = bool(res.success)
    return out
