"""
Single-stage joint logistic fit over raw runs.

Model:
    logit(P(success)) = a + b * date_z + c * log_time_z + d * date_z * log_time_z

This avoids the "regression on regression" structure by fitting the raw run
outcomes directly. It is the simplest pooled joint model: all models share the
same surface, and release date shifts both baseline success and the effect of
task duration.
"""

from __future__ import annotations

import argparse

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from joint_model_utils import (
    DEFAULT_PERCENTS,
    DEFAULT_REF_DATE,
    DEFAULT_RELEASE_DATES_FILE,
    DEFAULT_RUNS_FILE,
    EXCLUDE_AGENTS,
    REPORT_DIR,
    Standardization,
    apply_standardization,
    ensure_requested_percents,
    fit_logistic_model,
    load_joint_runs,
    make_curve_table,
    make_standardization,
    save_dataframe,
    make_summary_table,
    target_logit,
    horizon_minutes_from_linear_terms,
)


def pooled_horizon_fn(
    beta: np.ndarray,
    standardization: Standardization,
):
    def horizon(date_num: float, percent: float) -> float:
        date_z = (date_num - standardization.date_mean) / standardization.date_scale
        intercept_term = beta[0] + beta[1] * date_z
        slope_term = beta[2] + beta[3] * date_z
        return horizon_minutes_from_linear_terms(
            target_logit(percent),
            intercept_term,
            slope_term,
            standardization,
        )

    return horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-file", default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--release-dates-file", default=str(DEFAULT_RELEASE_DATES_FILE))
    parser.add_argument("--weighting-col", default="invsqrt_task_weight")
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--ref-date", default=DEFAULT_REF_DATE)
    parser.add_argument("--percents", nargs="+", type=int, default=DEFAULT_PERCENTS)
    parser.add_argument(
        "--summary-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/joint_logistic_pooled_summary.csv"),
    )
    parser.add_argument(
        "--curve-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/joint_logistic_pooled_curve.csv"),
    )
    args = parser.parse_args()

    ensure_requested_percents(args.percents)
    runs = load_joint_runs(
        runs_file=args.runs_file,
        release_dates_file=args.release_dates_file,
        weighting_col=args.weighting_col,
        exclude_agents=EXCLUDE_AGENTS,
    )
    standardization = make_standardization(runs)
    runs = apply_standardization(runs, standardization)

    X = np.column_stack(
        [
            np.ones(len(runs)),
            runs["date_z"].to_numpy(),
            runs["log_time_z"].to_numpy(),
            (runs["date_z"] * runs["log_time_z"]).to_numpy(),
        ]
    )
    y = runs["score_binarized"].to_numpy(dtype=float)
    weights = runs["weight"].to_numpy(dtype=float)

    model = fit_logistic_model(X, y, weights, regularization=args.regularization)
    beta = model.coef_[0]
    preds = model.predict_proba(X)[:, 1]
    eps = 1e-12
    bce = -np.average(
        y * np.log(np.clip(preds, eps, 1 - eps))
        + (1 - y) * np.log(np.clip(1 - preds, eps, 1 - eps)),
        weights=weights,
    )

    horizon_fn = pooled_horizon_fn(beta, standardization)
    summary = make_summary_table(runs, args.percents, horizon_fn, args.ref_date)
    curve = make_curve_table("2023-01-01", "2031-03-01", args.percents, horizon_fn)

    summary_path = save_dataframe(summary, args.summary_output)
    curve_path = save_dataframe(curve, args.curve_output)

    ref_date_num = float(mdates.date2num(pd.Timestamp(args.ref_date)))
    print("Pooled joint logistic fit")
    print(f"Rows: {len(runs)}")
    print(f"Agents: {runs['alias'].nunique()}")
    print(f"Weighted BCE: {bce:.4f}")
    print("Coefficients:")
    print(f"  intercept={beta[0]:.6f}")
    print(f"  date_z={beta[1]:.6f}")
    print(f"  log_time_z={beta[2]:.6f}")
    print(f"  date_z:log_time_z={beta[3]:.6f}")
    print(f"Reference date: {args.ref_date}")
    for p in args.percents:
        print(f"  p={p}% horizon at ref date: {horizon_fn(ref_date_num, float(p)):.2f} minutes")
    print(f"Saved summary to {summary_path}")
    print(f"Saved curve to {curve_path}")


if __name__ == "__main__":
    main()
