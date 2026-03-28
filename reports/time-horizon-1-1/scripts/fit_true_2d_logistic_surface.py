"""
Fit a true 2D logistic success surface over release date and task duration.

Model:
    logit(P(success)) =
        alpha0 + alpha1 * date_z
        + beta0 * log_time_z + beta1 * date_z * log_time_z

This is the direct 2D analogue of the original 1D logistic methodology:
for any fixed release date, you get a logistic curve over task time; for any fixed
task time, you get a success curve over release date. Horizons at 50/80/90/... are
recovered by inverting the same surface, not by fitting a second regression.
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
    local_doubling_time_days,
    make_curve_table,
    make_standardization,
    save_dataframe,
    target_logit,
    horizon_minutes_from_linear_terms,
)


def surface_horizon_fn(
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


def success_fn(
    beta: np.ndarray,
    standardization: Standardization,
):
    def predict_success(date_num: float, task_minutes: float) -> float:
        date_z = (date_num - standardization.date_mean) / standardization.date_scale
        log_time = np.log2(task_minutes)
        log_time_z = (log_time - standardization.log_time_mean) / standardization.log_time_scale
        logit_value = (
            beta[0]
            + beta[1] * date_z
            + beta[2] * log_time_z
            + beta[3] * date_z * log_time_z
        )
        return float(1.0 / (1.0 + np.exp(-logit_value)))

    return predict_success


def make_surface_summary(
    runs: pd.DataFrame,
    percents: list[int],
    horizon_fn,
    ref_date: str,
) -> pd.DataFrame:
    ref_date_num = float(mdates.date2num(pd.Timestamp(ref_date)))
    rows: list[dict[str, str | float]] = []
    for agent, agent_runs in runs.groupby("alias", as_index=False):
        date_num = float(agent_runs["date_num"].iloc[0])
        release_date = pd.to_datetime(agent_runs["release_date"].iloc[0])
        row: dict[str, str | float] = {
            "agent": str(agent),
            "release_date": release_date.strftime("%Y-%m-%d"),
        }
        for p in percents:
            row[f"p{p}"] = horizon_fn(date_num, float(p))
        row["doubling_time_at_50_ref_days"] = local_doubling_time_days(
            horizon_fn, ref_date_num, 50.0
        )
        row["doubling_time_at_80_ref_days"] = local_doubling_time_days(
            horizon_fn, ref_date_num, 80.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("release_date")


def make_fixed_task_success_table(
    date_start: str,
    date_end: str,
    task_minutes: list[float],
    predict_success,
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for release_date in pd.date_range(date_start, date_end, freq="MS"):
        date_num = float(mdates.date2num(release_date))
        for minutes in task_minutes:
            rows.append(
                {
                    "release_date": release_date.strftime("%Y-%m-%d"),
                    "task_minutes": minutes,
                    "success_probability": predict_success(date_num, minutes),
                }
            )
    return pd.DataFrame(rows)


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
        default=str(REPORT_DIR / "data/wrangled/joint_models/true_2d_surface_summary.csv"),
    )
    parser.add_argument(
        "--curve-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/true_2d_surface_curve.csv"),
    )
    parser.add_argument(
        "--fixed-task-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/true_2d_surface_fixed_task_success.csv"),
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

    horizon_fn = surface_horizon_fn(beta, standardization)
    predict_success = success_fn(beta, standardization)
    summary = make_surface_summary(runs, args.percents, horizon_fn, args.ref_date)
    curve = make_curve_table("2023-01-01", "2031-03-01", args.percents, horizon_fn)
    fixed_task = make_fixed_task_success_table(
        "2023-01-01",
        "2031-03-01",
        [1, 5, 15, 60, 240, 480, 960],
        predict_success,
    )

    summary_path = save_dataframe(summary, args.summary_output)
    curve_path = save_dataframe(curve, args.curve_output)
    fixed_task_path = save_dataframe(fixed_task, args.fixed_task_output)

    ref_date_num = float(mdates.date2num(pd.Timestamp(args.ref_date)))
    print("True 2D logistic success surface")
    print(f"Rows: {len(runs)}")
    print(f"Agents: {runs['alias'].nunique()}")
    print("Coefficients:")
    print(f"  intercept={beta[0]:.6f}")
    print(f"  date_z={beta[1]:.6f}")
    print(f"  log_time_z={beta[2]:.6f}")
    print(f"  date_z:log_time_z={beta[3]:.6f}")
    for p in args.percents:
        print(f"  p={p}% horizon at ref date: {horizon_fn(ref_date_num, float(p)):.2f} minutes")
    print(f"Saved summary to {summary_path}")
    print(f"Saved curve to {curve_path}")
    print(f"Saved fixed-task success table to {fixed_task_path}")


if __name__ == "__main__":
    main()
