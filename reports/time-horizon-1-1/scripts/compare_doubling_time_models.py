"""
Compare doubling-time curves from:
1. The existing two-stage horizon regression.
2. The pooled one-stage joint logistic model over raw runs.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_model_utils import (
    DEFAULT_PERCENTS,
    DEFAULT_REF_DATE,
    DEFAULT_RELEASE_DATES_FILE,
    DEFAULT_RUNS_FILE,
    EXCLUDE_AGENTS,
    REPORT_DIR,
    apply_standardization,
    fit_logistic_model,
    load_joint_runs,
    local_doubling_time_days,
    make_standardization,
)
from regress_2d_horizon import doubling_time_at_p, fit_2d, load_and_reshape


def pooled_horizon_fn(beta: np.ndarray, standardization):
    def horizon(date_num: float, percent: float) -> float:
        date_z = (date_num - standardization.date_mean) / standardization.date_scale
        intercept_term = beta[0] + beta[1] * date_z
        slope_term = beta[2] + beta[3] * date_z
        if abs(slope_term) < 1e-12:
            return float("nan")
        target_logit = np.log(percent / (100.0 - percent))
        log_time_z = (target_logit - intercept_term) / slope_term
        log_time = (
            log_time_z * standardization.log_time_scale + standardization.log_time_mean
        )
        if not np.isfinite(log_time) or abs(log_time) > 40:
            return float("nan")
        return float(np.exp2(log_time))

    return horizon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        default=str(REPORT_DIR / "data/wrangled/logistic_fits/headline.csv"),
    )
    parser.add_argument("--runs-file", default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--release-dates-file", default=str(DEFAULT_RELEASE_DATES_FILE))
    parser.add_argument("--weighting-col", default="invsqrt_task_weight")
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--ref-date", default=DEFAULT_REF_DATE)
    parser.add_argument("--percents", nargs="+", type=int, default=DEFAULT_PERCENTS)
    parser.add_argument(
        "--output-file",
        default=str(REPORT_DIR / "plots/logistic/doubling_time_model_comparison.png"),
    )
    args = parser.parse_args()

    long = load_and_reshape(args.input_file, args.percents, exclude_agents=EXCLUDE_AGENTS)
    two_stage = fit_2d(long)
    two_stage_beta = two_stage["beta"]

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
    pooled_model = fit_logistic_model(X, y, weights, regularization=args.regularization)
    pooled_beta = pooled_model.coef_[0]
    pooled_horizon = pooled_horizon_fn(pooled_beta, standardization)

    ref_date_num = float(mdates.date2num(pd.Timestamp(args.ref_date)))
    p_range = np.linspace(50, 99.99, 600)
    dt_two_stage = np.array([doubling_time_at_p(two_stage_beta, p) for p in p_range])
    dt_pooled = np.array(
        [local_doubling_time_days(pooled_horizon, ref_date_num, p) for p in p_range]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.plot(p_range, dt_two_stage, color="#2166AC", linewidth=2.5, label="Two-stage")
    ax.plot(p_range, dt_pooled, color="#B2182B", linewidth=2.5, label="Pooled one-stage")
    for p in args.percents:
        ax.scatter([p], [doubling_time_at_p(two_stage_beta, p)], color="#2166AC", s=35)
        ax.scatter(
            [p],
            [local_doubling_time_days(pooled_horizon, ref_date_num, float(p))],
            color="#B2182B",
            s=35,
        )
    ax.set_xlabel("Reliability threshold (%)")
    ax.set_ylabel("Doubling time (days)")
    ax.set_title("Doubling Time Curves")
    ax.axvspan(99, 100, alpha=0.08, color="red", zorder=0)
    ax.text(99.45, np.nanmin(dt_two_stage) + 3, "extrapolation", color="red", alpha=0.7, ha="center")
    ax.set_xlim(49, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax2 = axes[1]
    diff = dt_pooled - dt_two_stage
    ax2.plot(p_range, diff, color="#762A83", linewidth=2.5)
    ax2.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Reliability threshold (%)")
    ax2.set_ylabel("Pooled - Two-stage (days)")
    ax2.set_title("Difference in Doubling Time")
    ax2.axvspan(99, 100, alpha=0.08, color="red", zorder=0)
    ax2.set_xlim(49, 100)
    ax2.grid(True, alpha=0.25)

    ax3 = axes[2]
    p_zoom = p_range[p_range >= 98.5]
    dt_two_stage_zoom = dt_two_stage[p_range >= 98.5]
    dt_pooled_zoom = dt_pooled[p_range >= 98.5]
    ax3.plot(p_zoom, dt_two_stage_zoom, color="#2166AC", linewidth=2.5, label="Two-stage")
    ax3.plot(p_zoom, dt_pooled_zoom, color="#B2182B", linewidth=2.5, label="Pooled one-stage")
    for p in [98, 99, 99.9, 99.99]:
        if p >= 98.5:
            ax3.scatter([p], [doubling_time_at_p(two_stage_beta, p)], color="#2166AC", s=35)
            ax3.scatter(
                [p],
                [local_doubling_time_days(pooled_horizon, ref_date_num, float(p))],
                color="#B2182B",
                s=35,
            )
    ax3.axvspan(99, 100, alpha=0.08, color="red", zorder=0)
    ax3.set_xlabel("Reliability threshold (%)")
    ax3.set_ylabel("Doubling time (days)")
    ax3.set_title("High-Nines Zoom")
    ax3.set_xlim(98.5, 100.0)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(f"Model Comparison at reference date {args.ref_date}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = pathlib.Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    print(f"Saved comparison plot to {output_file}")
    print(
        f"At 50%: two-stage={doubling_time_at_p(two_stage_beta, 50):.1f}d, "
        f"pooled={local_doubling_time_days(pooled_horizon, ref_date_num, 50):.1f}d"
    )
    print(
        f"At 80%: two-stage={doubling_time_at_p(two_stage_beta, 80):.1f}d, "
        f"pooled={local_doubling_time_days(pooled_horizon, ref_date_num, 80):.1f}d"
    )


if __name__ == "__main__":
    main()
