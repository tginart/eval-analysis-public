"""
Single-stage joint logistic fit with regularized agent-level deviations.

Model:
    logit(P(success)) =
        a + b * date_z + c * log_time_z + d * date_z * log_time_z
        + u_agent + v_agent * log_time_z

The `u_agent` and `v_agent` terms are fit jointly with the raw run outcomes in
one logistic model, so this still avoids the "regression on regression" setup.
This is not a full Bayesian hierarchical model, but it is a practical
approximation using L2-regularized agent intercept/slope deviations.
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
    target_logit,
    horizon_minutes_from_linear_terms,
)


def build_agent_effect_design(
    runs: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    agents = sorted(runs["alias"].unique())

    parts = [
        np.ones((len(runs), 1)),
        runs[["date_z", "log_time_z"]].to_numpy(),
        (runs["date_z"] * runs["log_time_z"]).to_numpy().reshape(-1, 1),
    ]
    feature_names = ["intercept", "date_z", "log_time_z", "date_z:log_time_z"]

    log_time = runs["log_time_z"].to_numpy(dtype=float).reshape(-1, 1)
    for agent in agents:
        indicator = (runs["alias"] == agent).to_numpy(dtype=float).reshape(-1, 1)
        centered_indicator = indicator - indicator.mean()
        parts.append(centered_indicator)
        parts.append(centered_indicator * log_time)
        feature_names.append(f"agent_intercept[{agent}]")
        feature_names.append(f"agent_slope[{agent}]")

    return np.hstack(parts), feature_names


def horizon_fn_with_agent_effects(
    beta: np.ndarray,
    feature_names: list[str],
    standardization: Standardization,
):
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def horizon(date_num: float, percent: float, agent: str | None = None) -> float:
        date_z = (date_num - standardization.date_mean) / standardization.date_scale
        intercept_term = beta[name_to_idx["intercept"]] + beta[name_to_idx["date_z"]] * date_z
        slope_term = beta[name_to_idx["log_time_z"]] + beta[name_to_idx["date_z:log_time_z"]] * date_z

        if agent is not None:
            intercept_name = f"agent_intercept[{agent}]"
            slope_name = f"agent_slope[{agent}]"
            intercept_term += beta[name_to_idx[intercept_name]]
            slope_term += beta[name_to_idx[slope_name]]

        return horizon_minutes_from_linear_terms(
            target_logit(percent),
            intercept_term,
            slope_term,
            standardization,
        )

    return horizon


def make_agent_summary_table(
    runs: pd.DataFrame,
    percents: list[int],
    horizon_fn,
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = []
    for agent, agent_runs in runs.groupby("alias", as_index=False):
        date_num = float(agent_runs["date_num"].iloc[0])
        release_date = pd.to_datetime(agent_runs["release_date"].iloc[0])
        row: dict[str, str | float] = {
            "agent": str(agent),
            "release_date": release_date.strftime("%Y-%m-%d"),
        }
        for p in percents:
            row[f"p{p}"] = horizon_fn(date_num, float(p), str(agent))
            row[f"population_p{p}"] = horizon_fn(date_num, float(p), None)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("release_date")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-file", default=str(DEFAULT_RUNS_FILE))
    parser.add_argument("--release-dates-file", default=str(DEFAULT_RELEASE_DATES_FILE))
    parser.add_argument("--weighting-col", default="invsqrt_task_weight")
    parser.add_argument("--regularization", type=float, default=2e-2)
    parser.add_argument("--ref-date", default=DEFAULT_REF_DATE)
    parser.add_argument("--percents", nargs="+", type=int, default=DEFAULT_PERCENTS)
    parser.add_argument(
        "--summary-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/joint_logistic_agent_effects_summary.csv"),
    )
    parser.add_argument(
        "--curve-output",
        default=str(REPORT_DIR / "data/wrangled/joint_models/joint_logistic_agent_effects_curve.csv"),
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

    X, feature_names = build_agent_effect_design(runs)
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

    horizon_fn = horizon_fn_with_agent_effects(
        beta=beta,
        feature_names=feature_names,
        standardization=standardization,
    )
    summary = make_agent_summary_table(runs, args.percents, horizon_fn)
    curve = make_curve_table(
        "2023-01-01",
        "2031-03-01",
        args.percents,
        lambda date_num, percent: horizon_fn(date_num, percent, None),
    )

    summary_path = save_dataframe(summary, args.summary_output)
    curve_path = save_dataframe(curve, args.curve_output)

    ref_date_num = float(mdates.date2num(pd.Timestamp(args.ref_date)))
    print("Joint logistic fit with regularized agent effects")
    print(f"Rows: {len(runs)}")
    print(f"Agents: {runs['alias'].nunique()}")
    print(f"Weighted BCE: {bce:.4f}")
    print("Population-level horizons at reference date:")
    for p in args.percents:
        print(f"  p={p}% horizon at ref date: {horizon_fn(ref_date_num, float(p), None):.2f} minutes")
    print(f"Saved summary to {summary_path}")
    print(f"Saved curve to {curve_path}")


if __name__ == "__main__":
    main()
