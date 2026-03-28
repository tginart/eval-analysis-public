from __future__ import annotations

import pathlib
from dataclasses import dataclass

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
ROOT_DIR = REPORT_DIR.parent.parent

DEFAULT_RUNS_FILE = REPORT_DIR / "data/raw/runs.jsonl"
DEFAULT_RELEASE_DATES_FILE = ROOT_DIR / "data/external/release_dates.yaml"
DEFAULT_PERCENTS = [50, 80, 90, 95, 98, 99]
DEFAULT_REF_DATE = "2026-02-05"

# Mirror the exclusions used in the existing jaggedness analysis so comparisons stay aligned.
EXCLUDE_AGENTS = [
    "Claude 3 Opus (Inspect)",
    "GPT-4 Turbo (Inspect)",
    "Claude 4 Opus (Inspect)",
    "GPT-5.1-Codex-Max (Inspect)",
    "Claude 4.1 Opus (Inspect)",
]


@dataclass
class Standardization:
    date_mean: float
    date_scale: float
    log_time_mean: float
    log_time_scale: float


def load_joint_runs(
    runs_file: pathlib.Path | str,
    release_dates_file: pathlib.Path | str,
    weighting_col: str,
    exclude_agents: list[str] | None = None,
) -> pd.DataFrame:
    runs_file = pathlib.Path(runs_file)
    release_dates_file = pathlib.Path(release_dates_file)
    runs = pd.read_json(runs_file, lines=True)
    release_dates = yaml.safe_load(release_dates_file.read_text())["date"]

    runs = runs.copy()
    runs = runs[runs["alias"] != "human"]
    if exclude_agents:
        runs = runs[~runs["alias"].isin(exclude_agents)]

    runs["release_date"] = pd.to_datetime(runs["alias"].map(release_dates), errors="coerce")
    runs = runs.dropna(subset=["release_date", "human_minutes", "score_binarized", weighting_col])
    runs = runs[runs["human_minutes"] > 0].copy()

    runs["date_num"] = mdates.date2num(runs["release_date"])
    runs["log2_human_minutes"] = np.log2(runs["human_minutes"])
    runs["weight"] = runs[weighting_col].astype(float)
    runs["weight"] = runs["weight"] / runs["weight"].sum()
    return runs


def make_standardization(runs: pd.DataFrame) -> Standardization:
    date_values = runs["date_num"].to_numpy(dtype=float)
    log_time_values = runs["log2_human_minutes"].to_numpy(dtype=float)
    date_scale = float(date_values.std())
    log_time_scale = float(log_time_values.std())
    if date_scale == 0 or log_time_scale == 0:
        raise ValueError("Need variation in both release date and task duration.")
    return Standardization(
        date_mean=float(date_values.mean()),
        date_scale=date_scale,
        log_time_mean=float(log_time_values.mean()),
        log_time_scale=log_time_scale,
    )


def apply_standardization(
    runs: pd.DataFrame,
    standardization: Standardization,
) -> pd.DataFrame:
    standardized = runs.copy()
    standardized["date_z"] = (
        standardized["date_num"] - standardization.date_mean
    ) / standardization.date_scale
    standardized["log_time_z"] = (
        standardized["log2_human_minutes"] - standardization.log_time_mean
    ) / standardization.log_time_scale
    return standardized


def fit_logistic_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> LogisticRegression:
    model = LogisticRegression(
        fit_intercept=False,
        C=1 / regularization,
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(X, y, sample_weight=weights)
    return model


def horizon_minutes_from_linear_terms(
    target_logit: float,
    intercept_term: float,
    slope_term: float,
    standardization: Standardization,
) -> float:
    if abs(slope_term) < 1e-12:
        return float("nan")
    log_time_z = (target_logit - intercept_term) / slope_term
    log_time = (
        log_time_z * standardization.log_time_scale + standardization.log_time_mean
    )
    if not np.isfinite(log_time) or abs(log_time) > 40:
        return float("nan")
    return float(np.exp2(log_time))


def local_doubling_time_days(
    horizon_fn,
    date_num: float,
    percent: float,
    eps_days: float = 14.0,
) -> float:
    horizon_now = horizon_fn(date_num, percent)
    horizon_next = horizon_fn(date_num + eps_days, percent)
    if horizon_now <= 0 or horizon_next <= 0:
        return float("nan")
    d_log_h = (np.log(horizon_next) - np.log(horizon_now)) / eps_days
    if abs(d_log_h) < 1e-12:
        return float("nan")
    return float(np.log(2) / d_log_h)


def make_summary_table(
    runs: pd.DataFrame,
    percents: list[int],
    horizon_fn,
    ref_date: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    ref_date_num = mdates.date2num(pd.Timestamp(ref_date))
    for agent, agent_runs in runs.groupby("alias", as_index=False):
        release_date = pd.to_datetime(agent_runs["release_date"].iloc[0])
        date_num = float(agent_runs["date_num"].iloc[0])
        row: dict[str, float | str] = {
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
        row["doubling_time_at_98_ref_days"] = local_doubling_time_days(
            horizon_fn, ref_date_num, 98.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("release_date")


def make_curve_table(
    date_start: str,
    date_end: str,
    percents: list[int],
    horizon_fn,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for release_date in pd.date_range(date_start, date_end, freq="MS"):
        date_num = float(mdates.date2num(release_date))
        for p in percents:
            rows.append(
                {
                    "release_date": release_date.strftime("%Y-%m-%d"),
                    "percent": p,
                    "horizon_minutes": horizon_fn(date_num, float(p)),
                }
            )
    return pd.DataFrame(rows)


def percent_horizon_columns(percents: list[int]) -> list[str]:
    return [f"p{p}" for p in percents]


def ensure_requested_percents(percents: list[int]) -> None:
    for p in percents:
        if not 0 < p < 100:
            raise ValueError(f"Reliability percents must lie in (0, 100), got {p}")


def target_logit(percent: float) -> float:
    return float(logit(percent / 100.0))


def save_dataframe(df: pd.DataFrame, output_path: pathlib.Path | str) -> pathlib.Path:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
