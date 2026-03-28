"""
2D regression of time horizon over (release_date, reliability_level).

Model: log(time_horizon) = β0 + β1*date_num + β2*logit(p) + β3*date_num*logit(p)

This allows continuous interpolation of both the intercept and doubling time
across arbitrary reliability thresholds, not just the fixed ones (50, 80, 90, ...).
"""

import argparse
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.special import logit


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent

EXCLUDE_AGENTS = [
    "Claude 3 Opus (Inspect)",
    "GPT-4 Turbo (Inspect)",
    "Claude 4 Opus (Inspect)",
    "GPT-5.1-Codex-Max (Inspect)",
    "Claude 4.1 Opus (Inspect)",
]


def load_and_reshape(
    csv_path: str,
    percents: list[int],
    exclude_agents: list[str] | None = None,
) -> pd.DataFrame:
    """Reshape wide-format logistic fits into long-format (agent, date, p, horizon)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["release_date"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["date_num"] = mdates.date2num(df["release_date"])

    if exclude_agents:
        df = df[~df["agent"].isin(exclude_agents)]

    missing_cols = []
    rows = []
    for p in percents:
        col = f"p{p}"
        if col not in df.columns:
            missing_cols.append(col)
            continue
        sub = df[["agent", "release_date", "date_num", col]].copy()
        sub = sub.rename(columns={col: "horizon_min"})
        sub["percent"] = p
        sub = sub[sub["horizon_min"] > 0]
        rows.append(sub)

    if missing_cols:
        raise ValueError(
            f"Missing requested reliability columns in {csv_path}: {', '.join(missing_cols)}"
        )

    long = pd.concat(rows, ignore_index=True)
    long["log_horizon"] = np.log(long["horizon_min"])
    long["logit_p"] = logit(long["percent"] / 100.0)
    return long


def fit_2d(long: pd.DataFrame) -> dict:
    """Fit log(horizon) = β0 + β1*date + β2*logit(p) + β3*date*logit(p)."""
    X = np.column_stack([
        np.ones(len(long)),
        long["date_num"].values,
        long["logit_p"].values,
        long["date_num"].values * long["logit_p"].values,
    ])
    y = long["log_horizon"].values

    # OLS
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        "beta": beta,
        "r_squared": r_squared,
        "n": len(y),
    }


def doubling_time_at_p(beta: np.ndarray, p: float) -> float:
    """Doubling time in days at reliability level p (0-100)."""
    lp = logit(p / 100.0)
    slope = beta[1] + beta[3] * lp  # d(log_horizon)/d(date_num)
    return np.log(2) / slope  # date_num is in days


def intercept_at_p(beta: np.ndarray, p: float, ref_date_num: float) -> float:
    """log(horizon) at a reference date for reliability p."""
    lp = logit(p / 100.0)
    return beta[0] + beta[1] * ref_date_num + beta[2] * lp + beta[3] * ref_date_num * lp


def plot_results(
    long: pd.DataFrame,
    beta: np.ndarray,
    r_squared: float,
    output_path: str,
    percents: list[int],
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left panel: trendlines from 2D model at each percent ---
    ax = axes[0]
    cmap = plt.cm.RdYlGn_r
    norm = Normalize(vmin=40, vmax=100)

    date_range = pd.date_range("2022-11-01", "2026-07-01", freq="MS")
    date_nums = mdates.date2num(date_range)

    # Plot continuous family of curves
    continuous_percents = list(range(50, 100, 1))
    for p in continuous_percents:
        lp = logit(p / 100.0)
        log_h = beta[0] + beta[1] * date_nums + beta[2] * lp + beta[3] * date_nums * lp
        alpha = 0.15 if p not in percents else 0.0  # hide labeled ones from background
        if alpha > 0:
            ax.plot(date_range, np.exp(log_h), color=cmap(norm(p)), alpha=alpha, linewidth=0.8)

    # Overlay the specific percents with labels
    for p in percents:
        lp = logit(p / 100.0)
        log_h = beta[0] + beta[1] * date_nums + beta[2] * lp + beta[3] * date_nums * lp
        color = cmap(norm(p))
        dt = doubling_time_at_p(beta, p)
        ax.plot(date_range, np.exp(log_h), color=color, linewidth=2.5, label=f"{p}% (DT: {dt:.0f}d)")

    # Scatter actual data points
    for p in percents:
        sub = long[long["percent"] == p]
        color = cmap(norm(p))
        ax.scatter(sub["release_date"], sub["horizon_min"], color=color, s=20, alpha=0.4, zorder=5)

    ax.set_yscale("log")
    ax.set_ylabel("Task time (for humans) at given success rate")
    ax.set_xlabel("Model release date")
    ax.set_title("2D Regression: Time Horizon by Date & Reliability")

    # y-axis formatting
    tick_vals = [1/60, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1920]
    tick_labels = ["1s", "30s", "1m", "2m", "4m", "8m", "15m", "30m", "1h", "2h", "4h", "8h", "16h", "32h"]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(0.5, 1920)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # --- Right panel: doubling time as function of reliability ---
    ax2 = axes[1]
    p_range = np.linspace(50, 99.5, 200)
    dt_range = [doubling_time_at_p(beta, p) for p in p_range]
    ax2.plot(p_range, dt_range, color="steelblue", linewidth=2.5)

    # Mark the fixed percents
    for p in percents:
        dt = doubling_time_at_p(beta, p)
        ax2.scatter([p], [dt], color=cmap(norm(p)), s=80, zorder=5, edgecolors="black", linewidths=0.5)
        ax2.annotate(f"{dt:.0f}d", (p, dt), textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax2.set_xlabel("Reliability threshold (%)")
    ax2.set_ylabel("Doubling time (days)")
    ax2.set_title("How Doubling Time Varies with Reliability")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(45, 100)

    fig.suptitle(
        f"2D Joint Regression: log(horizon) ~ date × logit(reliability)   (R² = {r_squared:.3f})",
        fontsize=13,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_hero(
    long: pd.DataFrame,
    beta: np.ndarray,
    r_squared: float,
    output_path: str,
    percents: list[int],
):
    """Clean hero figure: trendlines at multiple reliability levels with scatter at 50%."""
    fig, ax = plt.subplots(figsize=(12, 8))

    date_range = pd.date_range("2022-11-01", "2027-06-01", freq="MS")
    date_nums = mdates.date2num(date_range)

    # Distinct, colorblind-friendlier palette
    colors = {
        50: "#2166AC",   # strong blue
        80: "#4DAC26",   # green
        90: "#F5A623",   # amber
        95: "#E8451E",   # red-orange
        98: "#9E1A8A",   # purple
        99: "#888888",   # dark gray
    }

    # Faint interpolated background lines
    for p in range(50, 100):
        if p in percents:
            continue
        lp = logit(p / 100.0)
        log_h = beta[0] + beta[1] * date_nums + beta[2] * lp + beta[3] * date_nums * lp
        frac = (p - 50) / 50.0
        r = int(0x21 + frac * (0x88 - 0x21))
        g = int(0x66 + frac * (0x88 - 0x66))
        b = int(0xAC + frac * (0x88 - 0xAC))
        ax.plot(date_range, np.exp(log_h), color=f"#{r:02x}{g:02x}{b:02x}",
                alpha=0.08, linewidth=0.7)

    # Main trendlines — solid for data range, dashed for projection
    last_data_date = long["release_date"].max()
    date_range_solid = date_range[date_range <= last_data_date]
    date_range_dashed = date_range[date_range >= last_data_date]
    date_nums_solid = mdates.date2num(date_range_solid)
    date_nums_dashed = mdates.date2num(date_range_dashed)

    for p in percents:
        lp = logit(p / 100.0)
        color = colors.get(p, "black")
        # Solid portion (fit range)
        log_h_s = beta[0] + beta[1] * date_nums_solid + beta[2] * lp + beta[3] * date_nums_solid * lp
        dt = doubling_time_at_p(beta, p)
        # Current best for this threshold
        sub = long[long["percent"] == p]
        best_min = sub["horizon_min"].max() if len(sub) > 0 else 0
        if best_min >= 60:
            best_str = f"{best_min/60:.0f}h"
        elif best_min >= 1:
            best_str = f"{best_min:.0f}m"
        else:
            best_str = f"{best_min*60:.0f}s"
        ax.plot(date_range_solid, np.exp(log_h_s), color=color, linewidth=2.5, zorder=3,
                label=f"{p}% (doubling time: {dt:.0f}d, best today: {best_str})")
        # Dashed portion (projection)
        log_h_d = beta[0] + beta[1] * date_nums_dashed + beta[2] * lp + beta[3] * date_nums_dashed * lp
        ax.plot(date_range_dashed, np.exp(log_h_d), color=color, linewidth=2.5, zorder=3,
                linestyle="dashed", alpha=0.6)

    # Scatter actual data on all lines
    for p in percents:
        sub = long[long["percent"] == p]
        color = colors.get(p, "black")
        ax.scatter(sub["release_date"], sub["horizon_min"],
                   color=color, s=14, alpha=0.4, zorder=4,
                   edgecolors="none")

    # Labels attached to each line near the right edge
    label_date = pd.Timestamp("2026-10-01")
    label_date_num = mdates.date2num(label_date)
    for p in percents:
        lp = logit(p / 100.0)
        log_h = beta[0] + beta[1] * label_date_num + beta[2] * lp + beta[3] * label_date_num * lp
        h = np.exp(log_h)
        dt = doubling_time_at_p(beta, p)
        color = colors.get(p, "black")
        # Compute angle of the line for rotation
        slope = beta[1] + beta[3] * lp
        ax.annotate(
            f" {p}% ({dt:.0f}d)",
            (label_date, h),
            fontsize=10, fontweight="bold", color=color,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7),
        )

    # Y-axis
    ax.set_yscale("log")
    tick_vals = [1/60, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1920]
    tick_labels = ["1s", "30s", "1m", "2m", "4m", "8m", "15m", "30m",
                   "1h", "2h", "4h", "8h", "16h", "32h"]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels, fontsize=11)
    ax.set_ylim(0.3, 6000)

    # X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_xlim(pd.Timestamp("2022-11-01"), pd.Timestamp("2027-06-01"))

    # Labels
    ax.set_ylabel("Time horizon (human-equivalent task length)", fontsize=12)
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_title(
        "AI Agent Time Horizons Across Reliability Levels\n"
        "Task length AI agents can complete at each success rate, with exponential fits",
        fontsize=14, fontweight="bold", pad=8,
    )

    ax.grid(True, alpha=0.2, which="major")
    ax.grid(True, alpha=0.08, which="minor")
    ax.legend(loc="upper left", fontsize=11.5, framealpha=0.9, edgecolor="none")

    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved hero plot to {output_path}")


def plot_jaggedness_hero(
    long: pd.DataFrame,
    beta: np.ndarray,
    output_path: str,
):
    """Non-log plot of 80% vs 98% exponential trendlines showing divergence."""
    fig, ax = plt.subplots(figsize=(12, 7))

    last_data_date = long["release_date"].max()

    # Date ranges
    date_range_solid = pd.date_range("2023-06-01", last_data_date, freq="MS")
    date_range_dashed = pd.date_range(last_data_date, "2031-03-01", freq="MS")

    for dates, ls, alpha in [(date_range_solid, "solid", 1.0), (date_range_dashed, "dashed", 0.6)]:
        dns = mdates.date2num(dates)
        for p, color, label in [(80, "#4DAC26", "80% success rate"), (98, "#9E1A8A", "98% success rate")]:
            lp = logit(p / 100.0)
            log_h = beta[0] + beta[1] * dns + beta[2] * lp + beta[3] * dns * lp
            h_hours = np.exp(log_h) / 1440.0  # convert minutes to days
            lbl = label if ls == "solid" else None
            ax.plot(dates, h_hours, color=color, linewidth=2.5, linestyle=ls, alpha=alpha, label=lbl)

    # Scatter actual data
    for p, color in [(80, "#4DAC26"), (98, "#9E1A8A")]:
        sub = long[long["percent"] == p]
        ax.scatter(sub["release_date"], sub["horizon_min"] / 1440.0,
                   color=color, s=25, alpha=0.5, zorder=5, edgecolors="none")

    # Shade the gap between the two curves across the full range
    all_dates = pd.date_range("2023-06-01", "2031-03-01", freq="MS")
    all_dns = mdates.date2num(all_dates)
    lp80 = logit(80 / 100.0)
    lp98 = logit(98 / 100.0)
    h80 = np.exp(beta[0] + beta[1] * all_dns + beta[2] * lp80 + beta[3] * all_dns * lp80) / 1440.0
    h98 = np.exp(beta[0] + beta[1] * all_dns + beta[2] * lp98 + beta[3] * all_dns * lp98) / 1440.0
    ax.fill_between(all_dates, h98, h80, alpha=0.08, color="red", label="Jaggedness gap")

    # Vertical line at present
    ax.axvline(last_data_date, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(last_data_date, ax.get_ylim()[1] * 0.02, "  Today",
            fontsize=10, color="black", alpha=0.6, va="bottom")

    # Annotate the gap at a few points
    for date_str in ["2026-02-01", "2028-01-01", "2031-01-01"]:
        d = mdates.date2num(pd.Timestamp(date_str))
        v80 = np.exp(beta[0] + beta[1] * d + beta[2] * lp80 + beta[3] * d * lp80) / 1440.0
        v98 = np.exp(beta[0] + beta[1] * d + beta[2] * lp98 + beta[3] * d * lp98) / 1440.0
        ratio = v80 / v98
        mid = np.sqrt(v80 * v98)  # geometric mean for placement
        ax.annotate(f"{ratio:.0f}x gap",
                    (pd.Timestamp(date_str), mid),
                    fontsize=11, fontweight="bold", color="red", alpha=0.7,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_ylabel("Time horizon (days)", fontsize=12)
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_title(
        "The Jaggedness Gap: 80% vs 98% Time Horizon\n"
        "What models can sometimes do vs. what they can reliably do",
        fontsize=14, fontweight="bold",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlim(pd.Timestamp("2023-06-01"), pd.Timestamp("2031-03-01"))
    ax.set_ylim(0, 200)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved jaggedness hero to {output_path}")


def plot_banner_heatmap(
    beta: np.ndarray,
    r_squared: float,
    output_path: str,
    ref_date: str = "2026-02-05",
):
    """5:2 banner version of the calculator heatmap."""
    ref_date_num = mdates.date2num(pd.Timestamp(ref_date))

    reliabilities = [50, 70, 80, 90, 95, 98, 99, 99.9]
    horizons_min = [5, 30, 60, 240, 480, 960, 4800, 20160]
    horizon_labels = [format_horizon(h) for h in horizons_min]

    grid = np.zeros((len(horizons_min), len(reliabilities)))
    for i, h in enumerate(horizons_min):
        for j, p in enumerate(reliabilities):
            grid[i, j] = days_until(beta, p, h, ref_date_num)

    fig, ax = plt.subplots(figsize=(22, 8))
    display_grid = np.clip(grid.copy(), 0, None)

    im = ax.imshow(display_grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=3000)

    for i in range(len(horizons_min)):
        for j in range(len(reliabilities)):
            val = grid[i, j]
            if val <= 0:
                text = "TODAY"
                color = "white"
                weight = "bold"
            else:
                target_date = mdates.num2date(ref_date_num + val)
                text = f"{format_days(val)}\n{target_date.strftime('%b %Y')}"
                color = "white" if val > 1500 else "black"
                weight = "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    color=color, fontweight=weight)

    X, Y = np.meshgrid(range(len(reliabilities)), range(len(horizons_min)))
    isocline_years = [0, 1, 2, 3, 5]
    isocline_days = [y * 365 for y in isocline_years]
    isocline_labels = ["Now", "+1 year", "+2 years", "+3 years", "+5 years"]
    contours = ax.contour(X, Y, display_grid, levels=isocline_days,
                          colors="black", linewidths=2.5, linestyles="solid", alpha=0.8)
    ax.clabel(contours, fmt={d: l for d, l in zip(isocline_days, isocline_labels)},
              fontsize=11, inline=True, inline_spacing=12, use_clabeltext=True)
    for txt in ax.texts:
        if txt.get_text() in isocline_labels:
            txt.set_fontweight("bold")
            txt.set_bbox(dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="black", alpha=0.85, linewidth=0.5))

    ax.set_xticks(range(len(reliabilities)))
    ax.set_xticklabels([f"{p}%" for p in reliabilities], fontsize=13)
    ax.set_yticks(range(len(horizons_min)))
    ax.set_yticklabels(horizon_labels, fontsize=13)
    ax.set_xlabel("Target reliability (success rate) →", fontsize=14)
    ax.set_ylabel("Target time horizon", fontsize=14)
    ax.set_title(
        "When Will AI Agents Reach Each Capability Target?",
        fontsize=18, fontweight="bold", pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Days from now", fontsize=12)
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved banner heatmap to {output_path}")


def plot_banner_jaggedness(
    long: pd.DataFrame,
    beta: np.ndarray,
    output_path: str,
):
    """5:2 banner version of the jaggedness gap."""
    fig, ax = plt.subplots(figsize=(20, 8))

    last_data_date = long["release_date"].max()

    for dates, ls, alpha in [
        (pd.date_range("2023-06-01", last_data_date, freq="MS"), "solid", 1.0),
        (pd.date_range(last_data_date, "2031-03-01", freq="MS"), "dashed", 0.6),
    ]:
        dns = mdates.date2num(dates)
        for p, color, label in [(80, "#4DAC26", "80% success rate"), (98, "#9E1A8A", "98% success rate")]:
            lp = logit(p / 100.0)
            log_h = beta[0] + beta[1] * dns + beta[2] * lp + beta[3] * dns * lp
            h_days = np.exp(log_h) / 1440.0
            lbl = label if ls == "solid" else None
            ax.plot(dates, h_days, color=color, linewidth=3, linestyle=ls, alpha=alpha, label=lbl)

    for p, color in [(80, "#4DAC26"), (98, "#9E1A8A")]:
        sub = long[long["percent"] == p]
        ax.scatter(sub["release_date"], sub["horizon_min"] / 1440.0,
                   color=color, s=25, alpha=0.5, zorder=5, edgecolors="none")

    all_dates = pd.date_range("2023-06-01", "2031-03-01", freq="MS")
    all_dns = mdates.date2num(all_dates)
    lp80 = logit(80 / 100.0)
    lp98 = logit(98 / 100.0)
    h80 = np.exp(beta[0] + beta[1] * all_dns + beta[2] * lp80 + beta[3] * all_dns * lp80) / 1440.0
    h98 = np.exp(beta[0] + beta[1] * all_dns + beta[2] * lp98 + beta[3] * all_dns * lp98) / 1440.0
    ax.fill_between(all_dates, h98, h80, alpha=0.08, color="red", label="Jaggedness gap")

    ax.axvline(last_data_date, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(last_data_date, 5, "  Today", fontsize=12, color="black", alpha=0.6, va="bottom")

    for date_str in ["2026-02-01", "2028-01-01", "2031-01-01"]:
        d = mdates.date2num(pd.Timestamp(date_str))
        v80 = np.exp(beta[0] + beta[1] * d + beta[2] * lp80 + beta[3] * d * lp80) / 1440.0
        v98 = np.exp(beta[0] + beta[1] * d + beta[2] * lp98 + beta[3] * d * lp98) / 1440.0
        ratio = v80 / v98
        mid = np.sqrt(v80 * v98)
        ax.annotate(f"{ratio:.0f}x gap",
                    (pd.Timestamp(date_str), mid),
                    fontsize=14, fontweight="bold", color="red", alpha=0.7,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8))

    ax.set_ylabel("Time horizon (days)", fontsize=14)
    ax.set_xlabel("Model release date", fontsize=14)
    ax.set_title(
        "The Jaggedness Gap: What Models Can Sometimes Do vs. What They Can Reliably Do",
        fontsize=18, fontweight="bold", pad=10,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlim(pd.Timestamp("2023-06-01"), pd.Timestamp("2031-03-01"))
    ax.set_ylim(0, 200)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved banner jaggedness to {output_path}")


def plot_doubling_time(
    beta: np.ndarray,
    output_path: str,
    percents: list[int],
):
    """Standalone figure: doubling time vs reliability from 50% to 99.99%."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_dt = {
        50: "#2166AC",
        80: "#4DAC26",
        90: "#F5A623",
        95: "#E8451E",
        98: "#9E1A8A",
        99: "#888888",
    }

    # Solid range: 50% to 99%
    p_solid = np.linspace(50, 99, 300)
    dt_solid = [doubling_time_at_p(beta, p) for p in p_solid]
    ax.plot(p_solid, dt_solid, color="steelblue", linewidth=2.5, zorder=3)

    # Dashed range: 99% to 99.99% (extrapolation)
    p_dashed = np.linspace(99, 99.99, 200)
    dt_dashed = [doubling_time_at_p(beta, p) for p in p_dashed]
    ax.plot(p_dashed, dt_dashed, color="steelblue", linewidth=2.5, linestyle="dashed",
            alpha=0.6, zorder=3)

    # Shade the extrapolation region
    ax.axvspan(99, 100, alpha=0.06, color="red", zorder=0)
    ax.text(99.5, ax.get_ylim()[0] + 1, "extrapolation\n(limited data)",
            ha="center", va="bottom", fontsize=9, color="red", alpha=0.6, style="italic")

    # Mark the fitted percents
    for p in percents:
        dt = doubling_time_at_p(beta, p)
        color = colors_dt.get(p, "steelblue")
        ax.scatter([p], [dt], color=color, s=80, zorder=5,
                   edgecolors="black", linewidths=0.5)
        # Label offset: push right for most, left for 99%
        ax.annotate(f"{p}%: {dt:.0f}d",
                    (p, dt), textcoords="offset points",
                    xytext=(10, 6), fontsize=10, fontweight="bold", color=color)

    # Mark extrapolated points
    for p in [99.9, 99.99]:
        dt = doubling_time_at_p(beta, p)
        ax.scatter([p], [dt], color="steelblue", s=60, zorder=5,
                   edgecolors="black", linewidths=0.5, marker="D")
        ax.annotate(f"{p}%: {dt:.0f}d",
                    (p, dt), textcoords="offset points",
                    xytext=(10, 6), fontsize=10, color="steelblue", style="italic")

    ax.set_xlabel("Reliability threshold (success rate %)", fontsize=12)
    ax.set_ylabel("Doubling time (days)", fontsize=12)
    ax.set_title(
        "How Doubling Time Varies with Reliability\n"
        "Higher reliability thresholds show modestly slower progress",
        fontsize=14, fontweight="bold",
    )

    # X-axis: use specific ticks to handle the nonlinear spacing
    xticks = [50, 60, 70, 80, 90, 95, 98]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x}%" for x in xticks], fontsize=10)
    ax.set_xlim(48, 100.1)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved doubling time plot to {output_path}")


def plot_r_squared(
    long: pd.DataFrame,
    output_path: str,
    percents: list[int],
):
    """Plot per-threshold R² of exponential fit vs reliability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_r2 = {
        50: "#2166AC",
        80: "#4DAC26",
        90: "#F5A623",
        95: "#E8451E",
        98: "#9E1A8A",
        99: "#888888",
    }

    r2_values = []
    for p in percents:
        sub = long[long["percent"] == p]
        x = sub["date_num"].values
        y = sub["log_horizon"].values
        A = np.column_stack([np.ones(len(x)), x])
        beta_fit, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        y_hat = A @ beta_fit
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        r2_values.append(r2)

    # Plot line
    ax.plot(percents, r2_values, color="steelblue", linewidth=2.5, zorder=3, marker="o", markersize=0)

    # Plot colored dots with labels
    for p, r2 in zip(percents, r2_values):
        color = colors_r2.get(p, "steelblue")
        ax.scatter([p], [r2], color=color, s=100, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(f"{r2:.2f}",
                    (p, r2), textcoords="offset points",
                    xytext=(10, -4), fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Reliability threshold (success rate %)", fontsize=12)
    ax.set_ylabel("R² of exponential fit (log-linear regression)", fontsize=12)
    ax.set_title(
        "Goodness of Fit Decreases at Higher Reliability\n"
        "Per-threshold R² of log(time horizon) ~ release date",
        fontsize=14, fontweight="bold",
    )

    xticks = [50, 60, 70, 80, 90, 95, 98]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x}%" for x in xticks], fontsize=10)
    ax.set_xlim(45, 101)
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved R² plot to {output_path}")


def predict_date(beta: np.ndarray, p: float, target_horizon_min: float) -> float:
    """Predict the date_num when the trendline reaches target_horizon at reliability p."""
    lp = logit(p / 100.0)
    log_h = np.log(target_horizon_min)
    # log_h = β0 + β1*d + β2*lp + β3*d*lp
    # log_h = (β0 + β2*lp) + (β1 + β3*lp)*d
    # d = (log_h - β0 - β2*lp) / (β1 + β3*lp)
    slope = beta[1] + beta[3] * lp
    return (log_h - beta[0] - beta[2] * lp) / slope


def days_until(beta: np.ndarray, p: float, target_horizon_min: float, ref_date_num: float) -> float:
    """Days from ref_date until trendline predicts target_horizon at reliability p."""
    return predict_date(beta, p, target_horizon_min) - ref_date_num


def format_horizon(minutes: float) -> str:
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    if minutes < 60:
        return f"{minutes:.0f}m"
    if minutes < 1440:
        return f"{minutes/60:.0f}h"
    return f"{minutes/1440:.0f}d"


def format_days(d: float) -> str:
    if d <= 0:
        return "now"
    if d < 365:
        return f"{d:.0f}d"
    return f"{d/365:.1f}y"


def plot_calculator(
    beta: np.ndarray,
    r_squared: float,
    output_path: str,
    ref_date: str = "2026-02-05",
):
    """Heatmap: reliability on x-axis (left to right), horizon on y-axis (increasing down).
    Includes isoclines for key dates."""
    ref_date_num = mdates.date2num(pd.Timestamp(ref_date))

    # Reliability on x-axis (increasing right = harder)
    reliabilities = [50, 60, 70, 80, 85, 90, 93, 95, 97, 98, 99, 99.9, 99.99]

    # Horizon on y-axis (increasing down = longer tasks)
    horizons_min = [1, 5, 15, 30, 60, 120, 240, 480, 960, 1920, 4800, 9600, 20160, 40320, 80640]
    horizon_labels = [format_horizon(h) for h in horizons_min]

    # Build grid: rows = horizons, cols = reliabilities
    grid = np.zeros((len(horizons_min), len(reliabilities)))
    for i, h in enumerate(horizons_min):
        for j, p in enumerate(reliabilities):
            grid[i, j] = days_until(beta, p, h, ref_date_num)

    fig, ax = plt.subplots(figsize=(14, 9))

    display_grid = np.clip(grid.copy(), 0, None)

    im = ax.imshow(
        display_grid,
        cmap="RdYlGn_r",
        aspect="auto",
        vmin=0,
        vmax=3000,
    )

    # Label each cell
    for i in range(len(horizons_min)):
        for j in range(len(reliabilities)):
            val = grid[i, j]
            if val <= 0:
                text = "TODAY"
                color = "white"
                weight = "bold"
            else:
                target_date = mdates.num2date(ref_date_num + val)
                text = f"{format_days(val)}\n{target_date.strftime('%b %Y')}"
                color = "white" if val > 1500 else "black"
                weight = "normal"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=color, fontweight=weight)

    # Isoclines: contour lines for key year boundaries
    # We need continuous coordinates for contour
    X, Y = np.meshgrid(range(len(reliabilities)), range(len(horizons_min)))
    isocline_years = [0, 1, 2, 3, 5]
    isocline_days = [y * 365 for y in isocline_years]
    isocline_labels = ["Now", "+1 year", "+2 years", "+3 years", "+5 years"]

    contours = ax.contour(
        X, Y, display_grid,
        levels=isocline_days,
        colors="black",
        linewidths=2.5,
        linestyles="solid",
        alpha=0.8,
    )
    ax.clabel(contours, fmt={d: l for d, l in zip(isocline_days, isocline_labels)},
              fontsize=11, inline=True, inline_spacing=12,
              use_clabeltext=True)
    # Bold the contour labels with white background for readability
    for txt in ax.texts:
        if txt.get_text() in isocline_labels:
            txt.set_fontweight("bold")
            txt.set_bbox(dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="black", alpha=0.85, linewidth=0.5))

    ax.set_xticks(range(len(reliabilities)))
    ax.set_xticklabels([f"{p}%" for p in reliabilities], fontsize=11)
    ax.set_yticks(range(len(horizons_min)))
    ax.set_yticklabels(horizon_labels, fontsize=11)

    ax.set_xlabel("Target reliability (success rate) →", fontsize=13)
    ax.set_ylabel("Target time horizon (human-equivalent task length)", fontsize=13)
    ax.set_title(
        f"When Will AI Agents Reach Each Capability Target?\n"
        f"Projected from 2D regression (R²={r_squared:.3f}), extrapolating from {ref_date}",
        fontsize=14, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Days from now", fontsize=11)

    plt.tight_layout()
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved calculator to {output_path}")


def plot_jaggedness(
    beta: np.ndarray,
    r_squared: float,
    output_path: str,
    long: pd.DataFrame,
    percents: list[int],
):
    """Plot how the gap between low and high reliability evolves over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    date_range = pd.date_range("2023-01-01", "2027-06-01", freq="MS")
    date_nums = mdates.date2num(date_range)

    p_lo, p_hi = 50, 99
    lp_lo = logit(p_lo / 100.0)
    lp_hi = logit(p_hi / 100.0)
    dlp = lp_lo - lp_hi
    log_ratio = (beta[2] + beta[3] * date_nums) * dlp
    ratio = np.exp(log_ratio)

    ax.plot(date_range, ratio, color="steelblue", linewidth=2.5,
            label=f"h({p_lo}%) / h({p_hi}%)")
    ax.fill_between(date_range, 1, ratio, alpha=0.15, color="steelblue")

    # Scatter actual ratios
    sub_lo = long[long["percent"] == p_lo].set_index("agent")
    sub_hi = long[long["percent"] == p_hi].set_index("agent")
    common = sub_lo.index.intersection(sub_hi.index)
    if len(common) > 0:
        ratios = sub_lo.loc[common, "horizon_min"] / sub_hi.loc[common, "horizon_min"]
        dates = sub_lo.loc[common, "release_date"]
        ax.scatter(dates, ratios, color="steelblue", s=50, alpha=0.6, zorder=5,
                   edgecolors="black", linewidths=0.5, label="Actual per-model ratio")

    # Annotate key dates
    for yr in [2023, 2024, 2025, 2026, 2027]:
        d = mdates.date2num(pd.Timestamp(f"{yr}-07-01"))
        r = np.exp((beta[2] + beta[3] * d) * dlp)
        ax.annotate(f"{r:.0f}x", (pd.Timestamp(f"{yr}-07-01"), r),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=11, ha="center", fontweight="bold", color="steelblue")

    ax.set_yscale("log")
    ax.set_ylabel("Ratio: 50% horizon / 99% horizon", fontsize=12)
    ax.set_xlabel("Model release date", fontsize=12)
    ax.set_title(
        "The Jaggedness Gap: what models can sometimes do vs. reliably do\n"
        "Ratio of 50% time horizon to 99% time horizon over time",
        fontsize=13,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2027-06-01"))
    plt.tight_layout()

    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved jaggedness plot to {output_path}")


def print_calculator(beta: np.ndarray, ref_date: str = "2026-02-05"):
    """Print a text table of predictions."""
    ref_date_num = mdates.date2num(pd.Timestamp(ref_date))

    horizons = [5, 15, 30, 60, 120, 240, 480, 960]
    reliabilities = [50, 80, 90, 95, 98, 99]

    header = f"{'':>6}" + "".join(f"{format_horizon(h):>12}" for h in horizons)
    print(f"\nProjected days until target (from {ref_date}):")
    print(header)
    print("-" * len(header))
    for p in reliabilities:
        row = f"{p:>5}%"
        for h in horizons:
            d = days_until(beta, p, h, ref_date_num)
            if d <= 0:
                row += f"{'DONE':>12}"
            else:
                target = mdates.num2date(ref_date_num + d)
                row += f"{format_days(d) + ' (' + target.strftime('%b %y') + ')':>12}"
            row += ""
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Resolve defaults relative to this report so the script is reproducible even when
    # invoked from the repo root instead of from reports/time-horizon-1-1/.
    parser.add_argument(
        "--input-file",
        default=str(REPORT_DIR / "data/wrangled/logistic_fits/headline.csv"),
    )
    parser.add_argument(
        "--output-file",
        default=str(REPORT_DIR / "plots/logistic/2d_regression.png"),
    )
    parser.add_argument(
        "--calculator-output",
        default=str(REPORT_DIR / "plots/logistic/horizon_calculator.png"),
    )
    parser.add_argument(
        "--jaggedness-output",
        default=str(REPORT_DIR / "plots/logistic/jaggedness_gap.png"),
    )
    parser.add_argument(
        "--hero-output",
        default=str(REPORT_DIR / "plots/logistic/hero_all_thresholds.png"),
    )
    parser.add_argument("--percents", nargs="+", type=int, default=[50, 80, 90, 95, 98, 99])
    parser.add_argument("--ref-date", default="2026-02-05")
    args = parser.parse_args()

    long = load_and_reshape(args.input_file, args.percents, exclude_agents=EXCLUDE_AGENTS)
    result = fit_2d(long)
    beta = result["beta"]

    print(f"Coefficients: β0={beta[0]:.4f}, β1={beta[1]:.6f}, β2={beta[2]:.4f}, β3={beta[3]:.8f}")
    print(f"R² = {result['r_squared']:.4f}, N = {result['n']}")
    print()
    for p in args.percents:
        dt = doubling_time_at_p(beta, p)
        print(f"  p={p}%: doubling time = {dt:.1f} days")

    plot_results(long, beta, result["r_squared"], args.output_file, args.percents)
    print_calculator(beta, args.ref_date)
    plot_calculator(beta, result["r_squared"], args.calculator_output, args.ref_date)
    plot_jaggedness(beta, result["r_squared"], args.jaggedness_output, long, args.percents)
    plot_hero(long, beta, result["r_squared"], args.hero_output, args.percents)
    plot_jaggedness_hero(long, beta, str(REPORT_DIR / "plots/logistic/jaggedness_hero.png"))
    plot_banner_heatmap(beta, result["r_squared"], str(REPORT_DIR / "plots/logistic/banner_heatmap.png"))
    plot_banner_jaggedness(long, beta, str(REPORT_DIR / "plots/logistic/banner_jaggedness.png"))
    plot_doubling_time(beta, str(REPORT_DIR / "plots/logistic/doubling_time_vs_reliability.png"), args.percents)
    plot_r_squared(long, str(REPORT_DIR / "plots/logistic/r_squared_vs_reliability.png"), args.percents)


if __name__ == "__main__":
    main()
