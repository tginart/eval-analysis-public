# 2D Reliability Regression Analysis

## Overview

`regress_2d_horizon.py` fits a joint regression across both model release date and reliability threshold to analyze how AI agent time horizons evolve.

## Model

```
log(time_horizon) = β0 + β1 · date + β2 · logit(p) + β3 · date · logit(p)
```

where:
- `p` is the success rate (reliability threshold)
- `logit(p) = log(p / (1-p))` maps probabilities to (-∞, ∞)
- `date` is the model release date in matplotlib date numbers (days)

For a fixed reliability `p`, this reduces to:
- **Effective intercept**: `β0 + β2 · logit(p)`
- **Effective slope**: `β1 + β3 · logit(p)`
- **Doubling time**: `ln(2) / (β1 + β3 · logit(p))`

The interaction term `β3 < 0` means higher reliability thresholds have slower doubling times — the jaggedness gap widens over time.

## Data

- **Source**: METR Time Horizon 1.1 benchmark, `headline` logistic fits
- **Observations**: 15 frontier models × 6 reliability thresholds (50, 80, 90, 95, 98, 99%) = 90 data points
- **Excluded agents** (non-frontier or superseded, consistent with main analysis):
  - Claude 3 Opus (Inspect)
  - GPT-4 Turbo (Inspect)
  - Claude 4 Opus (Inspect)
  - GPT-5.1-Codex-Max (Inspect)
  - Claude 4.1 Opus (Inspect)

## Results

- **R² = 0.93** on the joint model
- Doubling times: 130d (50%) → 135d (80%) → 138d (90%) → 142d (95%) → 146d (98%) → 149d (99%)
- Per-threshold R² ranges from 0.94 (50%) to 0.71 (99%)

## Generated figures

| Figure | Description |
|--------|-------------|
| `hero_all_thresholds.png` | Fan of trendlines at 50-99% with scatter, solid/dashed for fit/projection |
| `doubling_time_vs_reliability.png` | Doubling time as continuous function of reliability, with extrapolation to 99.99% |
| `r_squared_vs_reliability.png` | Per-threshold R² showing decreasing fit quality at higher reliability |
| `horizon_calculator.png` | Heatmap of projected dates for (reliability, task-length) milestones with isoclines |
| `2d_regression.png` | Two-panel: trendlines + doubling time curve |
| `jaggedness_gap.png` | 50%/99% horizon ratio over time |

## Usage

```bash
cd reports/time-horizon-1-1
python3 scripts/regress_2d_horizon.py
```

Options:
- `--input-file`: path to logistic fits CSV (default: `data/wrangled/logistic_fits/headline.csv`)
- `--percents`: reliability thresholds to fit (default: `50 80 90 95 98 99`)
- `--ref-date`: reference date for calculator projections (default: `2026-02-05`)

## Signal-to-noise ratio across reliability levels

An important consideration: the exponential fit quality degrades at higher reliability thresholds.

| Reliability | R² |
|------------|-----|
| 50% | 0.94 |
| 80% | 0.91 |
| 90% | 0.87 |
| 95% | 0.83 |
| 98% | 0.77 |
| 99% | 0.71 |

At 50%, the exponential fit explains 94% of the variance — a strong fit. By 99%, that drops to 71%. That's still a real, meaningful trend, but there's substantially more scatter. This makes intuitive sense: at 50% we're measuring near the inflection point of each model's success curve, where the signal is cleanest. At 99% we're in the tails, where noise dominates.

This means projections at 50-90% should be taken more seriously than those at 99%+. Extrapolations to 99.9% or 99.99% are directionally informative but carry wide implicit error bars. As we get more reliable models and larger task pools, we should be able to get cleaner fits for higher reliabilities.

See `r_squared_vs_reliability.png` for the visualization.

## Caveats

- Projections assume continued exponential progress — no saturation or acceleration
- Predictions beyond 99% reliability extrapolate past the training range
- Results are specific to METR's task suite (228 SWE, cybersecurity, reasoning, and ML tasks)
- The 2D regression is a model fitted to outputs of per-agent logistic regressions — a regression on regressions
