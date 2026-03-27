# AI Jaggedness in the Agentic Era

## Where we are today

AI agents are everywhere. They write code, debug pipelines, answer questions, and scaffold entire workflows. But anyone who has used them for real work knows the gap between what they *can* do and what they *reliably* do. A model might nail an 8-hour coding task on one attempt and completely flail on the next. This is jaggedness -- the inconsistency between peak capability and dependable performance.

Using METR's time horizon benchmark, we can put numbers to this. The time horizon measures the longest task (in human-equivalent time) that a model can complete at a given success rate, across 228 software engineering, cybersecurity, reasoning, and ML tasks.

Today's best models (Claude Opus 4.6, GPT-5.2) have a 50% time horizon of roughly **12 hours** -- meaning there exist half-day tasks they can sometimes complete autonomously. But their 99% time horizon is about **1 minute**. The longest task they almost always get right is trivially short.

Models are useful and heavily deployed. But they are not yet truly autonomous or reliable for even mid-sized tasks. The frontier of what's *possible* is exciting; the frontier of what's *dependable* is sobering.

## Expanding the lens: reliability as a continuous dimension

The standard time horizon analysis fits separate exponential trendlines at fixed reliability thresholds -- typically 50% and 80%. We extended this by fitting a single 2D regression across both release date and reliability level:

```
log(time_horizon) = B0 + B1 * date + B2 * logit(p) + B3 * date * logit(p)
```

where `p` is the success rate and `logit(p) = log(p/(1-p))`. This jointly estimates how capability grows over time at any reliability level, using a single model with R^2 = 0.93.

The results across six thresholds:

| Reliability | Doubling Time | Current Best |
|------------|--------------|-------------|
| 50%        | 130 days     | ~12 hours   |
| 80%        | 135 days     | ~70 min     |
| 90%        | 138 days     | ~25 min     |
| 95%        | 142 days     | ~11 min     |
| 98%        | 146 days     | ~4 min      |
| 99%        | 149 days     | ~1 min      |

![All thresholds on one plot](../plots/logistic/all_thresholds.png)

Two things stand out. First, the doubling times are remarkably close -- all between 130 and 149 days. Progress is exponential at every reliability level, and the rates are broadly similar. Second, the doubling times do increase as reliability rises. Not dramatically, but measurably: the gap between 50% and 99% is about 19 days of doubling time, or roughly 15%.

That 15% matters because exponentials diverge. A small difference in growth rate, compounded over years, produces a large and growing gap between what models can sometimes do and what they can reliably do.

## The jaggedness gap is widening

We can measure this directly. The ratio between the 50% time horizon and the 99% time horizon -- a single number capturing how "jaggy" a model is -- has been growing:

- **2023**: ~200x
- **2025**: ~300x
- **2027** (projected): ~420x

![Jaggedness gap over time](../plots/logistic/jaggedness_gap.png)

This isn't because reliability is getting *worse*. Every threshold is improving exponentially. But the frontier of peak capability is outrunning the frontier of dependable performance. Each new generation of models unlocks longer tasks at 50% success before that capability becomes robust at 99%.

Concretely: models are racing ahead at "impressive demos" faster than at "production-grade reliability."

## But does jaggedness even matter?

Here's where it gets interesting. The jaggedness gap is widening, but everything is still exponential. If these trends hold, every reliability level -- even extreme ones -- eventually catches up to any fixed capability target. The question isn't *whether* reliability arrives, but *when*.

We built a calculator to project exactly that:

| Target | 50% | 90% | 99% | 99.9% | 99.99% |
|--------|-----|-----|-----|-------|--------|
| **1h tasks** | DONE | Aug 2026 | Jun 2028 | Jul 2030 | Jan 2033 |
| **4h tasks** | DONE | May 2027 | Apr 2029 | Jun 2031 | Dec 2033 |
| **8h tasks** | Mar 2026 | Sep 2027 | Sep 2029 | Nov 2031 | Jun 2034 |
| **16h tasks** | Jul 2026 | Feb 2028 | Feb 2030 | Apr 2032 | Dec 2034 |

![Calculator heatmap](../plots/logistic/horizon_calculator.png)

Each additional nine of reliability costs roughly 2-2.5 years. The pattern is strikingly regular.

If you're building products that need 90% reliability on hour-long tasks, the trendline says late 2026. If you need 99% reliability on 8-hour tasks -- genuine full-day autonomous work -- that's late 2029. If you need five nines on anything non-trivial, you're looking at the early 2030s.

Of course, these are extrapolations. They assume the current exponential continues -- no saturation, no acceleration, no architectural breakthroughs that specifically target reliability. Any of those could change the picture. But the framework gives us a structured way to think about the timeline, and a baseline to measure against.

## What this means

The jaggedness gap frames a tension in the current moment. AI agents are capable enough to be transformative in workflows where occasional failure is tolerable -- coding assistance, research, exploration, drafting. They are not yet reliable enough for workflows that demand consistency -- production systems, safety-critical applications, fully autonomous operation.

The good news: reliability is improving exponentially, just like capability. The sobering news: it's improving slightly slower, so the gap between "what's possible" and "what's dependable" will likely keep growing before it stabilizes.

For practitioners, this suggests a near-term strategy: design systems that exploit the high-capability frontier (human-in-the-loop, retry logic, verification layers) while waiting for the reliability frontier to catch up. The calculator above gives a rough timeline for when that catch-up arrives at various thresholds.

For researchers, the interaction term in our 2D model (B3 < 0) is a measurable target. Techniques that specifically close the jaggedness gap -- improving reliability without just scaling capability -- would show up as B3 trending toward zero. That's a concrete metric for a problem that has so far been discussed mostly in qualitative terms.

## Methodology

We fit a joint OLS regression on log(time_horizon) using METR's Time Horizon 1.1 benchmark data (15 frontier models, 6 reliability thresholds, N=90 observations). The reliability dimension is parameterized via logit(p), which provides a natural and smooth mapping for success probabilities. The model includes a date-by-reliability interaction term that captures the differential in growth rates across thresholds. Full code and figures are available in this repository.

**Caveats**: Projections assume continued exponential progress. Predictions beyond 99% reliability extrapolate past the training range. Results are specific to METR's task suite. The 2D regression is a model fitted to outputs of per-agent logistic regressions -- it is a regression on regressions.
