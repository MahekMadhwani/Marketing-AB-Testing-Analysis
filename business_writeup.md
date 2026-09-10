# Marketing Campaign A/B Test: Business Report

## Executive Summary

We tested whether showing users an advertisement ("ad") drives more conversions than showing a Public Service Announcement ("psa", the control). Across 588,101 users, **ads produced a 43% relative lift in conversion rate** (2.55% vs 1.79%), a result that is both statistically significant and, using a Bayesian analysis, essentially certain (>99.99% probability that ads are genuinely better).

**Recommendation: Continue showing ads. The evidence for this decision is about as strong as real-world experimentation gets.**

---

## 1. The Business Question

Does replacing a Public Service Announcement with a paid advertisement increase the rate at which users take the target conversion action?

## 2. The Experiment

- **588,101 users** were split into two groups: 564,577 saw ads, 23,524 saw a PSA (a real, if unbalanced, randomized experiment)
- Each user's exposure (number of ads/PSAs seen, day, hour) and whether they converted was recorded
- Data was clean: no missing values, no duplicates

## 3. Key Findings

### Finding 1: Ads significantly outperform the PSA
| Group | Conversion Rate | 95% Confidence Interval |
|---|---|---|
| Ad | 2.55% | [2.51%, 2.60%] |
| PSA | 1.79% | [1.62%, 1.96%] |

The gap (0.77 percentage points) is statistically significant (p < 0.0001) and translates to a **43% relative improvement** in conversions.

**Important nuance:** the *statistical* effect size (Cohen's h = 0.05) is technically "small" — with over half a million users, even tiny differences become statistically detectable. What makes this result matter isn't the raw percentage-point gap, it's the relative lift: almost double the conversion rate is a meaningful business outcome regardless of how "small" it looks in standardized statistical terms.

### Finding 2: The experiment was more than adequately sized
A sample of just ~70,000 users (12% of what was actually used) would have been enough to detect this effect reliably. The experiment could have reliably caught an effect less than a third the size of what was actually found. Practically, this means the company could run smaller, faster, cheaper tests in the future and still trust the results.

### Finding 3: The ad advantage is consistent, not a fluke of one segment
Across every day of the week and 23 of 24 hours of the day, ads outperformed the PSA. There is no evidence of a hidden segment where the PSA actually wins (no Simpson's Paradox). This consistency is what makes the result trustworthy enough to act on company-wide, rather than only in a specific time window.

### Finding 4: Conversion is strongly tied to how many ads/PSAs a user was shown
Users shown 80-160 ads convert at roughly 60x the rate of users shown 1-5 ads. This relationship saturates — beyond ~150-200 exposures, additional ads stop adding conversion lift.

**Caveat worth stating plainly:** this is very likely partly a measurement artifact, not pure causation. Users who convert quickly are, by definition, exposed to fewer total ads (delivery likely stops after conversion). This means "more ads seen" is partly a symptom of "hasn't converted yet," not purely a cause of conversion. This finding is best read as: *heavy exposure and conversion are strongly linked*, not *showing more ads causes more sales*, without further data on exposure timing relative to conversion.

### Finding 5: A predictive model confirms exposure volume as the dominant signal
A machine learning model built to predict conversion (ROC-AUC 0.86) confirms that exposure volume is by far the strongest predictor of conversion — more so than which group a user was in, or what day/hour they were exposed. This is consistent with Finding 4 and reinforces the same causality caveat.

### Finding 6: A Bayesian analysis puts a number on our confidence
Rather than relying only on a p-value, we directly estimated the probability that ads are truly better than the PSA:
- **P(ad beats psa) ≈ 100%**
- **Most likely lift: 43%** (95% credible range: 30% to 58%)
- **Expected cost of wrongly choosing PSA:** 0.77 percentage points of lost conversion
- **Expected cost of wrongly choosing ads (if we're somehow wrong):** effectively zero

This framing directly answers the business question ("how likely is it that we're making the right call, and what do we risk either way?") more directly than a p-value does.

## 4. Limitations

- This dataset has no demographic, revenue, or cost information — we can say ads convert more often, but not whether they're more *profitable* once ad spend is factored in
- The exposure-count relationship (Finding 4) is likely confounded by reverse causality, as noted above
- This is a single, time-boxed snapshot — no data on longer-term retention after conversion

## 5. Final Recommendation

**Roll out ads over the PSA.** The statistical evidence, the consistency across every time segment, and the Bayesian probability all point the same direction with very little ambiguity. Before scaling further, it would be worth instrumenting cost/revenue data so a future analysis can speak to ROI, not just conversion rate.
