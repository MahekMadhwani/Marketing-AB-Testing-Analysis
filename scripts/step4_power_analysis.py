"""
STEP 4: Power Analysis
Was the experiment big enough to reliably detect the effect we found?
"""

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# Values carried over from Step 3
n_ad, n_psa = 564577, 23524
conv_ad, conv_psa = 14423, 420
p_ad = conv_ad / n_ad
p_psa = conv_psa / n_psa

power_analysis = NormalIndPower()

# =========================================================
# 1. Effect size for this comparison (Cohen's h)
# =========================================================
effect_size = proportion_effectsize(p_ad, p_psa)
print(f"Observed proportions: p_ad={p_ad:.4f}, p_psa={p_psa:.4f}")
print(f"Effect size (Cohen's h): {effect_size:.4f}")

# =========================================================
# 2. Achieved (post-hoc) power at the ACTUAL sample sizes used
#    statsmodels ratio = nobs2 / nobs1
# =========================================================
ratio = n_psa / n_ad
achieved_power = power_analysis.power(
    effect_size=effect_size, nobs1=n_ad, alpha=0.05, ratio=ratio
)
print(f"\n--- Achieved power at actual sample sizes ---")
print(f"n_ad={n_ad}, n_psa={n_psa} (ratio psa/ad = {ratio:.4f})")
print(f"Achieved power: {achieved_power:.6f}  ({achieved_power*100:.2f}%)")

# =========================================================
# 3. Required sample size PER GROUP for 80% power, if the
#    experiment had used a balanced 50/50 split
# =========================================================
required_n_balanced = power_analysis.solve_power(
    effect_size=effect_size, alpha=0.05, power=0.80, ratio=1.0
)
print(f"\n--- Required sample size (balanced 50/50 design, 80% power) ---")
print(f"Required n per group: {required_n_balanced:.0f}")
print(f"Total required (both groups): {required_n_balanced*2:.0f}")
print(f"Actual total used: {n_ad + n_psa}")

# =========================================================
# 4. Required n_psa (control group) for 80% power, GIVEN the
#    actual n_ad and the actual imbalance ratio the experiment used
# =========================================================
required_n_ad_given_ratio = power_analysis.solve_power(
    effect_size=effect_size, alpha=0.05, power=0.80, ratio=ratio
)
print(f"\n--- Required n_ad for 80% power, at the ACTUAL ad:psa ratio used ---")
print(f"Required n_ad: {required_n_ad_given_ratio:.0f}")
print(f"Actual n_ad used: {n_ad}")

# =========================================================
# 5. Minimum Detectable Effect (MDE): given the actual n_ad and
#    ratio used, what is the smallest effect size we COULD have
#    reliably detected at 80% power?
# =========================================================
mde_effect_size = power_analysis.solve_power(
    nobs1=n_ad, alpha=0.05, power=0.80, ratio=ratio
)
print(f"\n--- Minimum Detectable Effect (given actual sample sizes) ---")
print(f"Minimum detectable Cohen's h at 80% power: {mde_effect_size:.4f}")
print(f"Observed Cohen's h was: {effect_size:.4f}  (comfortably above the minimum)")

# Convert MDE effect size back to an approximate proportion difference for intuition
# using p_psa as the baseline
from scipy.optimize import brentq

def h_from_p2(p2, p1_baseline, target_h):
    return (2*np.arcsin(np.sqrt(p1_baseline)) - 2*np.arcsin(np.sqrt(p2))) - target_h

# solve for the ad conversion rate that would produce exactly the MDE, holding psa fixed
try:
    p_ad_mde = brentq(lambda p: h_from_p2(p, p_psa, -mde_effect_size), 0.0001, 0.5)
    mde_pp = (p_ad_mde - p_psa) * 100
    print(f"Approx. minimum detectable absolute lift: {mde_pp:.4f} percentage points")
except Exception as e:
    print("Could not invert MDE to percentage points:", e)

# =========================================================
# Verdict
# =========================================================
print(f"\n=== VERDICT ===")
if achieved_power >= 0.80:
    print(f"Achieved power ({achieved_power*100:.2f}%) exceeds the 80% convention.")
    print("The experiment was more than adequately powered to detect this effect.")
else:
    print(f"Achieved power ({achieved_power*100:.2f}%) is below the 80% convention.")
    print("The experiment may have been underpowered.")
