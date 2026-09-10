"""
STEP 3: Core Hypothesis Test
Two-proportion z-test + Chi-square test: Ad vs PSA conversion rate
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

df = pd.read_csv("marketing_AB_cleaned.csv")

# =========================================================
# 1. Build contingency table
# =========================================================
contingency = pd.crosstab(df["test_group"], df["converted"])
print("Contingency table (rows=group, cols=converted):")
print(contingency)

# =========================================================
# 2. Chi-square test of independence
# =========================================================
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print(f"\n--- Chi-square test ---")
print(f"Chi2 statistic: {chi2:.4f}")
print(f"p-value: {p_chi2:.10f}")
print(f"Degrees of freedom: {dof}")

# =========================================================
# 3. Two-proportion z-test
# =========================================================
n_ad = df[df["test_group"] == "ad"].shape[0]
n_psa = df[df["test_group"] == "psa"].shape[0]
conv_ad = df[df["test_group"] == "ad"]["converted"].sum()
conv_psa = df[df["test_group"] == "psa"]["converted"].sum()

count = np.array([conv_ad, conv_psa])
nobs = np.array([n_ad, n_psa])

z_stat, p_z = proportions_ztest(count, nobs, alternative="two-sided")
print(f"\n--- Two-proportion z-test ---")
print(f"n (ad): {n_ad}, conversions: {conv_ad}, rate: {conv_ad/n_ad*100:.4f}%")
print(f"n (psa): {n_psa}, conversions: {conv_psa}, rate: {conv_psa/n_psa*100:.4f}%")
print(f"Z statistic: {z_stat:.4f}")
print(f"p-value: {p_z:.10f}")

# =========================================================
# 4. Confidence interval for the difference in proportions
# =========================================================
p_ad = conv_ad / n_ad
p_psa = conv_psa / n_psa
diff = p_ad - p_psa

se_diff = np.sqrt(p_ad*(1-p_ad)/n_ad + p_psa*(1-p_psa)/n_psa)
ci_low = diff - 1.96 * se_diff
ci_high = diff + 1.96 * se_diff

print(f"\n--- Confidence interval (95%) for difference in conversion rate ---")
print(f"Absolute difference: {diff*100:.4f} percentage points")
print(f"95% CI: [{ci_low*100:.4f}%, {ci_high*100:.4f}%]")

# Individual group CIs (Wilson score interval - more robust for rare events)
ci_ad = proportion_confint(conv_ad, n_ad, alpha=0.05, method="wilson")
ci_psa = proportion_confint(conv_psa, n_psa, alpha=0.05, method="wilson")
print(f"\nAd group 95% CI: [{ci_ad[0]*100:.4f}%, {ci_ad[1]*100:.4f}%]")
print(f"PSA group 95% CI: [{ci_psa[0]*100:.4f}%, {ci_psa[1]*100:.4f}%]")

# =========================================================
# 5. Effect size (Cohen's h - standard for comparing two proportions)
# =========================================================
def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

h = cohens_h(p_ad, p_psa)
print(f"\n--- Effect size ---")
print(f"Cohen's h: {h:.4f}")
if abs(h) < 0.2:
    magnitude = "small"
elif abs(h) < 0.5:
    magnitude = "medium"
else:
    magnitude = "large"
print(f"Interpretation: {magnitude} effect size")

# Relative lift (business-friendly framing)
relative_lift = (p_ad - p_psa) / p_psa * 100
print(f"Relative lift: ads convert {relative_lift:.2f}% better than PSA")

# =========================================================
# 6. Final verdict
# =========================================================
alpha = 0.05
print(f"\n=== VERDICT ===")
if p_z < alpha:
    print(f"p-value ({p_z:.2e}) < alpha ({alpha}) --> REJECT the null hypothesis.")
    print("The difference in conversion rate between ad and psa groups is statistically significant.")
else:
    print(f"p-value ({p_z:.4f}) >= alpha ({alpha}) --> FAIL TO REJECT the null hypothesis.")
    print("No statistically significant difference detected.")
